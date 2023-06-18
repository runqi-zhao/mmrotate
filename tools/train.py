# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed

from mmrotate.apis import train_detector
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import collect_env, get_root_logger, setup_multi_processes


# =========== optional arguments ===========
# --work-dir        存储日志和模型的目录
# --resume-from     加载 checkpoint 的目录
# --no-validate     是否在训练的时候进行验证
# 互斥组：
#   --gpus          使用的 GPU 数量
#   --gpu_ids       使用指定 GPU 的 id
# --seed            随机数种子
# --deterministic   是否设置 cudnn 为确定性行为
# --options         其他参数
# --launcher        分布式训练使用的启动器，可以为：['none', 'pytorch', 'slurm', 'mpi']
#                   none：不启动分布式训练，dist_train.sh 中默认使用 pytorch 启动。
# --local_rank      本地进程编号，此参数 torch.distributed.launch 会自动传入。

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # 从文件读取配置
    cfg = Config.fromfile(args.config)
    # 从命令行读取额外的配置
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    # 设置 cudnn_benchmark = True 可以加速输入大小固定的模型. 如：SSD300
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir 的优先程度为: 命令行 > 配置文件
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    # 当 work_dir 为 None 的时候, 使用 ./work_dir/配置文件名 作为默认工作目录
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        # os.path.basename(path)    返回文件名
        # os.path.splitext(path)    分割路径, 返回路径名和文件扩展名的元组
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # 是否继续上次的训练
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    # gpu id
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    # 如果 launcher 为 none，不启用分布式训练。不使用 dist_train.sh 默认参数为 none.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    # launcher 不为 none，启用分布式训练。使用 dist_train.sh，会传 ‘pytorch’
    else:
        distributed = True
        # 初始化 dist 里面会调用 init_process_group

        #此函数负责调用 init_process_group，完成分布式的初始化。
        # 在运行 dist_train.py 训练时，默认传递的 launcher 是 'pytorch'。所以此函数会进一步调用 _init_dist_pytorch 来完成初始化。
        # 因为 torch.distributed 可以采用单进程控制多 GPU，也可以一个进程控制一个 GPU。
        # 一个进程控制一个 GPU 是目前Pytorch中，无论是单节点还是多节点，进行数据并行训练最快的方式。
        # 在 mmdet 中也是这么实现的。既然是单个进程控制单个 GPU，那么我么就需要绑定当前进程控制的是哪个 GPU。
        # 可以理解为在使用 torch.distributed.launch 运行 py 文件时。 它会多次调用 py 文件，每个 py 文件控制一个 GPU。
        # 并向每个 py 文件传参 --local_rank。（local_rank 是在这台机器上的本地进程编号）这样对于每个 py 文件，
        # 都能拿到传入的本地进程编号，我们只需要把当前进程绑定到指定的 GPU 即可。
        # 在 _init_dist_pytorch 中就会设置当前进程控制的默认 GPU（torch.cuda.set_device），
        # 再使用 dist.init_process_group 初始化，
        # 初始化的方式为默认的 env://，即环境变量的方式。
        # 使用 env:// 方式初始化就需要用 torch.distributed.launch 运行 py 文件，
        # torch.distributed.launch 会根据传入的参数设置环境变量，并运行 py 文件。
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    # 创建 work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # 保存 config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    # 获取 root logger。
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    # 设置随机化种子
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    # 此函数会对 python、numpy、torch 都设置随机数种子。
    # 保持随机数种子相同时，卷积的结果在CPU上相同，在GPU上仍然不相同。
    # 这是因为，cudnn卷积行为的不确定性。
    # 使用 torch.backends.cudnn.deterministic = True 可以解决。
    # cuDNN 使用非确定性算法，并且可以使用 torch.backends.cudnn.enabled = False 来进行禁用。
    # 如果设置为 torch.backends.cudnn.enabled = True，说明设置为使用非确定性算法（即会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题）
    # 一般来讲，应该遵循以下准则：
    # 如果网络的输入数据维度或类型上变化不大，
    # 设置 torch.backends.cudnn.benchmark = true 可以增加运行效率
    # 如果网络的输入数据在每次 iteration 都变化的话，
    # 会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    # 设置 torch.backends.cudnn.benchmark = False 避免重复搜索。
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    #将文件读入 同时将每个类进行初始化
    # 构建模型: 需要传入 cfg.model，cfg.train_cfg，cfg.test_cfg
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    #将数据集读入
    # 构建数据集: 需要传入 cfg.data.train，表明是训练集
    datasets = [build_dataset(cfg.data.train)]
    # workflow 代表流程：
    # [('train', 2), ('val', 1)] 就代表，训练两个 epoch 验证一个 epoch
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # 训练检测器：需要传入模型、数据集、配置参数等
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
