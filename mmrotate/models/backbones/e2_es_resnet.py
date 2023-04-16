import torch
import torch.nn as nn
import e2cnn.nn as G
from .impl.ses_conv import SESNN

class E2SESResNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(E2SESResNet, self).__init__()

        # 使用SESNN构建卷积层
        self.sesnn = SESNN(in_channels=in_channels, out_channels=out_channels)

        # 使用E2CNN构建ResNet的基本块
        self.e2cnn_block = G.SequentialModule(
            G.Rot2dOnR2(N=8),
            G.R2Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            G.Rot2dOnR2(N=8),
            G.R2Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

        # 使用E2CNN构建整个ResNet
        self.e2cnn_resnet = G.SequentialModule(
            *[G.SequentialModule(
                *[self.e2cnn_block for _ in range(num_blocks)]
            ) for _ in range(4)]
        )

    def forward(self, x):
        # 使用SESNN进行放缩和旋转等变换
        x = self.sesnn(x)

        # 使用E2CNN构建ResNet
        x = self.e2cnn_resnet(x)

        return x
