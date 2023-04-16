from PIL import Image
import math

from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class Mmrotate:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, results):
        # 从字典对象中提取图像信息
        a = results['img']
        img = Image.fromarray(results['img'])

        # 将角度转为弧度
        angle = math.radians(self.angle)
        # 获取输入图片的宽高
        width, height,c = results['img_shape']


        # 计算旋转后的图片大小
        new_width = int(math.ceil(width * math.fabs(math.cos(angle)) + height * math.fabs(math.sin(angle))))
        new_height = int(math.ceil(height * math.fabs(math.cos(angle)) + width * math.fabs(math.sin(angle))))

        # 创建新的空白图片，背景为白色
        new_img = Image.new('RGB', (new_width, new_height), (0, 0, 0))

        # 计算中心点坐标
        cx = width / 2
        cy = height / 2

        # 遍历新图片的每个像素
        for x in range(new_width):
            for y in range(new_height):
                # 将当前像素坐标减去中心点坐标，再旋转angle角度，最后加上中心点坐标
                nx = int(math.floor((x - cx) * math.cos(angle) + (y - cy) * math.sin(angle) + cx))
                ny = int(math.floor(-(x - cx) * math.sin(angle) + (y - cy) * math.cos(angle) + cy))

                # 判断像素是否在输入图片范围内
                if nx >= 0 and nx < width and ny >= 0 and ny < height:
                    # 获取输入图片对应像素的颜色值，并将其赋值给新图片当前像素
                    pixel = img.getpixel((nx, ny))
                    new_img.putpixel((x, y), pixel)

        # 遍历新图片的每个像素，检查是否存在空洞
        # 遍历新图片的每个像素
        for x in range(new_width):
            for y in range(new_height):
                # 将当前像素坐标减去中心点坐标，再旋转angle角度，最后加上中心点坐标
                nx = int(math.ceil((x - cx) * math.cos(angle) + (y - cy) * math.sin(angle) + cx))
                ny = int(math.ceil(-(x - cx) * math.sin(angle) + (y - cy) * math.cos(angle) + cy))

                # 判断像素是否在输入图片范围内
                if nx >= 0 and nx < width and ny >= 0 and ny < height:
                    # 获取输入图片对应像素的颜色值，并将其赋值给新图片当前像素
                    pixel = img.getpixel((nx, ny))
                    new_img.putpixel((x, y), pixel)

        # 将处理后的图像信息放回字典对象中
        results['img_shape'] = new_img

        # 返回字典对象
        return results

