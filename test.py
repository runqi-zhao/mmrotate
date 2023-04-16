# import torch
# from PIL.Image import Image
# from matplotlib import pyplot as plt
# from torchvision.transforms import functional as F
#
# # 定义用于预处理输入图像的函数
# def preprocess_image(img):
#     img = F.resize(img, (512, 512))
#     img = F.to_tensor(img)
#     img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     return img.unsqueeze(0)
#
# def diffeomorphic_matching(im1,im2,x,y):
#     im_rol, im_col = im2.shape[0], im2.shape[1]
#     # 参数设置
#     paraNb = 150
#     paraBelta = 0.5
#     sigma = 0.01
#     nbData = x.shape[0]
#     z = y
#     rho_ = np.zeros((1, paraNb))
#     p = np.zeros((paraNb, 2))
#     q = np.zeros((paraNb, 2))
#     v = np.zeros((paraNb, 2))
#
#     #微分同胚映射
#     for i in range(0,paraNb):
#         if i !=0:
#             z = z.T
#         m = np.argmax( np.sum((z - x)**2,axis=1))
#         p[i,:] = z[m,:]
#         q[i,:] = x[m,:]
#         v[i,:] = paraBelta * (q[i,:] - p[i,:])
#         up_bound = 0.5*sqrt(np.exp(1)/2)/np.linalg.norm(v[i,:], ord=2)
#
#         fun = lambda rho: np.sum(np.sum((z.T+[v[i,0] * np.exp(-rho ** 2 * np.sum(sigma * (z - p[i, :]) ** 2, axis=1)),
#                                               v[i,1] * np.exp(-rho ** 2 * np.sum(sigma * (z - p[i, :]) ** 2, axis=1))]-x.T)**2,axis=1))/nbData
#         bnds = [(0,up_bound)]
#         res = minimize(fun,0.1,method='SLSQP',bounds=bnds)   #L-BFGS-B
#         rho_[0,i]=res.x
#
#         z = z.T + [v[i,0]* np.exp(-rho_[0,i]**2 * np.sum(sigma*(z - p[i,:])**2,axis=1)),v[i,1]* np.exp(-rho_[0,i]**2 * np.sum(sigma*(z - p[i,:])**2,axis=1))]
#         print("迭代次数：第%d次" %(i))
#     print("迭代完成")
#     #构造需变换的矩阵
#     im_loc = np.zeros((2,im_rol,im_col))
#     im_change_loc = np.zeros((2,im_rol,im_col))
#
#     for rol in range(0,512):
#         for col in range(0,512):
#             im_loc[0,rol,col] = rol
#             im_loc[1,rol,col] = col
#
#     #坐标映射
#     for r_loc in range(0,512):
#         z2 = im_loc[:,r_loc,:]
#         z2 = z2.T
#         for i in range(0,paraNb):
#             if i !=0:
#                 z2 = z2.T
#             test =np.array([(v[i, 0] *np.exp(-rho_[0, i] ** 2 * np.sum(sigma * (z2 - p[i, :]) ** 2, axis=1))),
#                             (v[i, 1] *np.exp(-rho_[0, i] ** 2 * np.sum(sigma * (z2 - p[i, :]) ** 2, axis=1)))]).reshape(2,-1)
#             z2 = z2.T+ test
#
#         im_change_loc[:,r_loc,:]=z2
#
#     im_change = np.zeros((512,512,3),dtype=np.uint8)
#
#     #图像变换
#     for i in range(0,512):
#         for j in range(0,512):
#             temp = im_change_loc[:,i,j]
#             if (temp[0]>0 and temp[0]<512 and temp[1]>0 and temp[1]<512):
#                 im_change[j, i,:] = im1[int(temp[1]), int(temp[0]),:]
#
#     return im_change
#
# # 加载 OrientedRCNN 模型
# model = torch.hub.load('/opt/ml/model/code/', 'custom', source ='local', path='last.pt',force_reload=True)
#
# # 加载需要处理的图像
# image = Image.open("data/hrsc/Train/AllImages/100000001.bmp")
#
# # 预处理图像并将其输入 OrientedRCNN 模型
# input_image = preprocess_image(image)
# model.eval()
# with torch.no_grad():
#     outputs = model(input_image)
#
# # 从 OrientedRCNN 模型的输出中提取出需要的参数
# rois = outputs['instances'].pred_boxes
# pred_classes = outputs['instances'].pred_classes
# pred_rotations = outputs['instances'].pred_rotations
# features = outputs['instances'].features
#
# # 将 OrientedRCNN 模型的输出转换为 numpy 数组
# rois = rois.tensor.cpu().numpy()
# pred_rotations = pred_rotations.cpu().numpy()
# features = features.cpu().numpy()
#
# # 将图像和相关参数输入你的微分同胚算法中，得到处理后的图像
# result_image = diffeomorphic_matching(image, pred_rotations, rois)
#
# # 展示处理后的图像
# plt.imshow(result_image)
# plt.show()

import torch
import torch.nn.functional as F

def half_circle_kernel(radius, angle):
    """创建一个半圆形的卷积核"""
    size = radius * 2 + 1
    kernel = torch.zeros(size, size)
    center = radius, radius
    for i in range(size):
        for j in range(size):
            if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2:
                if angle <= 180:
                    if j < center[1]:
                        kernel[i, j] = 1
                else:
                    if j > center[1]:
                        kernel[i, j] = 1
    return kernel

def half_circle_conv2d(input, radius, angle):
    """半圆卷积"""
    kernel = half_circle_kernel(radius, angle)
    # 将卷积核按照指定角度旋转
    kernel = torch.from_numpy(np.rot90(kernel.numpy(), int(angle / 45), axes=(0, 1))).float()
    # 添加一个通道维度，以适应torch的卷积函数
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    # 执行卷积操作
    output = F.conv2d(input.unsqueeze(0), kernel, padding=radius)
    return output.squeeze(0)

# 示例
input = torch.rand(3, 5, 5)  # 3个5x5的随机张量
output = half_circle_conv2d(input, radius=2, angle=45)  # 进行半圆卷积
print(output.shape)  # 输出为torch.Size([3, 5, 5])

