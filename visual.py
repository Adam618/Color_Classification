import pandas as pd
import matplotlib.pyplot as plt

# 读取 label.txt 文件
# label_file = './data/label/label.txt'  # 确保路径正确
# data = []

# # 按行读取文件
# with open(label_file, 'r') as file:
#     for line in file:
#         img_name, label = line.strip().split('\t')
#         data.append((img_name, int(label)))

# # 创建 DataFrame
# df = pd.DataFrame(data, columns=["Image", "Label"])

# # 计算标签分布
# label_distribution = df["Label"].value_counts()

# # 打印标签分布
# print("标签分布：")
# print(label_distribution)

# # 绘制标签分布的柱状图
# plt.figure(figsize=(8, 6))
# label_distribution.plot(kind='bar')
# plt.title('Label Distribution in label.txt')
# plt.xlabel('Labels')
# plt.ylabel('Frequency')
# plt.xticks(rotation=0)
# plt.show()


# import torch
# from models import *

# def getModelSize(model):
#     param_size = 0
#     for param in model.parameters():
#         param_size += param.nelement() * param.element_size()
#     size = param_size / 1024 / 1024  # 将字节转换为MB
#     print('模型大小：{:.3f}MB'.format(size))
#     return size

# # 假设你已经定义了模型架构，或者从命令行中动态选择了模型
# # 示例，假设使用mobilenetv3_small模型
# model = mobilenetv3_small()  # 这是你的模型架构
# model = torch.nn.DataParallel(model)

# # 加载模型参数
# checkpoint = torch.load('./checkpoint/mobienetv3_97.21.pth')  # 指定你的模型文件路径
# model.load_state_dict(checkpoint['net'])  # 假设保存时键为'net'

# # 计算并输出模型大小
# model_size = getModelSize(model)
# print(f"Loaded model size: {model_size:.3f} MB")

import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # 引入 tqdm

def calculate_hsv_mean_std(image_dir):
    h_vals, s_vals, v_vals = [], [], []

    # 获取图像文件列表
    img_files = os.listdir(image_dir)
    
    # 遍历文件夹中的所有图像文件，使用 tqdm 显示进度条
    for img_name in tqdm(img_files, desc="Processing Images"):
        img_path = os.path.join(image_dir, img_name)
        try:
            # 打开图像并转换为HSV格式
            img = Image.open(img_path).convert('HSV')
            img_np = np.array(img)
            
            # 分离 H, S, V 通道
            h_vals.append(img_np[:, :, 0].flatten())
            s_vals.append(img_np[:, :, 1].flatten())
            v_vals.append(img_np[:, :, 2].flatten())
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    
    # 将每个通道的所有图像像素值连接在一起
    h_vals = np.concatenate(h_vals)
    s_vals = np.concatenate(s_vals)
    v_vals = np.concatenate(v_vals)
    
    # 计算每个通道的均值和标准差
    h_mean, h_std = np.mean(h_vals), np.std(h_vals)
    s_mean, s_std = np.mean(s_vals), np.std(s_vals)
    v_mean, v_std = np.mean(v_vals), np.std(v_vals)
    
    return (h_mean, h_std), (s_mean, s_std), (v_mean, v_std)

# 指定图像文件夹路径
image_dir = './data/image_train'

# 计算并输出HSV通道的均值和标准差
(h_mean, h_std), (s_mean, s_std), (v_mean, v_std) = calculate_hsv_mean_std(image_dir)

print(f"H channel - Mean: {h_mean}, Std: {h_std}")
print(f"S channel - Mean: {s_mean}, Std: {s_std}")
print(f"V channel - Mean: {v_mean}, Std: {v_std}")






