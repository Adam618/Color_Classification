import pandas as pd
import matplotlib.pyplot as plt

# 读取 label.txt 文件
label_file = './data/label/label.txt'  # 确保路径正确
data = []

# 按行读取文件
with open(label_file, 'r') as file:
    for line in file:
        img_name, label = line.strip().split('\t')
        data.append((img_name, int(label)))

# 创建 DataFrame
df = pd.DataFrame(data, columns=["Image", "Label"])

# 计算标签分布
label_distribution = df["Label"].value_counts()

# 打印标签分布
print("标签分布：")
print(label_distribution)

# 绘制标签分布的柱状图
plt.figure(figsize=(8, 6))
label_distribution.plot(kind='bar')
plt.title('Label Distribution in label.txt')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


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



