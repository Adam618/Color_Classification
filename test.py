import argparse
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from models import mobilenetv3_small  # 假设你用的是MobileNetV3
import os

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test Model")
parser.add_argument('--input', type=str, required=True, help='Path to the index file')
parser.add_argument('--model_path', type=str, default='ckpt.pth', help='Path to the model checkpoint')
parser.add_argument('--device', type=str, default='cuda', help='Device to run inference (cpu/cuda)')
args = parser.parse_args()

# 读取索引文件
def load_test_data(index_file):
    data = []
    with open(index_file, 'r') as f:
        for line in f:
            img_path, label = line.strip().split('\t')
            data.append((img_path, int(label)))
    return data

# 计算模型大小
def get_model_size(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    size_in_mb = param_size / (1024 * 1024)  # 转换为 MB
    return round(size_in_mb, 3)

# 测试函数
def test_model(model, test_data, device):
    model.eval()
    correct = 0
    total = 0
    inference_times = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    with torch.no_grad():
        for img_path, label in test_data:
            # 加载图片
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            label = torch.tensor([label]).to(device)
            
            # 记录推理开始时间
            start_time = time.time()

            # 推理
            outputs = model(img)
            _, predicted = outputs.max(1)

            # 记录推理结束时间
            inference_times.append(time.time() - start_time)

            # 计算准确率
            total += label.size(0)
            correct += (predicted == label).sum().item()

    # 计算总准确率
    accuracy = correct / total
    # 计算平均推理时间
    avg_inference_time = sum(inference_times) / len(inference_times)

    return accuracy, avg_inference_time

def main():
    # 设备选择
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = mobilenetv3_small().to(device)

    # 加载模型权重
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['net'])

    # 计算模型大小
    model_size = get_model_size(model)

    # 读取测试数据
    test_data = load_test_data(args.input)

    # 测试模型并计算准确率和平均推理时间
    accuracy, avg_inference_time = test_model(model, test_data, device)

    # 输出结果
    print(f"Acc: {accuracy:.3f}, model_size: {model_size:.3f}M, ave_infer_time: {avg_inference_time:.3f}s")

if __name__ == "__main__":
    main()
