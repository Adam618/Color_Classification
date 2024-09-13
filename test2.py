import argparse
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from models import mobilenetv3_small,ShuffleNetV2  # 假设你用的是MobileNetV3
import os
import torch.backends.cudnn as cudnn

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test Model")
parser.add_argument('--input', type=str, default='./data/image_test/', help='Path to the index file')
parser.add_argument('--model_path', type=str, default='./checkpoint/ckpt.pth', help='Path to the model checkpoint')
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

def get_image_filenames(folder_path):
    """递归地从指定文件夹及其子文件夹中读取所有图片文件的名称"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_filenames = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                image_filenames.append(filename)
    return image_filenames


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
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    result = {}
    
    with torch.no_grad():
        for img_name in test_data:
            # 加载图片
            img = Image.open(os.path.join(args.input, img_name)).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            
            # 记录推理开始时间
            start_time = time.time()

            # 推理
            outputs = model(img)
            _, predicted = outputs.max(1)
            
            result[img_name] = predicted.item()

            # 记录推理结束时间
            inference_times.append(time.time() - start_time)

    # 计算平均推理时间
    avg_inference_time = sum(inference_times) / len(inference_times)
    # avg_inference_time = 0
    
    with open('test_reslut.txt', 'w') as file:
        for img_name, pred in result.items():
            # 将键值对格式化成用空格分隔的形式
            formatted_line = f"{img_name}\t{pred}\n"
            # 写入文件
            file.write(formatted_line)

    return avg_inference_time

def main():
    # 设备选择
    # device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device = args.device if torch.cuda.is_available() else 'cpu'

    # 加载模型
    # model = mobilenetv3_small().to(device)
    model = ShuffleNetV2(0.5,num_classes=9).to(device)
    # model = ShuffleNetV2(0.5)
    if device == 'cuda':
        # print(123)
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # 加载模型权重
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['net'])

    # 计算模型大小
    model_size = get_model_size(model)

    # 读取测试数据
    # test_data = load_test_data(args.input)
    test_data = get_image_filenames(args.input)

    # 测试模型并计算准确率和平均推理时间
    avg_inference_time = test_model(model, test_data, device)

    # 输出结果
    print(f"model_size: {model_size:.3f}M, ave_infer_time: {avg_inference_time:.3f}s")

if __name__ == "__main__":
    main()
