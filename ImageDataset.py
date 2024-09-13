import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms



class ImageDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        """
        Args:
            img_dir (str): 图像文件存放目录
            label_file (str): 标签文件路径
            transform_train (callable, optional): 训练集图像预处理函数
            transform_test (callable, optional): 测试集图像预处理函数
            train (bool): 指定当前是训练集还是验证集
            split_ratio (float): 训练集与验证集划分比例
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        self.images = {}  # 用于存储加载到内存中的图片

        # 读取标签文件
        # with open(label_file, 'r') as f:
        #     for line in f:
        #         img_name, label = line.strip().split('\t')
        #         self.img_labels.append((img_name, int(label)))

        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):  # 使用 enumerate(f, 1) 从 1 开始计数
                line = line.strip()
                try:
                    img_name, label = line.split('\t')
                    self.img_labels.append((img_name, int(label)))
                except ValueError:
                    print(f"Error processing line {line_num}: {line}")  # 输出出错行号和内容
                    raise  # 抛出异常以停止程序并显示错误堆栈

        # 一次性将所有图像加载到内存中
        for img_name, _ in self.img_labels:
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert("RGB")  # 将图像转换为RGB
            self.images[img_name] = image  # 将图像存储到字典中

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        image = self.images[img_name]  # 从内存中获取图像

        image = self.transform(image)

        return image, label


# 示例用法

# 定义训练集和验证集的transform
# transform_train = transforms.Compose([
#     transforms.RandomHorizontalFlip(),  # 数据增强：随机水平翻转
#     transforms.ToTensor(),  # 转换为Tensor
#     transforms.Resize((224, 224)),  # 调整图像大小
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),  # 转换为Tensor
#     transforms.Resize((224, 224)), # 调整图像大小
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# ])


