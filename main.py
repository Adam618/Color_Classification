'''Train CIFAR10 with PyTorch.'''
from datetime import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, random_split
from d2l import torch as d2l
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from ImageDataset import ImageDataset 
import os
import argparse
from models import *
from utils import progress_bar
from utils import Animator
writer = SummaryWriter("runs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# 添加学习率参数
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

# # 是否从checkpoint恢复
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# 添加batch size参数
parser.add_argument('--batch_size', default=128, type=int, help='batch size for training and testing')

# 添加epoch参数
parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train')

# # 添加模型选择参数
parser.add_argument('--model', default='shufflenetv2', type=str, help='choose model architecture')

# Early stopping参数 没启用
parser.add_argument('--early_stopping', action='store_false', help='enable early stopping')

# patience参数，用于early stopping
parser.add_argument('--patience', default=50, type=int, help='patience for early stopping')

# # log dir for TensorBoard
# parser.add_argument('--log_dir', default='runs/', type=str, help='directory to save TensorBoard logs')

parser.add_argument('--mode', default='train', type=str, help='set mode')

parser.add_argument('--load_model', default='ckpt.pth', type=str, help='load model name')
parser.add_argument('--save_model', default='ckpt.pth', type=str, help='save model name')

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),  # 颜色抖动
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 标准化
])


transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 创建训练集和验证集数据集实例
trainset = ImageDataset(img_dir='./data/image_train/', label_file='./data/label/label_train.txt',
                             transform=transform_train)

testset = ImageDataset(img_dir='./data/image_train/', label_file='./data/label/label_test.txt',
                            transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

examples = iter(testloader)
examples_data, examples_targets = examples.next()
img_grid = torchvision.utils.make_grid(examples_data)
writer.add_image("images", img_grid)

# Model
print('==> Building model..')

# 动态选择模型
if args.model == 'vgg19':
    net = VGG('VGG19')
elif args.model == 'resnet32':
    net = resnet32(num_classes=9)
elif args.model == 'resnet110':
    net = resnet110(num_classes=9)
elif args.model == 'mobilenetv3_small':
    net = mobilenetv3_small(num_classes=9)
elif args.model == 'preactresnet18':
    net = PreActResNet18(num_classes=9)
elif args.model == 'googlenet':
    net = GoogLeNet(num_classes=9)
elif args.model == 'densenet121':
    net = DenseNet121(num_classes=9)
elif args.model == 'resnext29_2x64d':
    net = ResNeXt29_2x64d(num_classes=9)
elif args.model == 'mobilenet':
    net = MobileNet(num_classes=9)
elif args.model == 'mobilenetv2':
    net = MobileNetV2(num_classes=9)
elif args.model == 'mobilenetv3':
    net = MobileNetV3(num_classes=9)
elif args.model == 'dpn92':
    net = DPN92(num_classes=9)
elif args.model == 'shufflenetg2':
    net = ShuffleNetG2(num_classes=9)
elif args.model == 'senet18':
    net = SENet18(num_classes=9)
elif args.model == 'shufflenetv2':
    net = ShuffleNetV2(0.5,num_classes=9)
elif args.model == 'efficientnetb0':
    net = EfficientNetB0(num_classes=9)
elif args.model == 'regnetx_200mf':
    net = RegNetX_200MF(num_classes=9)
elif args.model == 'simpledla':
    net = SimpleDLA(num_classes=9)
else:
    raise ValueError(f"Model {args.model} not recognized.")


net = net.to(device)

def getModelSize(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    size = param_size / 1024 / 1024
    print('模型大小：{:.3f}MB'.format(size))

getModelSize(net)
    
    

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


params = {
"Learning Rate": args.lr,
"Batch Size": args.batch_size,
"Epochs": args.epochs,
"Model": args.model,
"Early Stopping": args.early_stopping,
"Patience": args.patience,
"Mode": args.mode,
"Load Model": args.load_model,
"Save Model": args.save_model
}

# 打印表格
print("Training Configuration:")
print(tabulate(params.items(), headers=["Parameter", "Value"]))

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load(f'./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

## 自定义权重，类别数平方根倒数，缓解长尾问题
def custom_loss():
    # 假设你有9个类别（根据你的数据集调整）
    num_classes = 9
    class_counts = [0] * num_classes

    # 遍历整个训练集来统计每个类别的样本数量
    for _, targets in trainloader:
        for target in targets:
            class_counts[target.item()] += 1

    # 计算每个类别的权重，类别数的平方根倒数
    class_weights = []
    for count in class_counts:
        if count == 0:
            class_weights.append(1.0)  # 默认权重，可以调整
        else:
            # class_weights.append(1.0 / (count ** 0.5))
            class_weights.append(1.0 / (count))
    class_weights = torch.tensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion

writer.add_graph(net, examples_data.to(device))
# criterion = nn.CrossEntropyLoss()
criterion = custom_loss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                     weight_decay=2e-4, momentum=0.9)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    writer.add_scalar('train loss', train_loss, epoch)
    train_loss = 0.0
    # print('Loss: %.3f | Acc: %.3f%% (%d/%d)'%(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    num_classes = 9  # 假设有9个类别
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 记录每个类别的正确预测数和总数
            for i in range(len(targets)):
                label = targets[i].item()
                pred = predicted[i].item()
                if label == pred:
                    correct_per_class[label] += 1
                total_per_class[label] += 1

    # 输出每个类别的准确率
    for i in range(num_classes):
        if total_per_class[i] > 0:
            accuracy = 100 * correct_per_class[i] / total_per_class[i]
            print(f'Class {i} accuracy: {accuracy:.2f}%')
            writer.add_scalar(f'Class_{i}_accuracy', accuracy, epoch)

    writer.add_scalar('test_acc', 100 * correct / total, epoch)
    print(f'Epoch {epoch} accuracy on test set: {100 * correct / total:.2f}%')

    # Early Stopping 检查
    test_acc = 100. * correct / total
    if test_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{args.save_model}')
        best_acc = test_acc
    return test_acc


def evaluate_model(checkpoint_path):
    print('==> Loading model from checkpoint..')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    correct = 0
    total = 0
    correct_per_class = [0] * 10
    total_per_class = [0] * 10

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 记录每个类别的正确预测数和总数
            for i in range(len(targets)):
                label = targets[i].item()
                pred = predicted[i].item()
                if label == pred:
                    correct_per_class[label] += 1
                total_per_class[label] += 1

    # 输出每个类别的准确率
    for i in range(10):
        if total_per_class[i] > 0:
            accuracy = 100 * correct_per_class[i] / total_per_class[i]
            print(f'Class {i} accuracy: {accuracy:.2f}%')

    print(f'Total accuracy on test set: {100 * correct / total:.2f}%')


def train_model():
    global best_acc
    early_stop_counter = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = train(epoch)
        test_acc = test(epoch)
        print(f'Epoch {epoch}, Train Loss: {train_loss}, Test Accuracy: {test_acc:.2f}%')

        # Early Stopping 判断
        if test_acc > best_acc:
            best_acc = test_acc
            early_stop_counter = 0  # 重置计数器
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break


if args.mode == 'train':
    train_model()
else:
    evaluate_model(f'./checkpoint/{args.load_model}')


# for epoch in range(start_epoch, start_epoch + 50):
#     train_loss = train(epoch)
#     test_acc = test(epoch)
#     print(epoch, train_loss, test_acc)
#     for name, param in net.named_parameters():
#         writer.add_histogram(tag=name + '_grad', values=param.grad, global_step=epoch)
#         writer.add_histogram(tag=name + '_data', values=param.data, global_step=epoch)
