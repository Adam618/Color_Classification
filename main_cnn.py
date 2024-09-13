'''Train CIFAR10 with PyTorch.'''
from datetime import datetime

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from d2l import torch as d2l
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter



import os
import argparse

from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label])
          for img, label in trainset
          if label in [0, 2]]
cifar2_val = [(img, label_map[label])
              for img, label in testset
              if label in [0, 2]]




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
    return train_loss


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    writer.add_scalar('test_acc', 100 * correct / total, epoch)
    print('epoch %d accuracy on test set: %d %% ' % (epoch, 100 * correct / total))

    # Save checkpoint.
    test_acc = 100. * correct / total
    # if test_acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'test_acc': test_acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = test_acc
    return test_acc

batch_sizes = [32, 64, 128]
learning_rates = [0.1, 0.01, 0.001]

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        writer = SummaryWriter(f"runs/parameter/MiniBatch {batch_size} LR {learning_rate}" )
        trainloader = torch.utils.data.DataLoader(
            cifar2, batch_size=batch_size, shuffle=True, num_workers=0)

        testloader = torch.utils.data.DataLoader(
            cifar2_val, batch_size=batch_size, shuffle=False, num_workers=0)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

        # Model
        print('==> Building model..')

        net = CNN()  # cifar2
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)

        for epoch in range(start_epoch, start_epoch + 150):

            train_loss = train(epoch)
            test_acc = test(epoch)
            if(epoch == 149):
               writer.add_hparams({'lr': learning_rate, 'bsize': batch_size}, {'accuracy': test_acc})
            print(epoch, train_loss, test_acc)

