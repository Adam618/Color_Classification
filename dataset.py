import random
from collections import defaultdict

def read_labels(filename):
    """读取标签文件，返回一个字典，键为文件路径，值为标签"""
    labels = defaultdict(list)
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue  # 跳过格式不正确的行
            filepath, label = parts
            labels[int(label)].append((filepath, label))
    return labels

def split_data(labels, train_ratio=0.9):
    """将数据按比例随机划分为训练集和测试集"""
    train_data = []
    test_data = []
    for label, items in labels.items():
        random.shuffle(items)  # 打乱同一标签下的数据
        split_index = int(len(items) * train_ratio)
        train_data.extend(items[:split_index])
        test_data.extend(items[split_index:])
    return train_data, test_data

def write_labels(data, filename):
    """将数据写入文件"""
    with open(filename, 'w') as file:
        for filepath, label in data:
            file.write(f"{filepath}\t{label}\n")

# 读取原始标签文件
labels = read_labels('./data/label/label.txt')


# 按照 9:1 比例分割数据
train_data, test_data = split_data(labels, train_ratio=0.9)

# 写入训练集文件
write_labels(train_data, './data/label/label_train.txt')

# 写入测试集文件
write_labels(test_data, './data/label/label_test.txt')