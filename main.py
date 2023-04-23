import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import os

"""
加载并正则化CIFAR10数据集
"""
# print("***********- ***********- READ DATA and processing-*************")
# 封装一组转换函数对象作为转换器
transform = transforms.Compose(  # Compose是transforms的组合类
    [transforms.ToTensor(),  # ToTensor()类把PIL Image格式的图片和Numpy数组转换成张量
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 用均值和标准差归一化张量图像
)

# 声明批量大小，一批4张图片
batch_size = 4

# 实例化训练集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         transform=transform, download=True)

# 实例化训练集加载器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=12)



# CIFAR10数据集所有类别名称
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


"""
训练可视化
"""
# print("'''***********- VISUALIZE -*************'''")
import matplotlib.pyplot as plt
import numpy as np


def img_show(img):
    img = img / 2 + 0.5  # 反正则化
    npimg = img.numpy()  # 转换成Numpy数组
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



"""
定义神经网络
"""
import torch.nn as nn
import torch.nn.functional as F

# print("'''***********- Model -*************'''")
# 定义网络模型，继承自torch.nn.Module类
class Model(nn.Module):
    # 构造器
    def __init__(self):
        super().__init__()  # 初始化父类的属性
        # Model类的属性
        self.conv1 = nn.Conv2d(3, 6, 5)  # 卷积层1
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积层2
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层1
        self.fc2 = nn.Linear(120, 84)  # 全连接层2
        self.fc3 = nn.Linear(84, 10)  # 全连接层3

    # 前向传播方法
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # 把x沿着水平方向展开
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
训练模型
"""
# 开始训练

if __name__ == '__main__':
    dataiter = iter(train_loader)  # 训练集图片的迭代器
    images, labels = dataiter.next()  # 获取每个图片和标签
    img_show(torchvision.utils.make_grid(images))  # 显示图片
    print(" ".join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))  # 打印标签
    # 实例化模型对象
    model = Model()
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("'''***********- train -*************'''")
    for epoch in range(4):  # 训练4个epoch
        running_loss = 0.0  # 损失函数记录
        for i, data in enumerate(train_loader, 0):
            # 获取模型输入；data是由[inputs, labels]组成的列表
            inputs, labels = data
            # 把参数的梯度清零
            optimizer.zero_grad()
            # 前向传播+反向传播+更新权重
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 打印统计数据
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个mini-batch打印一次统计信息
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0  # 把损失函数记录清零
    print("训练结束")
    """
    保存模型
    """
    # 保存模型参数
    PATH = './output/cifar_model.pth'  # 指定模型参数保存路径
    torch.save(model.state_dict(), PATH)  # 保存模型参数


