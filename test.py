import torch
from main import Model
import torchvision
import torchvision.transforms as transforms


# 封装一组转换函数对象作为转换器
transform = transforms.Compose(  # Compose是transforms的组合类
    [transforms.ToTensor(),  # ToTensor()类把PIL Image格式的图片和Numpy数组转换成张量
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 用均值和标准差归一化张量图像
)

# 声明批量大小，一批4张图片
batch_size = 4

# 实例化测试集
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        transform=transform, download=True)

# 实例化测试集加载器
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=12)

# CIFAR10数据集所有类别名称
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# print("'''***********- VISUALIZE -*************'''")
import matplotlib.pyplot as plt
import numpy as np


def img_show(img):
    img = img / 2 + 0.5  # 反正则化
    npimg = img.numpy()  # 转换成Numpy数组
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()






if __name__ == '__main__':
    # 随机选取测试集图片
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # 打印输出测试集图片
    img_show(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


    PATH = './output/cifar_model.pth'  # 指定模型参数保存路径
    # 加载模型参数
    model = Model()
    # # 实例化模型对象
    # model = Model().to('cuda')

    model.load_state_dict(torch.load(PATH))
    correct = 0  # 预测正确的数量
    total = 0  # 测试集的总数

    # 由于我们不是训练，我们不需要计算输出的梯度
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # images = images.to('cuda')
            # labels = labels.to('cuda')

            # 模型输出预测结果
            outputs = model(images)

            # 选择置信度最高的类别作为预测类别
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 打印准确率
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total}%')
