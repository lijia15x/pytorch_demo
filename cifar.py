import torchvision
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
 

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),  # 展平
            nn.Linear(1024, 64),  # 1024=64*4*4
            nn.Linear(64, 10)
        )
 
    def forward(self, x):
        x = self.model(x)
        return x
 
# 准备数据集
# 训练数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

print(train_data)
# 测试数据集
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
# 观察训练数据集、测试数据集中的图像有多少张
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size)) # 训练数据集的长度为：50000
print("测试数据集的长度为：{}".format(test_data_size)) # 测试数据集的长度为：10000
 
# 使用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64) # 训练数据集加载
test_dataloader = DataLoader(test_data, batch_size=64)  # 测试数据集加载
 
# 创建网络模型
M = Module()
 
# 创建损失函数
# 分类问题采用交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
 
# 设定优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(params=M.parameters(), lr=learning_rate)
 
# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数
epoch = 30
 
# 添加tensorboard
 
# writer = SummaryWriter("./CIFAR10_train")
for i in range(epoch):
    print("———————第{}轮训练开始——————".format(i+1))
 
    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = M(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad() # 梯度清零
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            # writer.add_scalar("train_loss", loss.item(), total_train_step)
 
    # 测试步骤
    total_test_loss = 0
    total_accuracy_num = 0
    with torch.no_grad():
       for data in test_dataloader:
            imgs, targets = data
            outputs = M(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy_num = total_accuracy_num + accuracy
 
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy_num/test_data_size))
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar(("test_accuracy", (total_accuracy_num/test_data_size), total_test_step))
    total_test_step = total_test_step + 1
 
    # 保存每轮运行的模型
    torch.save(M, "./pth/M_{}.pth".format(i))
    print("模型已保存！")
    
print(train_data.classes)
# writer.close()