import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 使用 Sequential 定义模型结构，确保与保存模型时的结构一致（super继承父类nn.Module）
        self.net = nn.Sequential(
            # 第一层：卷积->激活->池化
            
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # net.0
            # 参数含义：卷积核大小（3*3常用）步长 填充
            nn.ReLU(), 
            # 激活函数relu=max(0,x)
            nn.MaxPool2d(2),  # net.1
            # 取区域内的最大值，适合特征提取任务

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # net.2
            nn.ReLU(),
            nn.MaxPool2d(2),  # net.3

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # net.4
            nn.ReLU(),
            nn.MaxPool2d(2),  # net.5
        )

        # 全连接层，将提取好的特征映射到具体类别
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # net.6 将三维图映射到512个神经元的隐藏层（infeatuers,outfeatures）
        self.fc2 = nn.Linear(512, 10)  # net.7  输出 最后分类数量为10

    def forward(self, x):
        x = self.net(x)  # 使用 Sequential 包装的卷积层
        x = x.view(-1, 128 * 3 * 3)  # 展平操作
        x = torch.relu(self.fc1(x))  # net.6 通过第一层并加激活函数（非线性一下）
        x = self.fc2(x)  # net.7 通过第二层，得到最终 10 个分类的得分
        return x
