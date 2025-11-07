import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 使用 Sequential 定义模型结构，确保与保存模型时的结构一致
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # net.0
            nn.ReLU(),
            nn.MaxPool2d(2),  # net.1

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # net.2
            nn.ReLU(),
            nn.MaxPool2d(2),  # net.3

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # net.4
            nn.ReLU(),
            nn.MaxPool2d(2),  # net.5
        )
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # net.6
        self.fc2 = nn.Linear(512, 10)  # net.7

    def forward(self, x):
        x = self.net(x)  # 使用 Sequential 包装的卷积层
        x = x.view(-1, 128 * 3 * 3)  # 展平操作
        x = torch.relu(self.fc1(x))  # net.6
        x = self.fc2(x)  # net.7
        return x
