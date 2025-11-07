import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据预处理：数据增强 + 标准化
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转
    transforms.ToTensor(),              # 转换为 Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 使用 MNIST 的标准化
])

# 加载数据集
train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 定义更复杂的模型结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

# 初始化模型
model = CNN()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器

# 训练模型
def train(model, train_loader, criterion, optimizer, epochs=5):  # 训练 5 轮
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)
            loss.backward()  # 反向传播

            # 更新参数
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # 每100个批次打印一次
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

    print("Finished Training")

# 在测试集上评估模型
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

# 训练模型
train(model, train_loader, criterion, optimizer, epochs=20)  # 训练 20 轮

# 评估模型
evaluate(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model saved!")



