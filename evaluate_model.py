import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN  # 引用修改后的 CNN 类

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 使用 MNIST 的标准化
])

# 加载测试集
test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 加载模型
model = CNN()  # 初始化模型结构
model.load_state_dict(torch.load('mnist_cnn.pth'))  # 加载训练好的权重
model.eval()  # 切换到评估模式

# 评估模型
def evaluate(model, test_loader):
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

# 运行评估
evaluate(model, test_loader)


