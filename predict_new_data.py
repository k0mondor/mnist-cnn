import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import CNN  # 假设模型定义在 model.py 文件中

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载测试集
test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

# 加载模型
model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()  # 切换到评估模式

# 获取一个测试样本
data_iter = iter(test_loader)
images, labels = next(data_iter)

# 显示图像
plt.imshow(images[0][0], cmap='gray')
plt.title(f"True label: {labels[0].item()}")
plt.show()

# 进行预测
with torch.no_grad():
    output = model(images)
    _, predicted = torch.max(output, 1)

print(f'Predicted label: {predicted.item()}')
