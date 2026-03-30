import sys
print(sys.executable)
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1037,), (0.3081,))
])




# 加载测试集
try:
    test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
except Exception as e:
    print(f"数据加载失败，请检查路径: {e}")
    sys.exit()



# 加载模型
model = CNN().to(device)

try:
    model.load_state_dict(torch.load('mnist_fast.pth', map_location=device))
    model.eval()  # 加载成功后再切换模式
    print("模型权重加载成功！")
except FileNotFoundError:
    print("找不到权重文件，请确认文件名是否为 mnist_fast.pth")
    exit() # 找不到就直接退出，防止后面报错



# 获取一个测试样本
data_iter = iter(test_loader)
images, labels = next(data_iter)



# 进行预测
with torch.no_grad():
    images = images.to(device)
    output = model(images)
    _, predicted = torch.max(output, 1)


print(f"预测结果: {predicted.item()}, 真实标签: {labels[0].item()}")



# 显示图像
plt.imshow(images[0][0].cpu().numpy(), cmap='gray')
plt.title(f"True label: {labels[0].item()}")
plt.show()