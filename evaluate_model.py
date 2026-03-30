import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN  # 引用修改后的 CNN 类


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 使用 MNIST 的标准化
])



# 加载测试集
test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)



def evaluate_and_report(model_path):
    
    model = CNN().to(device)
    
    # 加载权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载权重文件: {model_path}")
    except FileNotFoundError:
        print(f"错误：找不到权重文件 {model_path}，请先运行 train.py")
        return

    model.eval()  # 切换到评估模式（会关闭 Dropout 和 BatchNorm）

    correct = 0
    total = 0
    
    # 开始评估
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("-" * 30)
    print(f'测试集总样本数: {total}')
    print(f'准确率 (Accuracy): {accuracy:.2f}%')
    print("-" * 30)



if __name__ == "__main__":
    evaluate_and_report('mnist_fast.pth')


