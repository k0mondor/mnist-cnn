import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # 用于可视化训练过程
from model import CNN  # 导入模型定义




# 数据预处理：数据增强 + 标准化
transform = transforms.Compose([
    transforms.RandomRotation(10),      # 随机旋转
    transforms.ToTensor(),              # 转换为 Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 使用 MNIST 的标准化
])




# 加载数据集（训练集和测试集）
train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True) # shuffle=True 打乱数据顺序（训练时常用，测试时不需要）
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)




# 训练模型
def train(model, train_loader, criterion, optimizer, device, writer, epoch): 
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)
            loss.backward()  # 反向传播

            # 更新参数
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # 每 100 个批次记录一次损失（可视化）
                global_step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/train', running_loss / 100, global_step)

                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.4f}")
                running_loss = 0.0





# 评估模型
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度（只前向传播）
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy




# 执行逻辑
if __name__ == "__main__":

    # 1. 动态参数
    parser = argparse.ArgumentParser(description="MNIST 训练任务")
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--save_path', type=str, default='mnist_cnn.pth', help='模型保存路径')
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. TensorBoard 可视化设置
    writer = SummaryWriter('runs/mnist_experiment')

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    

    # early stopping 相关变量
    best_acc = 0.0
    stagnant_epochs = 0
    
    # 3. 训练和评估循环
    for epoch in range(args.epochs):
        train(model, train_loader, criterion, optimizer, device, writer, epoch)

        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}, Test Accuracy: {acc:.2f}%")
        writer.add_scalar('Accuracy/test', acc, epoch)

        # 早停逻辑：如果准确率没有提升，增加 stagnant_epochs 计数器
        if acc > best_acc:
            best_acc = acc
            stagnant_epochs = 0
            torch.save(model.state_dict(), args.save_path)  # 保存最佳模型
        else:
            stagnant_epochs += 1

        # 如果连续 3 个 epoch 没有提升，提前停止训练
        if stagnant_epochs >= 3:
            print("Early stopping triggered.")
            break

    writer.close()
    print(f"Training completed. Best Accuracy: {best_acc:.2f}%")
