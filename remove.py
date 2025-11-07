import os

# 删除已经保存的模型权重文件（如果存在）
if os.path.exists('mnist_cnn.pth'):
    os.remove('mnist_cnn.pth')
    print("Old model weights removed.")
else:
    print("No existing model weights to remove.")
