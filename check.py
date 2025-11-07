import os

if os.path.exists('mnist_cnn.pth'):
    print("Model weights found!")
else:
    print("Model weights not found!")
