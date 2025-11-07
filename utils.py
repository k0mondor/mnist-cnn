import torch

def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean().item()
