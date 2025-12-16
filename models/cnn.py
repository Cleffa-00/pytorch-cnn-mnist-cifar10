"""
CNN Models for Image Classification

This module contains CNN architectures for CIFAR-10 and MNIST datasets.
"""

import torch
import torch.nn as nn


class CIFAR10Net(nn.Module):
    """
    CNN Architecture for CIFAR-10 Classification
    
    Architecture:
        - 6 Convolutional layers with BatchNorm
        - 3 MaxPooling layers
        - Global Average Pooling
        - Fully connected classifier
    
    Input: (B, 3, 32, 32)
    Output: (B, 10)
    """
    
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MNISTNet(nn.Module):
    """
    CNN Architecture for MNIST Classification
    
    Architecture:
        - 4 Convolutional layers with BatchNorm
        - 2 MaxPooling layers
        - Global Average Pooling + Dropout
        - Fully connected classifier
    
    Input: (B, 1, 28, 28)
    Output: (B, 10)
    """
    
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Test models
    print("Testing CIFAR10Net...")
    model = CIFAR10Net()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"  Input: {x.shape} -> Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting MNISTNet...")
    model = MNISTNet()
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    print(f"  Input: {x.shape} -> Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
