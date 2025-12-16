"""
Generate sample prediction images for README display.
Saves visualization of model predictions on test samples.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from models.cnn import CIFAR10Net, MNISTNet

# CIFAR-10 class names
CIFAR10_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

def generate_cifar10_demo():
    """Generate CIFAR-10 prediction visualization."""
    print("Generating CIFAR-10 demo image...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIFAR10Net().to(device)
    model.load_state_dict(torch.load('checkpoints/cifar10_best.pth', 
                                     map_location=device, weights_only=True))
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                            download=True, transform=transform)
    
    # Get 8 random samples
    torch.manual_seed(42)  # For reproducibility
    indices = torch.randperm(len(testset))[:8]
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('CIFAR-10 Predictions (90.49% Test Accuracy)', fontsize=14, fontweight='bold')
    
    for i, idx in enumerate(indices):
        image, label = testset[idx]
        
        # Predict
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            _, predicted = output.max(1)
        
        pred_class = CIFAR10_CLASSES[predicted.item()]
        true_class = CIFAR10_CLASSES[label]
        correct = predicted.item() == label
        
        # Denormalize for display
        img = image.numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5  # Denormalize
        
        ax = axes[i // 4, i % 4]
        ax.imshow(img)
        ax.axis('off')
        
        color = 'green' if correct else 'red'
        ax.set_title(f'Pred: {pred_class}\nTrue: {true_class}', 
                     color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('assets/cifar10_demo.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Saved: assets/cifar10_demo.png")


def generate_mnist_demo():
    """Generate MNIST prediction visualization."""
    print("Generating MNIST demo image...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load('checkpoints/mnist_best.pth', 
                                     map_location=device, weights_only=True))
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, 
                                          download=True, transform=transform)
    
    # Get 10 samples (one of each digit)
    torch.manual_seed(123)
    indices = torch.randperm(len(testset))[:10]
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('MNIST Predictions (99.70% Test Accuracy)', fontsize=14, fontweight='bold')
    
    for i, idx in enumerate(indices):
        image, label = testset[idx]
        
        # Predict
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            _, predicted = output.max(1)
        
        correct = predicted.item() == label
        
        # Denormalize for display
        img = image.squeeze().numpy()
        img = img * 0.5 + 0.5  # Denormalize
        
        ax = axes[i // 5, i % 5]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        color = 'green' if correct else 'red'
        ax.set_title(f'Pred: {predicted.item()} | True: {label}', 
                     color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('assets/mnist_demo.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Saved: assets/mnist_demo.png")


if __name__ == '__main__':
    import os
    os.makedirs('assets', exist_ok=True)
    
    generate_cifar10_demo()
    generate_mnist_demo()
    
    print("\n🎉 Demo images generated! Add to README with:")
    print("   ![CIFAR-10 Demo](assets/cifar10_demo.png)")
    print("   ![MNIST Demo](assets/mnist_demo.png)")
