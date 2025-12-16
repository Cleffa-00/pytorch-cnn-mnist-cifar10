"""
🔮 Quick Inference Demo
Load trained models and make predictions on sample images from the test set.
No training required - just run: python predict.py
"""

import torch
import torchvision
import torchvision.transforms as transforms
from models.cnn import CIFAR10Net, MNISTNet

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'Plane ✈️', 'Car 🚗', 'Bird 🐦', 'Cat 🐱', 'Deer 🦌',
    'Dog 🐕', 'Frog 🐸', 'Horse 🐴', 'Ship 🚢', 'Truck 🚚'
]

def demo_cifar10():
    """Demo inference on CIFAR-10 test images."""
    print("\n" + "="*50)
    print("🖼️  CIFAR-10 Inference Demo")
    print("="*50)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIFAR10Net().to(device)
    
    try:
        model.load_state_dict(torch.load('checkpoints/cifar10_best.pth', 
                                         map_location=device, weights_only=True))
    except FileNotFoundError:
        print("❌ Model not found. Run 'python train_cifar.py' first.")
        return
    
    model.eval()
    print(f"✅ Model loaded on {device}")
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                            download=True, transform=transform)
    
    # Predict on 10 random samples
    print("\n📊 Predictions on random test images:\n")
    indices = torch.randperm(len(testset))[:10]
    
    correct = 0
    for idx in indices:
        image, label = testset[idx]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = output.max(1)
        
        pred_class = CIFAR10_CLASSES[predicted.item()]
        true_class = CIFAR10_CLASSES[label]
        match = "✓" if predicted.item() == label else "✗"
        correct += (predicted.item() == label)
        
        print(f"  {match} Predicted: {pred_class:12} | Actual: {true_class}")
    
    print(f"\n  Accuracy on sample: {correct}/10 ({correct*10}%)")


def demo_mnist():
    """Demo inference on MNIST test images."""
    print("\n" + "="*50)
    print("🔢  MNIST Inference Demo")
    print("="*50)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    
    try:
        model.load_state_dict(torch.load('checkpoints/mnist_best.pth', 
                                         map_location=device, weights_only=True))
    except FileNotFoundError:
        print("❌ Model not found. Run 'python train_mnist.py' first.")
        return
    
    model.eval()
    print(f"✅ Model loaded on {device}")
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, 
                                          download=True, transform=transform)
    
    # Predict on 10 random samples
    print("\n📊 Predictions on random test images:\n")
    indices = torch.randperm(len(testset))[:10]
    
    correct = 0
    for idx in indices:
        image, label = testset[idx]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = output.max(1)
        
        match = "✓" if predicted.item() == label else "✗"
        correct += (predicted.item() == label)
        
        print(f"  {match} Predicted: {predicted.item()} | Actual: {label}")
    
    print(f"\n  Accuracy on sample: {correct}/10 ({correct*10}%)")


if __name__ == '__main__':
    print("\n🚀 PyTorch CNN Inference Demo")
    print("   No training required - using pre-trained weights\n")
    
    demo_cifar10()
    demo_mnist()
    
    print("\n" + "="*50)
    print("✨ Demo complete!")
    print("="*50 + "\n")
