"""
CIFAR-10 Image Classification Training Script

Train a CNN model on CIFAR-10 dataset with data augmentation
and cosine annealing learning rate schedule.

Usage:
    python train_cifar.py --epochs 50
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models.cnn import CIFAR10Net


# CIFAR-10 class names
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


def get_data_loaders(batch_size=128, num_workers=2, data_dir='./data'):
    """Create train and test data loaders with augmentation."""
    
    transform_train = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return trainloader, testloader


def train_epoch(model, trainloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(trainloader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    return running_loss / len(trainloader), 100. * correct / total


def evaluate(model, testloader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(testloader), 100. * correct / total


def evaluate_per_class(model, testloader, device):
    """Evaluate per-class accuracy."""
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    print("\nPer-class Accuracy:")
    print("-" * 30)
    for i in range(10):
        acc = 100. * class_correct[i] / class_total[i]
        print(f"  {CLASSES[i]:10s}: {acc:5.1f}%")
    print("-" * 30)


def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 CNN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"CIFAR-10 CNN Training")
    print(f"{'='*50}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*50}\n")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = get_data_loaders(
        batch_size=args.batch_size, data_dir=args.data_dir
    )
    print(f"  Train samples: {len(trainloader.dataset)}")
    print(f"  Test samples: {len(testloader.dataset)}\n")
    
    # Create model
    model = CIFAR10Net().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}  Test Acc:  {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir, 'cifar10_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  [*] Saved best model ({test_acc:.2f}%)")
        
        print()
    
    # Training summary
    elapsed = time.time() - start_time
    print(f"{'='*50}")
    print(f"Training Complete!")
    print(f"{'='*50}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Best accuracy: {best_acc:.2f}%")
    print(f"  Model saved to: {args.save_dir}/cifar10_best.pth")
    
    # Per-class accuracy
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'cifar10_best.pth')))
    evaluate_per_class(model, testloader, device)


if __name__ == '__main__':
    main()
