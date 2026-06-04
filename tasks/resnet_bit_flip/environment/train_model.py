"""
train_model.py — Train the frozen CIFAR-10 classifier at Docker build time.

DO NOT MODIFY. Run once during Docker build to create /app/model.pth.

Deterministic: fixed seed, fixed number of epochs, shuffle generator seeded,
BatchNorm in train mode. Target test accuracy ~70-80% in a few minutes on CPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MiniResNet

SEED = 42
EPOCHS = 8
BATCH_SIZE = 128
LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_data = datasets.CIFAR10('/data/cifar10', train=True, download=True,
                                  transform=transform)
    test_data = datasets.CIFAR10('/data/cifar10', train=False, download=True,
                                 transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=512, shuffle=False, num_workers=0)

    model = MiniResNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MiniResNet params: {n_params}")

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS}: loss={total_loss/total:.4f} "
              f"train_acc={correct/total:.4f} lr={scheduler.get_last_lr()[0]:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
    test_acc = correct / total
    print(f"Test accuracy: {test_acc:.4f} ({correct}/{total})")

    if test_acc < 0.55:
        raise RuntimeError(f"Training accuracy {test_acc:.4f} below sanity threshold")

    torch.save(model.state_dict(), '/app/model.pth')
    print("Model saved to /app/model.pth")


if __name__ == '__main__':
    main()
