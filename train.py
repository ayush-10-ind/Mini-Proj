from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Image transformations

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Load datasets

train_dataset = datasets.ImageFolder("data/train", transform=transform)
val_dataset = datasets.ImageFolder("data/val", transform=transform)
test_dataset = datasets.ImageFolder("data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Total training images:", len(train_dataset))
print("Classes:", train_dataset.classes)

# ResNet Model

model = models.resnet18(pretrained=True)

# Modify final layer
model.fc = nn.Linear(model.fc.in_features, 2)

# Freeze base layers (optional but recommended)
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Accuracy Function

def calculate_accuracy(model, loader):
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# Training Loop
for epoch in range(5):

    model.train()

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1} finished")


# Evaluate Model

train_acc = calculate_accuracy(model, train_loader)
val_acc = calculate_accuracy(model, val_loader)
test_acc = calculate_accuracy(model, test_loader)

print("\nModel Performance")
print("Training Accuracy  :", train_acc, "%")
print("Validation Accuracy:", val_acc, "%")
print("Test Accuracy      :", test_acc, "%")

# Save Model
torch.save(model.state_dict(), "model.pth")
print("ResNet model saved!")