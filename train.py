from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------------
# Image transformations
# -------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -------------------------------
# Load datasets
# -------------------------------
train_dataset = datasets.ImageFolder("data/train", transform=transform)
val_dataset = datasets.ImageFolder("data/val", transform=transform)
test_dataset = datasets.ImageFolder("data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Total training images:", len(train_dataset))
print("Classes:", train_dataset.classes)


# -------------------------------
# CNN Model
# -------------------------------
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(16 * 111 * 111, 2)

    def forward(self, x):

        x = torch.relu(self.conv(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


# -------------------------------
# Create model
# -------------------------------
model = SimpleCNN()


# -------------------------------
# Loss and Optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# -------------------------------
# Accuracy Function
# -------------------------------
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


# -------------------------------
# Training Loop
# -------------------------------
model.train()

for epoch in range(3):

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    print("Epoch", epoch + 1, "finished")


# -------------------------------
# Evaluate Model
# -------------------------------
train_acc = calculate_accuracy(model, train_loader)
val_acc = calculate_accuracy(model, val_loader)
test_acc = calculate_accuracy(model, test_loader)

print("\nModel Performance")
print("----------------------")
print("Training Accuracy  :", train_acc, "%")
print("Validation Accuracy:", val_acc, "%")
print("Test Accuracy      :", test_acc, "%")

torch.save(model.state_dict(), "model.pth")
print("Model saved successfully!")