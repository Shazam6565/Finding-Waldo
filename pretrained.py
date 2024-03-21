import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# Define a simple CNN model for classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: Waldo or not Waldo

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),         # Convert images to tensors
])

# Load the dataset (assuming you have a dataset directory named 'waldo_dataset')
transform_color = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

transform_bw = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

transform_gray = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

transform_64 = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),         # Convert images to tensors
])

transform_color_128 = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),           # Convert images to tensors
])

transform_color_256 = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),           # Convert images to tensors
])

# Load the Waldo datasets for each variation
root_path = "/unity/f2/asugandhi/MonsoonForecast/findWaldos/wheres-waldo/Hey-Waldo"
dataset_color = ImageFolder(root=os.path.join(root_path, "64"), transform=transform_color)
dataset_color_128 = ImageFolder(root=os.path.join(root_path, "128"), transform=transform_color_128) 
dataset_color_256 = ImageFolder(root=os.path.join(root_path, "256"), transform=transform_color_256)

# Create a DataLoader
data_loader = DataLoader(dataset_color, batch_size=32, shuffle=True)

train_size = int(0.8 * len(dataset_color))  # 80% training, 20% validation
val_size = len(dataset_color) - train_size
train_set, val_set = torch.utils.data.random_split(dataset_color, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)  # No need to shuffle validation data

# Check if GPU is available and move the model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(data_loader)}")

print("Training finished.")

# Save the trained model
torch.save(model.state_dict(), 'simple_cnn_waldo.pth')
print("Model saved.")

# Validation
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total}%")
