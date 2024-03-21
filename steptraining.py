import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset,random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os

# Define the transformation for training for each variation
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
# dataset_original = ImageFolder(root=os.path.join(root_path, "original-images"), transform=transform_color_256)
# dataset_bw = ImageFolder(root=os.path.join(root_path, "64-bw"), transform=transform_bw)
# dataset_gray = ImageFolder(root=os.path.join(root_path, "64-gray"), transform=transform_gray)
# Load the dataset (assuming you have a dataset directory named 'waldo_dataset')
train_size_color = int(0.8 * len(dataset_color))
val_size_color = len(dataset_color) - train_size_color
train_set_color, val_set_color = random_split(dataset_color, [train_size_color, val_size_color])

train_size_128 = int(0.8 * len(dataset_color_128))
val_size_128 = len(dataset_color_128) - train_size_128
train_set_128, val_set_128 = random_split(dataset_color_128, [train_size_128, val_size_128])

train_size_256 = int(0.8 * len(dataset_color_256))
val_size_256 = len(dataset_color_256) - train_size_256
train_set_256, val_set_256 = random_split(dataset_color_256, [train_size_256, val_size_256])

# Define data loaders
train_loader_color = DataLoader(train_set_color, batch_size=32, shuffle=True)
val_loader_color = DataLoader(val_set_color, batch_size=32, shuffle=False)

train_loader_128 = DataLoader(train_set_128, batch_size=32, shuffle=True)
val_loader_128 = DataLoader(val_set_128, batch_size=32, shuffle=False)

train_loader_256 = DataLoader(train_set_256, batch_size=32, shuffle=True)
val_loader_256 = DataLoader(val_set_256, batch_size=32, shuffle=False)

# Check if GPU is available and move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load pre-trained ResNet-50 model
pretrained_model = models.resnet50(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in pretrained_model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer to have 2 output classes
num_ftrs = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: Waldo or not Waldo

# Move the model to GPU if available
pretrained_model = pretrained_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

# Define the datasets and loaders for each resolution
datasets = [
    (train_set_color, train_loader_color, val_loader_color),
    (train_set_128, train_loader_128, val_loader_128),
    (train_set_256, train_loader_256, val_loader_256)
]

# Train the model sequentially on each dataset
for i, (train_set, train_loader, val_loader) in enumerate(datasets):
    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = pretrained_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            print(f"Epoch {epoch+1} for dataset {i+1}, Loss: {running_loss / len(train_loader)}")

    print(f"Training finished for dataset {i+1}.")

    # Validation
    pretrained_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = pretrained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy for dataset {i+1}: {100 * correct / total}%")

    # Save the trained model
    
    torch.save(pretrained_model.state_dict(), f'waldo_classifier_dataset_{i+1}.pth')
    print(f"Model for dataset {i+1} saved.")