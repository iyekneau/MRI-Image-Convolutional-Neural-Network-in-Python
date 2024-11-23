import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation and transformations
train_transforms = transforms.Compose([     
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([     
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Data loading
train_data = ImageFolder(r"your location for training set", transform=train_transforms)
test_data = ImageFolder(r"your location for test set", transform=test_transforms)

batch_size = 2

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# CNN Model definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(3, 32, 5)
        self.cnn2 = nn.Conv2d(32, 64, 4)
        self.cnn3 = nn.Conv2d(64, 64, 2)
        self.fc1 = nn.Linear(64 * 23 * 23, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.cnn1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.cnn2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.cnn3(x), 2, 2))
        x = x.view(-1, 64 * 23 * 23)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model initialization
model = CNN()
model.eval()
model = model.to(device)

# Optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Hook function to capture feature maps
def hook_fn(module, input, output):
    global feature_maps
    feature_maps = output.detach()

# Register hook on the convolutional layers
def register_hooks(model):
    # Hooking into all convolutional layers
    hooks = []
    hooks.append(model.cnn1.register_forward_hook(hook_fn))  # First convolution layer
    hooks.append(model.cnn2.register_forward_hook(hook_fn))  # Second convolution layer
    hooks.append(model.cnn3.register_forward_hook(hook_fn))  # Third convolution layer
    return hooks

# Visualize feature maps
def visualize_feature_maps(feature_maps, num_cols=8):
    num_feature_maps = feature_maps.size(1)
    num_rows = num_feature_maps // num_cols + (num_feature_maps % num_cols != 0)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(num_feature_maps):
        ax = axes[i]
        ax.imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Training loop with feature map visualization
loss_array = []
n_epochs = 80  # Number of epochs

# Register hooks to capture feature maps
hooks = register_hooks(model)

for epoch in range(n_epochs):
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_array.append(loss.item())

        # Visualize feature maps after the first convolution layer for a batch
        if i == 0:  # Only visualize for the first batch of each epoch
            visualize_feature_maps(feature_maps)

print("Done Training")

# Unregister hooks after training to free memory
for hook in hooks:
    hook.remove()
