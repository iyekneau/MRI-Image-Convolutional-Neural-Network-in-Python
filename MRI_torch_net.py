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


# If you'd like to use a GPU and if CUDA is available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Data augmentation
train_transforms = transforms.Compose([     
                                transforms.Resize((200,200)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                                ])

test_transforms = transforms.Compose([     
                                transforms.Resize((200,200)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                                ])

# procuring data from local source, you can use whatever root path you please
train_data = ImageFolder(r"C:\Users\19723\Desktop\main_dir\train",
                           transform=train_transforms)

test_data = ImageFolder(r"C:\Users\19723\Desktop\main_dir\test",
                           transform=test_transforms)

# batch size and loaders. optimal is 2 batches
batch_size = 2

dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

train_loader = iter(dataloader)

test_dataloader= DataLoader(test_data, batch_size = batch_size, shuffle = True)

test_loader = iter(test_dataloader)

# now to build the net itself

class CNN(nn.Module):
        def __init__(self):
                    super(CNN,self).__init__()
                    self.cnn1 = nn.Conv2d(3,32,5)
                    self.cnn2 = nn.Conv2d(32, 64, 4)
                    self.cnn3 = nn.Conv2d(64, 64, 2)
                    self.fc1 = nn.Linear(64*23*23, 512)
                    self.fc2 = nn.Linear(512, 3)
                    
                      
                    
                    
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.cnn1(x),2,2))
            x = F.relu(F.max_pool2d(self.cnn2(x),2,2))
            x = F.relu(F.max_pool2d(self.cnn3(x),2,2))
            #print("Shape before flattening:", x.shape)  # Debugging line

            x = x.view(-1, 64*23*23)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            
            
            return x
        
        
model = CNN()
model.eval()
#making sure it is on CPU
model = model.to(device)


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum= 0.9)
criterion =  nn.CrossEntropyLoss()
n_epochs = 80           #60 epochs seems to maximize accuracy


loss_array = []

for epoch in range(n_epochs):  # loop over the dataset multiple times
    
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()


        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_array.append(loss.item())
        
    #print(loss_array)    # you may uncomment this if you want to see the loss data
    
print("Finished Training")   


# reporting the accuracy

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network: %d %%' % (
    100 * correct / total))





# 11/21/24: accuracy low:60, high =83,  dataset near 400 images