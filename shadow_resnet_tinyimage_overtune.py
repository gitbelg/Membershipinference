import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
#import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34

device = "cpu" # set to gpu if you have one
model = torch.load("resnet_tinyimage/resnet34_shadow_tinyimage_overtrained.pth",map_location=torch.device('cpu'))
model2 = resnet34(pretrained = False,num_classes = 200).to(device)
model2.load_state_dict(model, strict=False)
model  = model2
DATA_PATH = 'resnet_tinyimage/shadow_tinyimage_resnet.p'
# Change the DATA_PATH to your local pickle file path

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(DATA_PATH, "rb") as f:
    dataset = pickle.load(f)


#splitting
#only use train set here
train_data, val_data = train_test_split(dataset, test_size=(1-0.5),shuffle=False)

dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=128 , shuffle=False, num_workers=4)
testloader =  torch.utils.data.DataLoader(val_data, batch_size=1,
                                          shuffle=True, num_workers=2)

for batch_idx, (img, label) in enumerate(dataloader):
    img = img.to(device)
model.train()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #choose adams

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    total_batches = 0

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    # Calculate and print the average loss per epoch
    average_loss = running_loss / total_batches
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

    
        
torch.save(model.state_dict(), 'resnet_tinyimage/resnet34_shadow_tinyimage_overtrained.pth')
print('Finished Training')
