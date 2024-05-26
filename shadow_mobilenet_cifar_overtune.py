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
import torchvision.models as models 

from helper_functions import train_loop

MULTI_THREATING_PROCESS_N = 0
TRAIN_NEW_MODEL = False
# Change the DATA_PATH to your local pickle file path
DATA_PATH = 'pickle/cifar10/mobilenetv2/shadow.p'
# Model state dictionary file
# Loading path
OLD_MODEL_PATH = 'shadown_models/mobilenet_shadow_cifar_overtrained.pth'
# Saving path
NEW_MODEL_PATH = 'shadow_models/mobilenet_shadow_cifar_overtrained.pth'
# Parameter
NUM_EPOCHS = 3
TRAIN_PERC = 0.5
LEARNING_RATE = 0.00001
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # sets to gpu if you have one

# load old stateditc
model =  models.mobilenet_v2(weights=None,num_classes=10).to(DEVICE)
# Load statedict of old trained model
if not TRAIN_NEW_MODEL: model.load_state_dict(torch.load(OLD_MODEL_PATH, map_location=DEVICE), strict=False) 

# Load dataset
with open(DATA_PATH, "rb") as f:
    dataset = pickle.load(f)
#splitting
#only use train set here
train_data, val_data = train_test_split(dataset, test_size=(1-TRAIN_PERC),shuffle=False)
dataloader = DataLoader(train_data, batch_size=128, shuffle=False, num_workers=MULTI_THREATING_PROCESS_N)
testloader =  DataLoader(val_data, batch_size=1, shuffle=True, num_workers=MULTI_THREATING_PROCESS_N)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #choose adams
# Training loop
train_loop(NUM_EPOCHS, model, dataloader, optimizer, criterion, DEVICE)


        
torch.save(model.state_dict(), NEW_MODEL_PATH)
print('Finished Training')
