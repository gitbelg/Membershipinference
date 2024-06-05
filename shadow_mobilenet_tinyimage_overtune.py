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

# Turn up number, if computer supports this 
MULTI_THREATING_PROCESS_N = 0
TRAIN_NEW_MODEL = False
# Change the DATA_PATH to your local pickle file path
DATA_PATH = 'pickle/tinyimagenet/mobilenetv2/shadow.p'
# Model state dictionary file
# Loading path
OLD_MODEL_PATH = 'shadow_models/mobilenet_shadow_tinyimage_overtrained.pth'
# Saving path
NEW_MODEL_PATH = 'shadow_models/mobilenetv2_shadow_tinyimage_overtrained.pth'
# Parameter
NUM_EPOCHS = 5
TRAIN_PERC = 0.5
LEARNING_RATE = 0.001
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # sets to gpu if you have one

model =  models.mobilenet_v2(weights=None,num_classes=200).to(DEVICE)
if not TRAIN_NEW_MODEL: model.load_state_dict(torch.load(OLD_MODEL_PATH,map_location=torch.device('cpu')), strict=False)
with open(DATA_PATH, "rb") as f:
    dataset = pickle.load(f)
#splitting
#only use train set here
train_data, val_data = train_test_split(dataset, test_size=(1-TRAIN_PERC),shuffle=False)
dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=MULTI_THREATING_PROCESS_N)
testloader =  torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, num_workers=MULTI_THREATING_PROCESS_N)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #choose adams

train_loop(NUM_EPOCHS, model, dataloader, optimizer, criterion, DEVICE)    
torch.save(model.state_dict(), NEW_MODEL_PATH)
print('Finished Training')
