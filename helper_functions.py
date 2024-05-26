import pickle
import torch
from os.path import isfile
from os import remove
import torch.nn as nn
from torch.utils.data import dataloader
from torch import optim, nn
from torch import device as d
from enum import Enum

class DatasetClassN (Enum):
    Cifar = 10
    Tinyimage = 200

def train_loop (num_epochs:int, model:nn.Module, dataloader:dataloader, optimizer:optim, criterion:nn, device:d) -> nn.Module:
    model.train()
    print ("Starting Training!")
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = 0

        for data in dataloader:
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


def train_or_load(model:nn.Module, train_loader:dataloader, optimizer:optim, criterion:nn, epochs:int, save_path:str=None):
    if save_path is not None and isfile (save_path):
        print (f'Attack model state dictionary already available, loading into input model from {save_path}!')
        model.load_state_dict(torch.load(save_path))
    else:
        model.train()
        for epoch in range(epochs):
            print (f'Epoch: {epoch+1}')
            for idx, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.float()  # Ensures input tensors are floats
                labels = labels.float().view(-1, 1)  # Ensures labels are floats and reshaped correctly

                optimizer.zero_grad()
                outputs = model(inputs.squeeze(dim=1))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if idx % (50*epochs) == 0:
                    print (inputs[0])
                    print (labels[0])
                    print(f"Batch {idx} with Loss: {loss.item()}")
        # Save trained model if savepath not None
        if save_path is not None:
            print (f'Finished Training\n Saving model state dictionary to path {save_path}!')
            try: 
                torch.save(model.state_dict(), save_path)
            except:
                print ("Saving failed due to exception!")
                
                
############################################ FOR JUPYTER NOTEBOOOK
# Use this function to create a training dataset for the attack model, based on shadow training/testing datasets
def create_post_train_loader (non_memb_loader:dataloader, memb_loader:dataloader, shadow_model:nn.Module, batch_size:int, multi_n:int, device, save_path:str=None)->dataloader:
    if save_path is not None and isfile(save_path):
        print (f"Attack dataset was already established previosly, loading dataset from {save_path}.")
        # Load already established dset
        with open(save_path, "rb") as att_dset_f:
            dataset_attack = pickle.load(att_dset_f)
    else:
        shadow_model.eval()
        dataset_attack = []
        # NON Members
        with torch.no_grad():
            for images, labels in non_memb_loader: #need only one
                    # Move images and labels to the appropriate device
                images, labels = images.to(device), labels.to(device)
                    # Forward pass
                logits = shadow_model(images)
                #take the 3 biggest logist
                top_values = torch.topk(logits, k=3).values
                top_values, indices = torch.sort(top_values, dim=1, descending=True)
                dataset_attack.append([top_values,0])
        # MEMBERS
        with torch.no_grad():
            for images, labels in memb_loader: #need only one
                    # Move images and labels to the appropriate device
                images, labels = images.to(device), labels.to(device)
                    # Forward pass
                logits = shadow_model(images)
                #take the 3 biggest logist
                top_values = torch.topk(logits, k=3).values
                top_values, _ = torch.sort(top_values, dim=1, descending=True)
                dataset_attack.append([top_values,1])
        if save_path is not None:
            # Save dset
            with open (save_path, "wb") as att_dset_f:
                print (f"Saving attack model training dataset at {save_path}!")
                try: 
                    pickle.dump(dataset_attack, att_dset_f)
                except:
                    print ("Saving failed, due to exception!")
    attack_dtloader = torch.utils.data.DataLoader(dataset_attack, batch_size=batch_size, shuffle=True, num_workers=multi_n)

    return attack_dtloader

# Use this function to create a dataset of the target model posteriors
def create_eval_post_loader (target_model:nn.Module, eval_dataloader:dataloader, multi_n:int, device)->dataloader:
    target_dataset_eval = []
    with torch.no_grad():
        for images,_, member in eval_dataloader: #need only one
            # Move images and labels to the appropriate device
            images = images.to(device)
            # Forward pass
            logits = target_model(images)
            #take the 3 biggest logist
            top_values = torch.topk(logits, k=3).values #order poseri
            sorted_tensor, _ = torch.sort(top_values, dim=1,descending=True)
            target_dataset_eval.append([sorted_tensor, member.item()])
    target_eval_dl = torch.utils.data.DataLoader(target_dataset_eval, batch_size=1 , shuffle=False, num_workers=multi_n)
    return target_eval_dl

def evaluate_attack_model(model:nn.Module, post_memb_loader:dataloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for sorted_logits, members in post_memb_loader:
            sorted_logits = sorted_logits.float()

            outputs = model(sorted_logits.squeeze(dim=1))
            predicted = torch.round(outputs)  # Round the outputs to 0 or 1
            total += members.size(0)  # Increment the total count by batch size
            correct += (predicted == members).sum().item()  # Count correct predictions

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.2f}')
    return accuracy
        
