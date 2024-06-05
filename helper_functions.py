import pickle
import torch
from os.path import isfile
from os import remove
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim, nn
from torch import device as d
from torch.nn.functional import one_hot
from enum import Enum

class DatasetClassN (Enum):
    Cifar = 10
    Tinyimage = 200

# Good Old Training function
def train_loop (num_epochs:int, model:nn.Module, dataloader:DataLoader, optimizer:optim, criterion:nn, device:d) -> nn.Module:
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

def print_first_sample (sample_number, inputs, labels, outputs, input_squeeze=None, one_hot=None, loss=None, top_sigmoids=None):
    print ("------SAMPLE WINDOW---------------------------------------------------------")
    print (f"Number Samples: {sample_number}")
    print (f"Batchsize: {inputs.size(0)}")
    inp_str = f"{inputs}"
    print ("Inputs:" + inp_str.replace('\n',"").replace("   ",""))
    if input_squeeze is not None:
        print (f"Squeezed Input: {input_squeeze}")
    labels_str = f"{labels}"
    print ("Labels:" + labels_str.replace('\n',"").replace("    ",""))
    outputs_str = f"{outputs}"
    print ("Outputs:" + outputs_str.replace('\n',"").replace("  ",""))
    if one_hot is not None:
        print (f"One hot:{one_hot}")
    if loss is not None:
        print (f"Loss:{loss}")
    if top_sigmoids is not None:
        print (f"Top Sigmoids: {top_sigmoids}")
    print ("---------------------------------------------------------------------------")

def standardize_dset (dataset:list[torch.tensor])->list[torch.tensor]:
    for i in range(3):
            #onvert all tensors to the same dtype first
        first_column_values = torch.cat([data[0][:, i] for data in dataset], dim=0)

        # Compute mean and std for the first column
        mean = first_column_values.mean()
        std = first_column_values.std()

        print("Mean of first column:", mean)
        print("Std of first column:", std)

        # Function to standardize only the first column
        def standardize_first_column(tensor, mean, std):
            tensor[:, 0] = (tensor[:, 0] - mean) / std
            return tensor

        # Normalize the first column in each tensor in the dataset
        standardized_dataset = [(standardize_first_column(data[0].clone(), mean, std), data[1]) for data in dataset]
        return standardized_dataset

####################################### MAIN FUNCTION

def train_or_load_and_eval_atck_model(model:nn.Module, train_loader:DataLoader, eval_loader:DataLoader, optimizer:optim, criterion:nn, epochs:int, save_path:str=None, device:str="cpu"):
    # Check if attack model file already available
    if save_path is not None and isfile (save_path):
        print (f'Attack model state dictionary already available, loading into input model from {save_path}!')
        model.load_state_dict(torch.load(save_path,map_location=torch.device('cpu'))) #CHANGE IF USING CUDA/CPU
    # Train + test on eval and Save attack model
    else:
        first_sample = True
        for epoch in range(epochs):
            print (f'Epoch: {epoch+1}')
            model.train()
            train_loss = 0.0
            sample_n = len(train_loader.sampler)
            for inputs, labels in train_loader:
                inputs = inputs.float().to(device)  # Ensures input tensors are floats
                labels = labels.float().to(device).view(-1, 1)  # Ensures labels are floats and reshaped correctly
                optimizer.zero_grad()
                outputs = model(inputs.squeeze(dim=1))
                loss = criterion(outputs, labels)
                if first_sample:
                    print_first_sample (sample_n, inputs, labels, outputs, inputs.squeeze(dim=1), None, loss)
                    first_sample = False
                loss.backward()
                optimizer.step()
                # update running training loss
                train_loss += loss.item()*inputs.size(0)
            print(f"Epoch Loss: {train_loss/sample_n}")
            evaluate_attack_model(model,eval_loader,device)
        print ("----------^^^^\nFinished Training!")
        # Save trained model if savepath not None
        if save_path is not None:
            print (f'Saving model state dictionary to path {save_path}!')
            try: 
                torch.save(model.state_dict(), save_path)
            except:
                print ("Saving failed due to exception!")
        else:
            print ('Not saving model as save_path is None!')
                
                
# Use this function to create a training dataset for the attack model, based on shadow training/testing datasets
def create_shadow_post_train_loader (non_memb_loader:DataLoader, memb_loader:DataLoader, shadow_model:nn.Module, batch_size:int, multi_n:int, data_class, device, save_path:str=None, standardize:bool=False)->DataLoader:
    if save_path is not None and isfile(save_path):
        print (f"Attack dataset was already established previosly, loading dataset from \"{save_path}\".")
        # Load already established dset
        with open(save_path, "rb") as att_dset_f:
            dataset_attack = pickle.load(att_dset_f)
    else:
        shadow_model.eval()
        dataset_attack = []
        onehot_l = []
        # NON Members
        with torch.no_grad():
            first_sample = True
            non_memb_num_samples = len(non_memb_loader.sampler)
            for images, labels in non_memb_loader: #need only one
                    # Move images and labels to the appropriate device
                images = images.to(device)
                # Forward pass
                logits = shadow_model(images)
                #take the 3 biggest logist
                top_values = torch.topk(logits, k=3).values
                sorted_sigmoids, _ = torch.sort(top_values.to("cpu"), dim=1, descending=True)
                class_enc = one_hot(labels, data_class.value)
                sigmoids_classes = torch.concat([sorted_sigmoids, class_enc], dim=1)
                dataset_attack.append([sigmoids_classes,0])
                # Logging
                if first_sample:
                    print ("NON Members")
                    print_first_sample(non_memb_num_samples, images, labels, logits, input_squeeze=None, one_hot=sigmoids_classes, loss=None, top_sigmoids=sorted_sigmoids)
                    first_sample = False
                
                

        # MEMBERS
        with torch.no_grad():
            first_sample = True
            memb_num_samples = len(non_memb_loader.sampler)
            for images, labels in memb_loader: #need only one
                    # Move images and labels to the appropriate device
                images = images.to(device)
                # Forward pass
                logits = shadow_model(images)
                #take the 3 biggest logist
                top_values = torch.topk(logits, k=3).values
                sorted_sigmoids, _ = torch.sort(top_values.to("cpu"), dim=1, descending=True)
                class_enc = one_hot(labels, data_class.value)
                sigmoids_classes = torch.concat([sorted_sigmoids, class_enc], dim=1)
                dataset_attack.append([sigmoids_classes,1])
                # Logging
                if first_sample:
                    print ("Members")
                    print_first_sample(memb_num_samples, images, labels, logits, input_squeeze=None, one_hot=sigmoids_classes, loss=None, top_sigmoids=sorted_sigmoids)
                    first_sample = False
        # Standardize
        if standardize:
            print ("Standarization")
            dataset_attack = standardize_dset(dataset_attack)
        else:
            print ("No Standarization")
        if save_path is not None:
            # Save dset
            with open (save_path, "wb") as att_dset_f:
                print (f"Saving attack model training dataset at \"{save_path}\"!")
                try: 
                    pickle.dump(dataset_attack, att_dset_f)
                except:
                    print ("Saving failed, due to exception!")
    attack_dtloader = DataLoader(dataset_attack, batch_size=batch_size, shuffle=True, num_workers=multi_n)

    return attack_dtloader


# Use this function to create a dataset of the target model posteriors
def create_eval_post_loader (target_model:nn.Module, eval_dataset:list, workers_n:int, data_class, device="cpu", test_dataset=False, standardize:bool=False)->DataLoader:
    target_model.eval()
    target_dataset_eval = []
    eval_dataloader = DataLoader(eval_dataset, batch_size=1 , shuffle=False, num_workers=1)
    with torch.no_grad():
        first_sample = True
        num_samples = len(eval_dataloader.sampler)
        # # FOR TEST DATASET ONLY
        # if test_dataset:
        #     for images, _ in eval_dataloader:
        #         # Move images and labels to the appropriate device
        #         images = images.to(device)
        #         # Forward pass
        #         logits = target_model(images)
        #         #take the 3 biggest logist
        #         top_values = torch.topk(logits, k=3).values #order poseri
        #         sorted_tensor, _ = torch.sort(top_values.to("cpu"), dim=1,descending=True)
        #         if first_sample:
        #                 print_first_sample(num_samples, images, None, logits, input_squeeze=None, loss=None, top_sigmoids=sorted_tensor)
        #                 first_sample = False
        #         target_dataset_eval.append(sorted_tensor)
        # else:
        for images, labels, member in eval_dataloader: #need only one
            # Move images and labels to the appropriate device
            images = images.to(device)
            # Forward pass
            logits = target_model(images)
            #take the 3 biggest logist
            top_values = torch.topk(logits, k=3).values #order poseri
            sorted_tensor, _ = torch.sort(top_values.to("cpu"), dim=1,descending=True)
            # Encoding
            class_enc = one_hot(labels, data_class.value)
            sigmoids_classes = torch.concat([sorted_tensor, class_enc], dim=1)
            if first_sample:
                    print_first_sample(num_samples, images, member, logits, input_squeeze=None, one_hot=sigmoids_classes, loss=None, top_sigmoids=sorted_tensor)
                    first_sample = False
            target_dataset_eval.append([sigmoids_classes, member.item()])
    # Standardize
    if standardize:
        print ("Standarization")
        target_dataset_eval = standardize_dset(target_dataset_eval)
    else:
        print ("No Standarization")
    target_eval_dl = torch.utils.data.DataLoader(target_dataset_eval, batch_size=64 , shuffle=False, num_workers=workers_n)
    return target_eval_dl

def evaluate_attack_model(model: nn.Module, post_memb_loader: DataLoader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for sorted_logits, members in post_memb_loader:
            sorted_logits = sorted_logits.to(device).float()
            members = members.to(device).float()

            outputs = model(sorted_logits.squeeze(dim=1))
            predicted = torch.round(outputs)  # Round the outputs to 0 or 1
            total += members.size(0)  # Increment the total count by batch size
            for idx, pred_label in enumerate(predicted):
                if members[idx] == pred_label:
                    correct += 1  # Count correct predictions
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.2f} with Correct: {correct} and Total: {total}')
    return accuracy  # Return the ac
        
