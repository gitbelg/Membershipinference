from torchvision.models import resnet34, mobilenet_v2
from torchvision import models
from torch.utils.data import dataloader
from torch import optim, nn, device

def train_loop (num_epochs:int, model:models, dataloader:dataloader, optimizer:optim, criterion:nn, device:device) -> models:
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
    return model

def train_model_save (modeltype:models):
    print (modeltype)
    return 