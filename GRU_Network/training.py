import os

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from dataset import CustomDataset
from model import GRUModel


import torch

# Create a DataLoader to load the data in batches
batch_size = 32
dataset = CustomDataset(data_dir="dataset_6")
# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define DataLoader for batching and shuffling the datasets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model
num_features = 4  # x, y, w, h for each bounding box
input_size = num_features
hidden_size = 64
num_layers = 1
output_size = num_features


model = GRUModel(input_size, hidden_size, num_layers, output_size, num_gru=6).cuda()
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


weight_path = "../weights"
checkpoint_path = os.path.join(weight_path, "gru.pth")

# Train the model
num_epochs = 100
best_val_loss =  float("Inf")
for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()
    for i, (inputs, targets) in enumerate(train_dataloader):
        # Convert inputs and targets to tensors
        inputs = torch.tensor(inputs, dtype=torch.float32).cuda()
        targets = torch.tensor(targets, dtype=torch.float32).cuda()

        # Forward pass
        # print(inputs.shape)
        outputs, loss = model(inputs, targets)

        # Compute loss
        # loss = criterion(outputs, targets)
        # loss = iou_loss(outputs, targets)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100==0:
            print("epoch:", epoch + 1, "batch:", i+1, "training loss:", loss.item())

    # Print average loss for the epoch
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs,targets in val_dataloader:
            val_inputs = torch.tensor(inputs).cuda()
            val_targets = torch.tensor(targets, dtype=torch.float32).cuda()

            val_outputs = model.get_output(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item()

    val_loss /= len(val_dataloader)
    #writer.add_scalar('Validation Loss', val_loss, epoch)

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)
        print("Model saved at epoch", epoch + 1, "with validation loss:", best_val_loss)




# model = model.cpu()
# torch.save(model.state_dict(), 'model_bbox_10.pth')