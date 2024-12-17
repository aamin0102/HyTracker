import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from HASBS.band_25.dataset import ReconDataset
from HASBS.has_bs.has_bs import HAS_BS

# Define your MSE loss function
criterion = nn.MSELoss()
# Move model to GPU
model = HAS_BS(in_channels=25).cuda()
# Initialize your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Load your custom dataset
path = "D:\\hsi_2023_dataset\\training\\hsi\\nir"
weight_path = "../weights"
checkpoint_path = os.path.join(weight_path, "band25_checkpoint.pth")
dataset = ReconDataset(path)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define DataLoader for batching and shuffling the datasets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Training loop
num_epochs = 500
best_val_loss = float('inf')
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for batch_idx, data in enumerate(train_dataloader):
        inputs = data.cuda()  # Move input data to GPU
        targets = data.cuda()  # Move target data to GPU

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, _ = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        total_loss += loss.item()

        # Print the loss every few batches
        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.10f}")


    # Evaluate on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_data in val_dataloader:
            val_inputs = val_data.cuda()
            val_targets = val_data.cuda()
            val_outputs, _ = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item()

    val_loss /= len(val_dataloader)

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)
        print("Model saved at epoch", epoch + 1, "with validation loss:", best_val_loss)
