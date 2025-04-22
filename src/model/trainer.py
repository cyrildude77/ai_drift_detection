import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train_model(model, dataloader, n_epochs=50, lr=0.001, device='cpu', save_path='models/lstm_autoencoder.pth'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader):
            # Unpack the batch (DataLoader yields a tuple of tensors)
            batch = batch[0].to(device)  # Extract the first tensor
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{n_epochs}] - Loss: {running_loss/len(dataloader):.6f}")

    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")