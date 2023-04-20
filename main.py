#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from VAE import *
from depth_image_dataset import *
import pickle

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 70

    # Load data
    with open('./data/input/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    

    with open('./data/input/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    model = VAE(image_height = 64, image_width = 64, latent_size = 20, hidden_size=500, beta = 0).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_vae(model, train_dataloader, optimizer, epoch)
        validation_loss = test_vae(model, test_dataloader, epoch)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {validation_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "./model/vae.pth")

if __name__ == '__main__':
    main()