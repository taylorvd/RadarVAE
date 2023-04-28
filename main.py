#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from VAE import *
from depth_image_dataset import *
import pickle
import argparse
import ray
from ray import tune

# define the train function
def tune_vae(config, train_dataset, test_dataset):
    batch_size = 64
    latent_size = config["latent_size"]
    image_height = 8
    image_width = 8
    hidden_size = 300

    # initialize the model
    model = VAE(image_height, image_width, latent_size, hidden_size=config["hidden_size"], beta=config["beta"])
    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(config["epochs"]):
        train_loss = 0
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            recon_data, mu, logvar = model(data)
            loss = model.loss_function(recon_data, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        tune.report(mean_loss=train_loss)
def main():
    # Set device
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="train or tune")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set hyperparameters
    batch_size = 64
    learning_rate = 0.003
    num_epochs = 120

    if args.mode == "tune":
        config = {
            "lr": tune.grid_search([0.001, 0.003, 0.005, 0.01]),
            "latent_size": tune.grid_search([15,20, 30, 40, 50]),
            "epochs": tune.grid_search([100, 110, 120]),
            "beta": tune.grid_search([0, 0.001, 0.01]),
            "hidden_size": tune.grid_search([50, 100, 150, 200, 300])

        }

        train_dataset, test_dataset = load_datasets()
        # initialize Ray Tune
        ray.init()
        analysis = tune.run(
            lambda config: tune_vae(config, train_dataset, test_dataset),
            config=config,
            metric="mean_loss",
            mode="min"
        )

       
        best_config = analysis.get_best_config(metric="mean_loss", mode="min")
        print("Best config:", best_config)
 

    elif args.mode == "train":
        # Load data
        train_dataset, test_dataset = load_datasets()
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model and optimizer
        model = VAE(image_height=8, image_width=8, latent_size=30, hidden_size=300, beta=0).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(1, num_epochs + 1):
            # tune.run(train_vae(model, train_dataloader, optimizer, epoch), config={"lr": tune.grid_search([0.001, 0.01, 0.1])})
            train_loss = train_vae(model, train_dataloader, optimizer, epoch)
            validation_loss = test_vae(model, test_dataloader, epoch)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {validation_loss:.4f}")

        # Save model
        torch.save(model.state_dict(), "./model/vae.pth")


def load_datasets():
    with open('./data/input/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('./data/input/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    return train_dataset, test_dataset


if __name__ == '__main__':
    main()
