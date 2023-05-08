#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from VAE import *
from depth_image_dataset import *
import pickle
import argparse
import ray
from ray import tune
import struct
from cv_bridge import CvBridge
import numpy as np
import torch
from data_parser import ros_depth_image_to_torch
from rosbag import Bag
from torch.utils.data import Subset
import random

# define the train function
def tune_vae(config, train_dataset, test_dataset):
    batch_size = 64
    image_height = 8
    image_width = 8

    # initialize the model
    model = VAE(image_height, image_width, latent_size = config["latent_size"], hidden_size=config["hidden_size"], beta=config["beta"])
    
    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    """
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
    """
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
        
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                recon_data, mu, logvar = model(data)
                loss = model.loss_function(recon_data, data, mu, logvar)
                test_loss += loss.item()
            test_loss /= len(test_dataloader)
        
        tune.report(test_loss=test_loss)
def main():
    # Set device
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="train or tune")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set hyperparameters
    batch_size = 64
    num_epochs = 150

    if args.mode == "tune":
        config = {
            "lr": tune.grid_search([0.0001, 0.0005, 0.001]),
            "latent_size": tune.grid_search([10, 20, 30]),
            "epochs": tune.grid_search([120]),
            "beta": tune.grid_search([0, 0.01, 0.001]),
            "hidden_size": tune.grid_search([64, 128, 200, 300])
        }

        train_dataset, test_dataset = load_datasets()

        # initialize Ray Tune
        ray.init()
        analysis = tune.run(
            lambda config: tune_vae(config, train_dataset, test_dataset),
            config=config,
            metric="test_loss",
            mode="min"
        )
       
        best_config = analysis.get_best_config(metric="test_loss", mode="min")
        print("Best config:", best_config)
 
    elif args.mode == "train":
        # Load data
        train_dataset, test_dataset = load_datasets()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        print(len(test_dataset))
        print(len(test_dataset)/batch_size)

        # Initialize model and optimizer
        model = VAE(image_height=8, image_width=8, latent_size=20, hidden_size=300, beta=0.001).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

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
        #print(train_dataset.size())
    
    with open('./data/input/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
        #print(test_dataset.size())
        #print(test_dataset)
        #.permute(0, 2, 3, 1)

    #print(temp[0][0])
    #print(type(temp))
    #print(len(temp))
    #print(type(temp[0][0]))
    #print((temp[0][0].shape))

    #test_dataset = DepthImageDataset(tensor_list)

   
    # with open('./data/input/test_dataset.pkl', 'wb') as f:
    #         pickle.dump(test_dataset, f)

    # with open('./data/input/test_dataset.pkl', 'rb') as f:
    #     test_dataset_2 = pickle.load(f)
    #     print(len(test_dataset_2[0][0]))
    #     print(len(test_dataset_2))
        
    # for i, img in enumerate(test_dataset_2):
    #     print(f"Test Image {i} has size {img.shape}")
    

    return train_dataset, test_dataset


if __name__ == '__main__':
    main()
