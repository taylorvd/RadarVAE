


    #!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from VAE import *
from depth_image_dataset import *
import pickle
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def train_vae_tune(config, train_dataset, test_dataset, device):
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = VAE(image_height=8, image_width=8, latent_size=40, hidden_size=200, beta=config["beta"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_vae(model, train_dataloader, optimizer, epoch)
        validation_loss = test_vae(model, test_dataloader, epoch)

        tune.report(train_loss=train_loss, validation_loss=validation_loss)

    return {"train_loss": train_loss, "validation_loss": validation_loss}



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('./data/input/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)

    with open('./data/input/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    config = {
        "batch_size": tune.grid_search([32, 64]),
        "learning_rate": tune.grid_search([0.001, 0.01]),
        "num_epochs": tune.grid_search([30, 60]),
        "beta": tune.grid_search([0.01, 0.1])
    }

    scheduler = ASHAScheduler(
        metric="validation_loss",
        mode="min",
        max_t=100,
        grace_period=10,
        reduction_factor=2)

    reporter = CLIReporter()
    analysis = tune.run(
        lambda config: train_vae_tune(config, train_dataset, test_dataset, device),
        resources_per_trial={"cpu": 1, "gpu": 0},
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = analysis.get_best_trial("validation_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial validation loss: {}".format(best_trial.last_result["validation_loss"]))
    print("Best trial train loss: {}".format(best_trial.last_result["train_loss"]))

    # Save model
    model = train_vae_tune(best_trial.config, train_dataset, test_dataset, device)
    torch.save(model.state_dict(), "./model/vae.pth")


if __name__ == '__main__':
    main()

