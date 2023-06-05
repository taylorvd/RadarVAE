import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Encoder(nn.Module):
    def __init__(self, image_width, image_height, latent_size, hidden_size):
        super(Encoder, self).__init__()

        self.latent_size = latent_size
        self.image_width = image_width
        self.image_height = image_height

        self.fc1 = nn.Linear(self.image_height * self.image_width, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.latent_size)

    def forward(self, x):
        x = x.view(-1, self.image_width * self.image_height)
        x = F.relu(self.fc1(x))
        z = self.fc2(x)
        return z

class Decoder(nn.Module):
    def __init__(self, image_width, image_height, latent_size, hidden_size):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        self.image_width = image_width
        self.image_height = image_height

        self.fc1 = nn.Linear(self.latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.image_height * self.image_width)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(z))
        x_recon = x_recon.view(-1, 1, self.image_height, self.image_width)
        return x_recon

class Autoencoder(nn.Module):
    def __init__(self, image_height, image_width, latent_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(image_height, image_width, latent_size, hidden_size)
        self.decoder = Decoder(image_height, image_width, latent_size, hidden_size)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def loss_function(self, x_recon, x):
        loss = F.mse_loss(x_recon, x)
        return loss

def train_autoencoder(model, train_dataloader, optimizer, epoch):
    model.train()
    loss_fn = model.loss_function

    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        recon_data = model(data)
        loss = loss_fn(recon_data, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_dataloader.dataset)
    return train_loss

def test_autoencoder(model, test_dataloader, epoch):
    model.eval()
    loss_fn = model.loss_function

    running_loss = 0.0
    for i, data in enumerate(test_dataloader):
        recon_data = model(data)
        loss = loss_fn(recon_data, data)
        running_loss += loss.item()

         
        if(i== 5 and epoch % 10 == 0):
            plt.figure()
            img = np.transpose(data[0].numpy(), [1,2,0])
            plt.subplot(121)
            plt.imshow(np.squeeze(img))

            outimg = np.transpose(recon_data[0].detach().numpy(), [1,2,0])
            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            plt.show()

    test_loss = running_loss / len(test_dataloader.dataset)
    return test_loss



