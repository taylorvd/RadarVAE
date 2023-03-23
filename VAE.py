import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import pickle
import os
from depth_image_dataset import DepthImageDataset
import torch.nn as nn
import torch.nn.functional as F
#https://github.com/sksq96/pytorch-vae/blob/master/vae.py
#https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
class Encoder(nn.Module):
    def __init__(self, input_height, input_width, latent_size):
        super(Encoder, self).__init__()
        
       
        self.input_height = input_height
        self.input_width = input_width
        self.latent_size = latent_size
        
        h_dim = 1024#self.input_height*self.input_width // 2
        #starting value = total pixel size, input_height*input width
       
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc21 = nn.Linear(512, self.latent_size)
        self.fc22 = nn.Linear(512, self.latent_size)
       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

    # x = x.view(x.size(0), 256*conv_out_height*conv_out_width)
class Decoder(nn.Module):
    def __init__(self, input_height, input_width, latent_size):
        super(Decoder, self).__init__()
        
    
        self.input_height = input_height
        self.input_width = input_width
        self.latent_size = latent_size


        self.fc3 = nn.Linear(self.latent_size, 512)
        self.fc4 = nn.Linear(512, 128 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
    def forward(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = z.view(-1, 128, 8, 8)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        x_hat = torch.sigmoid(self.deconv3(z))
        return x_hat


class VAE(nn.Module):
    def __init__(self, latent_size=4, input_height = 64, input_width = 64):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_height, input_width, latent_size)
        self.decoder = Decoder(input_height, input_width, latent_size)

    #take random sampling and make into noise that is added in
    #https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important
    #https://www.youtube.com/watch?v=9zKuYvjFFS8
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        #stochastic parameter
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        #train mean and log over variance 
        mu, logvar = self.encoder(x)
        
        #latent space vector
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def loss_function(self, x_recon, x, mu, logvar):
        #same as regular autoencoder, except now sampling from distribution
        #recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')

        #keep learning distribution close to normal distribution
        kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_div_loss
        return loss

    #https://github.com/pytorch/examples/blob/main/vae/main.py
    #https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    def train_vae(self):
        model = VAE(latent_size=128)
        model.train()
        loss_fn = model.loss_function

        #train_dataset = ImageFolder('/home/taylorlv/RadarVAE/input/bagfiles/testbag.bag')
        #TODO change name to train.pkl
        os.system('/home/taylorlv/RadarVAE/data_parser.py /home/taylorlv/RadarVAE/input/bagfiles/testbag.bag /output')
        with open('my_dataset.pkl', 'rb') as f:
            train_dataset = pickle.load(f)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        
        size = len(train_loader)
        epochs = 50
        optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
        for epoch in range(epochs):
            #for batch_idx, (data) in enumerate(train_loader):
            running_loss = 0.0
            for i, data in enumerate(train_loader):

                optimizer.zero_grad()

                recon_data, mu, logvar = model(data)
                loss = loss_fn(recon_data, data, mu, logvar)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, epochs, i+1, len(train_loader), running_loss/100))
            running_loss = 0.0
            #print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        #if batch_idx % 1 == 0:
            #loss, current = loss.item(), (batch_idx + 1) * len(data)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
"""     def test_vae(self):
        model.eval()
        running_loss = 0.0

        loss_fn = model.loss_function

        #train_dataset = ImageFolder('/home/taylorlv/RadarVAE/input/bagfiles/testbag.bag')
        #TODO change name to test.pkl
        os.system('/home/taylorlv/RadarVAE/data_parser.py /home/taylorlv/RadarVAE/input/bagfiles/testbag.bag /output')
        with open('my_dataset.pkl', 'rb') as f:
            train_dataset = pickle.load(f)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        
        size = len(train_loader)
        epochs = 50
        optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
        for epoch in range(epochs):
            #for batch_idx, (data) in enumerate(train_loader):
            running_loss = 0.0
            for i, data in enumerate(train_loader):

                optimizer.zero_grad()

                recon_data, mu, logvar = model(data)
                loss = loss_fn(recon_data, data, mu, logvar)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, epochs, i+1, len(train_loader), running_loss/100))
            running_loss = 0.0 """