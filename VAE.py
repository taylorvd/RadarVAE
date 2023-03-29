import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#https://github.com/sksq96/pytorch-vae/blob/master/vae.py
#https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
 
        self.latent_size = latent_size

        #1 channel because grayscale, not RGB
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 *8 *8, 512)
        self.fc21 = nn.Linear(512, self.latent_size)
        self.fc22 = nn.Linear(512, self.latent_size)
       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 *8 *8)
        x = F.relu(self.fc1(x))

        mu = self.fc21(x)
        logvar = self.fc22(x)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()
        
        self.latent_size = latent_size

        self.fc3 = nn.Linear(self.latent_size, 512)
        self.fc4 = nn.Linear(512, 128 *8 *8)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = z.view(-1, 128,8,8)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))

        x_hat = torch.sigmoid(self.deconv3(z))

        return x_hat


class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

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
        kl_div_loss = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_div_loss
        return loss

    #https://github.com/pytorch/examples/blob/main/vae/main.py
    #https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def train_vae(model, train_dataloader, optimizer):
    model.train()
    loss_fn = model.loss_function

    running_loss = 0.0
    for i, data in enumerate(train_dataloader):

        optimizer.zero_grad()
        recon_data, mu, logvar = model(data)
        loss = loss_fn(recon_data, data, mu, logvar)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_dataloader.dataset)
    return train_loss




def test_vae(model, test_dataloader, device, epoch):
    model.eval()
    loss_fn = model.loss_function
    plt.figure()
    running_loss = 0.0
    for i, data in enumerate(test_dataloader):

        imgs = data
        imgs = imgs.to(device)
        img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
        plt.subplot(121)
        plt.imshow(np.squeeze(img))

        recon_data, mu, logvar = model(data)
        loss = loss_fn(recon_data, data, mu, logvar)
        running_loss += loss.item()

        outimg = np.transpose(recon_data[0].cpu().detach().numpy(), [1,2,0])
        plt.subplot(122)
        plt.imshow(np.squeeze(outimg))
    if(epoch%10 == 0):
        plt.show()
    test_loss = running_loss / len(test_dataloader.dataset)
    return test_loss


"""
def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()  
"""