import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#https://github.com/sksq96/pytorch-vae/blob/master/vae.py
#https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
class Encoder(nn.Module):
    def __init__(self, image_width, image_height, latent_size, hidden_size):
        super(Encoder, self).__init__()
 
        self.latent_size = latent_size
        self.image_width = image_width
        self.image_height = image_height

        self.fc1 = nn.Linear(self.image_height* self.image_width, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, self.latent_size)
        self.fc_logvar = nn.Linear(hidden_size, self.latent_size)
       

    def forward(self, x):
        x = x.view(-1, self.image_width * self.image_height)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

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
        z = F.sigmoid(self.fc2(z))
        z = z.view(-1, 1, self.image_height, self.image_width)
        return z


class VAE(nn.Module):
    def __init__(self, image_height, image_width, latent_size, hidden_size, beta):
        super(VAE, self).__init__()
        self.encoder = Encoder(image_height, image_width, latent_size, hidden_size)
        self.decoder = Decoder(image_height, image_width,latent_size, hidden_size)
        self.beta = beta
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
        kl_div_loss = -0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta*kl_div_loss
        return loss

    #https://github.com/pytorch/examples/blob/main/vae/main.py
    #https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def train_vae(model, train_dataloader, optimizer, epoch):
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
        if(i== 5 and epoch % 50 == 0):
            plt.figure()
            img = np.transpose(data[0].numpy(), [1,2,0])
            plt.subplot(121)
            plt.imshow(np.squeeze(img))

            outimg = np.transpose(recon_data[0].detach().numpy(), [1,2,0])
            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            plt.show()
    train_loss = running_loss / len(train_dataloader.dataset)
    return train_loss



#https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
def test_vae(model, test_dataloader, epoch):
    model.eval()
    loss_fn = model.loss_function
    
    running_loss = 0.0
    
    for i, data in enumerate(test_dataloader):

       
        

        recon_data, mu, logvar = model(data)
        loss = loss_fn(recon_data, data, mu, logvar)
        running_loss += loss.item()

        

        
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