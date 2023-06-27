import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

NUM_ITER = 0
#https://github.com/sksq96/pytorch-vae/blob/master/vae.py
#https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
class Encoder(nn.Module):
    def __init__(self, image_width, image_height, latent_size, num_layers):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.image_width = image_width
        self.image_height = image_height
        self.num_layers = num_layers

        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        self.fc_mu_2 = nn.Linear(16 * (image_height // 2) * (image_width // 2), latent_size)
        self.fc_logvar_2 = nn.Linear(16 * (image_height // 2) * (image_width // 2), latent_size)
        
        self.fc_mu_3 = nn.Linear(32 * (image_height // 4) * (image_width // 4), latent_size)
        self.fc_logvar_3 = nn.Linear(32 * (image_height // 4) * (image_width // 4), latent_size)

        self.fc_mu_4 = nn.Linear(64 * (image_height // 8) * (image_width // 8), latent_size)
        self.fc_logvar_4 = nn.Linear(64 * (image_height // 8) * (image_width // 8), latent_size)


        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):

        if (self.num_layers == 2):
        
            x = x.view(-1, 1, self.image_height, self.image_width)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(-1, 16 * (self.image_height // 2) * (self.image_width // 2))
            mu = self.fc_mu_2(x)
            logvar = self.fc_logvar_2(x)
            return mu, logvar
        
        elif (self.num_layers == 3):
            x = x.view(-1, 1, self.image_height, self.image_width)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(-1, 32 * (self.image_height // 4) * (self.image_width // 4))
            mu = self.fc_mu_3(x)
            logvar = self.fc_logvar_3(x)
            return mu, logvar
        
        else:
            x = x.view(-1, 1, self.image_height, self.image_width)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = x.view(-1, 64 * (self.image_height // 8) * (self.image_width // 8))
            mu = self.fc_mu_4(x)
            logvar = self.fc_logvar_4(x)
            return mu, logvar

class Decoder(nn.Module):
    def __init__(self, image_width, image_height, latent_size, num_layers):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.image_width = image_width
        self.image_height = image_height
        self.num_layers = num_layers

        
        self.fc4 = nn.Linear(latent_size, 64 * (image_height // 8) * (image_width // 8))
        self.fc3 = nn.Linear(latent_size, 32 * (image_height // 4) * (image_width // 4))
        self.fc2 = nn.Linear(latent_size, 16 * (image_height // 2) * (image_width // 2))

        
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1)




    def forward(self, z):
        if (self.num_layers == 2):
            z = F.relu(self.fc2(z))
            z = z.view(-1, 16, self.image_height // 2, self.image_width // 2)
            z = F.relu(self.conv2(z))
            z = torch.sigmoid(self.conv1(z))
            return z.view(-1, 1, self.image_height, self.image_width)
        
        elif(self.num_layers == 3):
            z = F.relu(self.fc3(z))
            z = z.view(-1, 32, self.image_height // 4, self.image_width // 4)
            z = F.relu(self.conv3(z))
            z = F.relu(self.conv2(z))
            z = torch.sigmoid(self.conv1(z))
            return z.view(-1, 1, self.image_height, self.image_width)
        
        else:
            z = F.relu(self.fc4(z))
            z = z.view(-1, 64, self.image_height // 8, self.image_width // 8)
            z = F.relu(self.conv4(z))
            z = F.relu(self.conv3(z))
            z = F.relu(self.conv2(z))
            z = torch.sigmoid(self.conv1(z))
            return z.view(-1, 1, self.image_height, self.image_width)


class VAE(nn.Module):
    def __init__(self, image_height, image_width, latent_size, num_layers, beta):
        super(VAE, self).__init__()
        self.encoder = Encoder(image_height, image_width, latent_size, num_layers)
        self.decoder = Decoder(image_height, image_width,latent_size, num_layers)
        self.beta = beta
        self.num_iter = 0
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
        return x_recon, z, mu, logvar

    def loss_function(self, x_recon, x, mu, logvar):
        #same as regular autoencoder, except now sampling from distribution

        recon_loss =nn.functional.mse_loss(x_recon, x, reduction='sum')

        #error = torch.square(x_recon - x)
        #weighted_error = (1+x) * error
        
        #recon_loss = torch.sum(weighted_error)

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
        # print(data[0].max().item(), data[0].mean().item())
        recon_data, z, mu, logvar = model(data)
        loss = loss_fn(recon_data, data, mu, logvar)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # if(i== 5 and epoch % 10 == 0):
        #     plt.figure()
        #     img = np.transpose(data[0].numpy(), [1,2,0])
        #     plt.subplot(121)
        #     plt.imshow(np.squeeze(img))

        #     outimg = np.transpose(recon_data[0].detach().numpy(), [1,2,0])
        #     plt.subplot(122)
        #     plt.imshow(np.squeeze(outimg))
        #     plt.show()
        
    train_loss = running_loss / len(train_dataloader.dataset)
    return train_loss



#https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
def test_vae(model, test_dataloader, epoch):
    model.eval()
    loss_fn = model.loss_function
    
    running_loss = 0.0

    for i, data in enumerate(test_dataloader):
        #TODO [batch_size 8 8 1] -> [batch_size 1 8 8]
        #data = data.permute(0, 3, 1, 2)
      
        recon_data, z, mu, logvar = model(data)
        loss = loss_fn(recon_data, data, mu, logvar)
        running_loss += loss.item()
        
        # print("max input", data[0].max(), " min input", recon_data[0].min(), " avg input", recon_data[0].mean())
        # print("max recon", recon_data[0].max(), " min recon", recon_data[0].min(), " avg recon", recon_data[0].mean())

        if(i % 50 == 0 and epoch % 25 == 0):
            #print(data[0].detach().numpy())
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