import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class ShallowEncoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(ShallowEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, latent_size)

    def forward(self, x):
        x = self.fc1(x)
        return x

class ShallowDecoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(ShallowDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, input_size)

    def forward(self, x):
        x = self.fc1(x)
        return x

class VAE(nn.Module):
    def __init__(image_height, image_width, latent_size, hidden_size, beta):
        self.encoder = ShallowEncoder(self.input_size, self.latent_size)
        self.decoder = Decoder(input_size, latent_size)
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def loss_function(self, x_recon, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl_div_loss
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
      
        recon_data, mu, logvar = model(data)
        loss = loss_fn(recon_data, data, mu, logvar)
        running_loss += loss.item()
        
        if(i % 20 == 0 and epoch % 20 == 0):
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