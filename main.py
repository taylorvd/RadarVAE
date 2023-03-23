#!/usr/bin/env python3
from VAE import VAE
from depth_image_dataset import DepthImageDataset
def main():
   
    vae = VAE()
    vae.train_vae()










if __name__ == '__main__':
    main()