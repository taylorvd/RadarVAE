#!/usr/bin/env python3
from torch.utils.data import Dataset, DataLoader

# Custom dataset class that takes in list of tensors
#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class DepthImageDataset(Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list
        
    def __getitem__(self, index):
        return self.tensor_list[index]
        
    def __len__(self):
        return len(self.tensor_list)