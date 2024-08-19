
## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14

from torch.utils.data import DataLoader, Dataset
import torch

class SingleDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

    
    

    