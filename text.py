'''
Author: “glad47” ggffhh3344@gmail.com
Date: 2024-05-24 21:17:52
LastEditors: “glad47” ggffhh3344@gmail.com
LastEditTime: 2024-05-28 15:42:54

'''
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)
print(device)