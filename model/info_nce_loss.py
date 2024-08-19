## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np







def info_nce_loss(features, bs, n_views, device, temperature):
    """ from https://github.com/Zyh716/WSDM2022-C2CRS
        features = (n_views*bs, dim)
        n_views: rgcn, transformer's view on user_rep
    """
    labels = torch.cat([torch.arange(bs) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    x = similarity_matrix.shape
    assert similarity_matrix.shape == (n_views * bs, n_views * bs)
    assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)


    # Scale up the logits for hard positives
    # Identify hard positives as those with lower similarity scores
    # hard_positives = positives < positives.mean(dim=1, keepdim=True)
    

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels
