'''
Author: “glad47” ggffhh3344@gmail.com
Date: 2024-03-18 18:53:57
LastEditors: “glad47” ggffhh3344@gmail.com
LastEditTime: 2024-04-26 16:07:58
FilePath: \GraphRec\Social_Aggregators.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from model.attention import Attention


class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, u2e, embed_dim, cuda="cpu"):
        super(Social_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, to_neighs):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            # 
            #e_u = self.u2e.weight[list(tmp_adj)] # fast: user embedding 
            #slow: item-space user latent factor (item aggregation)
            feature_neigbhors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
            e_u = torch.t(feature_neigbhors)
            num_feature_neigbhors = len(e_u)

            #u_rep = self.u2e.weight[nodes[i]]
            u_rep = self.features(torch.LongTensor([nodes[i]]).to(self.device))

            att_w = self.att(e_u, u_rep.squeeze(), num_neighs)
            att_history = torch.mm(e_u.t(), att_w).t()
            embed_matrix[i] = att_history
        to_feats = embed_matrix

        return to_feats
