'''
Author: your name
Date: 2024-06-11 09:17:48
LastEditTime: 2024-06-11 09:17:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Step5\model\user_social_graph copy 2.py
'''

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pickle
from model.attention import SelfAttentionBatch, SelfAttentionSeq
from torch_geometric.nn import RGCNConv
from collections import defaultdict
import numpy as np
import math
import os
import json
class UserSocialGraph(nn.Module):
    def __init__(self, dir_data, embed_dim, user_embed_dim, weights_loc,num_bases, device, SELF_LOOP_ID=1000):
        super(UserSocialGraph, self).__init__()

        self.dir_data = dir_data
        self.embed_dim = embed_dim
        self.user_embed_dim = user_embed_dim
        self.device = device
        self.weights_loc= weights_loc
        self.SELF_LOOP_ID = SELF_LOOP_ID
        self.num_bases =num_bases
        self.initDataset()
        self.load_data()
        
        self._build_model()
    
    
    def initDataset(self):
        with open(os.path.join(self.dir_data, 'user_tie.json'), 'r', encoding='utf-8') as f:
            self.user_strength = json.load(f)

        with open(os.path.join(self.dir_data, 'user_social.json'), 'r', encoding='utf-8') as f:
            self.user_relations = json.load(f)
        
    
    def load_data(self):
        edge_list = []  # [(user, user)]
        for user, relations in self.user_relations.items():
            edge_list.append((user, user, self.SELF_LOOP_ID))  # add self loop
            for index,connectedd_user in enumerate(relations):
                edge_list.append((user , connectedd_user, self.user_strength[user][index]))
                #edge_list.append((connectedd_user , user))

       
        
        
        relation_cnt, relation2id, edges, self.nodes = defaultdict(int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if r not in relation2id:
                relation2id[r] = len(relation2id)
            edges.add((h, t, relation2id[r]))
            self.nodes.add(h)
            self.nodes.add(t)
         

        # Convert the set of edges into a list of lists
        edge_list = [[int(h), int(t), r] for h, t, r in edges]

        # Convert the list of lists into a tensor
        # edge_sets = torch.tensor(edge_list, dtype=torch.long)
        # x = edge_sets.shape
        # Extract the first two elements (head and tail nodes) for each edge
        self.edge_idx = [[h, t] for h, t, r in edge_list]

        # Extract the third element (edge type) for each edge
        self.edge_type = [r for h, t, r in edge_list]
        
       
        self.edge_idx = torch.tensor(self.edge_idx, dtype=torch.long).to(self.device)
     
        self.edge_idx = self.edge_idx.t()

        self.edge_type = torch.tensor(self.edge_type, dtype=torch.long).to(self.device)
        
       
        self.n_relation = len(relation2id)
       
        
        
    
  
    def _build_model(self):
        self._build_user_social_layer()
        self._build_user_social_project_head()
        self._build_user_esocial_project_head()
    
    def _build_user_social_layer(self):
        self.social_encoder = RGCNConv(len(self.nodes), self.user_embed_dim, self.n_relation, num_bases=self.num_bases)
        self.user_attn = SelfAttentionBatch(self.embed_dim, self.embed_dim)
     
    
    def _build_user_social_project_head(self):
        self.user_social_project_head_fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        #self.batch_norm1 = nn.BatchNorm1d(self.embed_dim)
        self.user_social_project_head_fc2 = nn.Linear(self.embed_dim, self.user_embed_dim)

    def _build_user_esocial_project_head(self):
        self.user_esocial_project_head_fc1 = nn.Linear(self.user_embed_dim * 2, self.embed_dim)
        self.batch_norm2 = nn.BatchNorm1d(self.embed_dim)
        self.user_esocial_project_head_fc2 = nn.Linear(self.embed_dim, self.user_embed_dim)    
    
   
    
    # get profile 
    def get_user_social_project_user_profile(self, batch):
        context_initiators = batch['initiator_ids']
        context_respondents = batch['respondent_ids']
        context_users = torch.stack((context_initiators, context_respondents), dim=0).t()
        user_social_rep = self._get_user_social_user_rep(context_users)  # (bs, ns_user * user_dim)
       
        #concatenate and then perform 
        context_users_attended =torch.stack([torch.cat((pair[0], pair[1]), dim=0) for pair in user_social_rep]).to(self.device)
        context_users_initiator = torch.stack([pair[0] for pair in user_social_rep]).to(self.device)
        context_users_respondent = torch.stack([pair[1] for pair in user_social_rep]).to(self.device)
        

        
        project_social_user_profile = self._project_user_esocial_user_project_profile(context_users_attended)  # (bs, user_dim)

        # respondent_social_user_profile = self.get_user_social_user_profile(context_respondents)  # (bs, dim)
        # project_respondent_social_user_profile = self._project_user_social_user_project_profile(respondent_social_user_profile)  # (bs, user_dim)

        return project_social_user_profile,context_users_initiator,context_users_respondent #  (bs, user_dim),  (bs, user_dim)
    
    


    def get_user_social_user_rep(self, batch):
        context_initiators = batch['initiator_ids']
        context_respondents = batch['respondent_ids']
        context_users = torch.stack((context_initiators, context_respondents), dim=0).t()
        user_social_rep = self._get_user_social_user_rep(context_users)  # (bs, ns_user * user_dim)
        context_users_attended =torch.stack([torch.cat((pair[0], pair[1]), dim=0) for pair in user_social_rep]).to(self.device)
        #project_social_user_profile = self._project_user_esocial_user_project_profile(context_users_attended)  # (bs, user_dim)

        context_users_initiator = torch.stack([pair[0] for pair in user_social_rep]).to(self.device)
        context_users_respondent = torch.stack([pair[1] for pair in user_social_rep]).to(self.device)

        # context_users_initiator = self._project_user_social_user_project_profile(context_users_initiator)
        # context_users_respondent = self._project_user_social_user_project_profile(context_users_respondent)
        return context_users_attended,context_users_initiator, context_users_respondent   # (bs, ns_user * user_dim)
    
    def get_user_social_user_profile(self, context_users):
    
        user_social_user_profile = self._get_user_social_user_profile(context_users)  # (bs, dim)

        return user_social_user_profile

    def _get_user_social_user_rep(self, context_users):
        social_embeddings = self.social_encoder(None,  self.edge_idx, self.edge_type)
        user_social_rep = self._encode_user(context_users, social_embeddings)  # (bs, ns_user * user_dim)
        return user_social_rep
    
    def _get_user_social_user_profile(self, context_users):
        social_embeddings = self.social_encoder(None, self.edge_idx, self.edge_type)
        user_social_user_profile = self._encode_user_profile(context_users, social_embeddings)  # (bs, dim)
        return user_social_user_profile


    def _encode_user_profile(self, context_users, social_embeddings):
        user_social_repr_list = []
        for user_list in context_users:
            if  user_list.numel() == 0:
                user_social_repr_list.append([torch.zeros( self.user_embed_dim, device=self.device), torch.zeros( self.user_embed_dim, device=self.device)])
                continue
            user_repr = social_embeddings[user_list]
            _ , user_repr = self.user_attn(user_repr)
            user_social_repr_list.append(user_repr)
        return torch.stack(user_social_repr_list, dim=0)  # (bs, dim)
    

    def _encode_user(self, context_users,social_embeddings):
        user_social_repr_list = []
        for user_list in context_users:
            if  user_list.numel() == 0:
                user_social_repr_list.append([torch.zeros(self.user_embed_dim, device=self.device), torch.zeros( self.user_embed_dim, device=self.device)])
                continue
            user_repr = social_embeddings[user_list]
            user_repr, _ = self.user_attn(user_repr)
            user_social_repr_list.append(user_repr)
        return user_social_repr_list  # (bs, ns_user * user_dim)

    def _project_user_social_user_project_profile(self, user_social_user_profile):
        user_social_user_profile = self.user_social_project_head_fc1(user_social_user_profile) # (bs, dim)
        user_social_user_profile = F.relu(user_social_user_profile) # (bs, dim)
        user_social_user_profile = self.user_social_project_head_fc2(user_social_user_profile) # (bs, user_dim)

        return user_social_user_profile # (bs, user_dim)
    
    def _project_user_esocial_user_project_profile(self, user_social_user_profile):
        user_social_user_profile = self.user_esocial_project_head_fc1(user_social_user_profile) # (bs, dim)
        if user_social_user_profile.size(0) > 1:
            user_social_user_profile = self.batch_norm2(user_social_user_profile)
        user_social_user_profile = F.relu(user_social_user_profile) # (bs, dim)
        user_social_user_profile = self.user_esocial_project_head_fc2(user_social_user_profile) # (bs, user_dim)

        return user_social_user_profile # (bs, user_dim)
    

    def get_all_embeddings(self):
        social_embedding = self.social_encoder(None, self.edge_idx, self.edge_type).to(self.device)
        return social_embedding
