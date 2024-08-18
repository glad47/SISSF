
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

class UserSocialGraph(nn.Module):
    def __init__(self, dir_data, embed_dim, user_embed_dim, users,num_bases, device, SELF_LOOP_ID=185):
        super(UserSocialGraph, self).__init__()

        self.dir_data = dir_data
        self.embed_dim = embed_dim
        self.user_embed_dim = user_embed_dim
        self.users = users
        self.device = device
        self.history_u_lists, self.history_ur_lists, self.history_v_lists, self.history_vr_lists,self.train_u,self.train_v, self.train_r, self.test_u, self.test_v, self.test_r, self.social_adj_lists, self.ratings_list = self.initDataset()
        self.SELF_LOOP_ID = SELF_LOOP_ID
        self.num_bases =num_bases
        self.num_users = self.history_u_lists.__len__()
        self.num_items = self.history_v_lists.__len__()
        self.num_ratings = self.ratings_list.__len__()
        self.load_data()
        
        self._build_model()
    
    
    def initDataset(self):
        path_data = self.dir_data + "/redial_interactions.pickle"
        data_file = open(path_data, 'rb')
        return pickle.load(data_file)
    
    def load_data(self):
        edge_list = []  # [(user, user)]
        for user, relations in self.social_adj_lists.items():
            edge_list.append((user, user, self.SELF_LOOP_ID))  # add self loop
            user_interactedd_list = self.history_u_lists[user].tolist()
            user_interactedd_ratings = self.history_ur_lists[user]
            for connectedd_user in relations:
                connectedd_user_interactedd_list= self.history_u_lists[connectedd_user].tolist()
                connectedd_user_interactedd_ratings = self.history_ur_lists[connectedd_user]
                intersection = list(set(user_interactedd_list) & set(connectedd_user_interactedd_list))
                user_indices = [user_interactedd_list.index(value) for value in intersection]
                connectedd_user_indices = [connectedd_user_interactedd_list.index(value) for value in intersection]
                user_ratings = sum([user_interactedd_ratings[i] for i in user_indices])
                connectedd_user_ratings = sum([connectedd_user_interactedd_ratings[i] for i in connectedd_user_indices])

                r = (user_ratings + connectedd_user_ratings) / len(intersection)
                edge_list.append((user , connectedd_user, r))
                #edge_list.append((connectedd_user , user))

       
        edges, nodes = set(), set()
        
        for h, t, r in edge_list:
            edges.add((h, t, self.ratings_list[5] if r == 185 else self.ratings_list[5 if math.floor(r) > 5 else math.floor(r)]))
            nodes.add(h)
            nodes.add(t)


        # Convert the set of edges into a list of lists
        edge_list = [[h, t, r] for h, t, r in edges]

        # Convert the list of lists into a tensor
        # edge_sets = torch.tensor(edge_list, dtype=torch.long)
        # x = edge_sets.shape
        # Extract the first two elements (head and tail nodes) for each edge
        self.edge_idx = [[h, t] for h, t, r in edge_list]

        # Extract the third element (edge type) for each edge
        self.edge_type = [r for h, t, r in edge_list]
        
       
        self.edge_idx = torch.tensor(self.edge_idx, dtype=torch.long).to(self.device)
        x= self.edge_idx.shape
        self.edge_idx = self.edge_idx.t()
        y= self.edge_idx.shape
        self.edge_type = torch.tensor(self.edge_type, dtype=torch.long).to(self.device)
        
       
        self.n_relation = len(self.ratings_list)
       
        
        
    
  
    def _build_model(self):
        self._build_user_social_layer()
        self._build_user_social_project_head()
        self._build_user_esocial_project_head()
    
    def _build_user_social_layer(self):
        self.social_encoder = RGCNConv(self.num_users, self.user_embed_dim, self.n_relation, num_bases=self.num_bases)
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
        user_social_rep = self._get_user_social_user_profile(context_users)  # (bs, ns_user * user_dim)
        # context_users_attended =torch.stack([torch.cat((pair[0], pair[1]), dim=0) for pair in user_social_rep]).to(self.device)
        # #project_social_user_profile = self._project_user_esocial_user_project_profile(context_users_attended)  # (bs, user_dim)

        # context_users_initiator = torch.stack([pair[0] for pair in user_social_rep]).to(self.device)
        # context_users_respondent = torch.stack([pair[1] for pair in user_social_rep]).to(self.device)

        context_users_initiator = self._project_user_social_user_project_profile(user_social_rep)
        #context_users_respondent = self._project_user_social_user_project_profile(context_users_respondent)
        return context_users_initiator,context_users_initiator, context_users_initiator   # (bs, ns_user * user_dim)
    
    def get_user_social_user_profile(self, context_users):
        neighbors_init = self.edge_idx[1][self.edge_idx[0] == context_users[0]]
        neighbors_resp = self.edge_idx[1][self.edge_idx[0] == context_users[1]]
        unique_neighbors = list(set(neighbors_init) | set(neighbors_resp))
        users = list(set(context_users) | set(unique_neighbors))
        user_social_user_profile = self._get_user_social_user_profile(users)  # (bs, dim)

        return user_social_user_profile

    def _get_user_social_user_rep(self, context_users):
        social_embeddings = self.social_encoder(None,  self.edge_idx, self.edge_type)
        user_social_rep = self._encode_user(context_users, social_embeddings)  # (bs, ns_user * user_dim)
        return user_social_rep

    def union_along_second_dim(self, tensor1, tensor2):
        result = []
        for row1, row2 in zip(tensor1, tensor2):
            union_row = torch.unique(torch.cat((row1, row2)))
            result.append(union_row)
        # Pad sequences to the same length and stack them into a tensor with padding value -1
        padded_result = torch.nn.utils.rnn.pad_sequence(result, batch_first=True, padding_value=-1)
        return padded_result
    
    
    def _get_user_social_user_profile(self, context_users):
        nighbors = []
        for users in context_users:
            users = users.to(self.device)
            # Create masks for the initiator and respondent
            mask_init = self.edge_idx[0] == users[0]
            mask_resp = self.edge_idx[0] == users[1]
            mask_init = mask_init.to(self.device)
            mask_resp = mask_resp.to(self.device)

            # Apply the masks to get the neighbors
            neighbors_init = self.edge_idx[1][mask_init]
            neighbors_resp = self.edge_idx[1][mask_resp]
            neighbors_init = neighbors_init.to(self.device)
            neighbors_resp=neighbors_resp.to(self.device)
            if self.training:
                union_row = torch.unique(torch.cat((neighbors_init, neighbors_resp)))
                union_row = union_row.to(self.device)
                # Randomly select five items
                if union_row.size(0) >= 5:
                    perm = torch.randperm(union_row.size(0))
                    union_row = union_row[perm[:5]]
                else:
                    # Pad with -1 if less than five items
                    union_row = F.pad(union_row, (0, 5 - union_row.size(0)), value=-1)
            else:
                union_row = torch.unique(torch.cat((neighbors_init, neighbors_resp, users)))
                union_row = union_row.to(self.device)        
            # Append neighbors to the lists
            nighbors.append(union_row)

        # Pad the neighbors tensors and convert to a tensor
        nighbors_padded = torch.nn.utils.rnn.pad_sequence(nighbors, batch_first=True, padding_value=-1)    
        social_embeddings = self.social_encoder(None, self.edge_idx, self.edge_type)
        if self.training :
            user_social_user_profile = self._encode_user_profile(nighbors_padded, social_embeddings)  # (bs, dim)
            return user_social_user_profile
        user_social_user_profile = self._encode_user_profile(context_users, social_embeddings)  # (bs, dim)
        return user_social_user_profile
        
    
        
        


    def _encode_user_profile(self, context_users, social_embeddings):
        user_social_repr_list = []
        for user_list in context_users:
            mask = user_list != -1
            user_list = user_list[mask]
            if  len(user_list) == 0:
                user_social_repr_list.append(torch.zeros(self.user_embed_dim, device=self.device))
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
        user_social_user_profile = F.relu(user_social_user_profile) # (bs, dim)
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
