## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import pickle
from model.attention import SelfAttentionBatch, SelfAttentionSeq
class SocialGraph(nn.Module):

    def __init__(self, dir_data, embed_dim, user_embed_dim, items, num_bases, device):
        super(SocialGraph, self).__init__()

        self.dir_data = dir_data
        self.embed_dim = embed_dim
        self.user_embed_dim = user_embed_dim
        self.items = items
        self.num_bases = num_bases
        self.device = device
        self.items = torch.tensor(self.items, dtype=torch.long).to(self.device)
        self.history_u_lists, self.history_ur_lists, self.history_v_lists, self.history_vr_lists,self.train_u,self.train_v, self.train_r, self.test_u, self.test_v, self.test_r, self.social_adj_lists, self.ratings_list = self.initDataset()
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
        for item, relations in self.history_v_lists.items():
            item_ratings = self.history_vr_lists[item]
            for index, user in enumerate(relations):
                edge_list.append((item , user, item_ratings[index]))
                # edge_list.append((user , item, item_ratings[index]))

       
        edges, nodes = set(), set()
        
        for h, t, r in edge_list:
            edges.add((h, t, r))
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
        self._build_user_interactions_layer()
        self._build_user_interaction_project_head()
    
    
    def _build_user_interactions_layer(self):
        # item feature: user * rating
        self.social_encoder = RGCNConv(self.num_items + self.num_users, self.user_embed_dim, self.n_relation, num_bases=self.num_bases)
        self.item_attn = SelfAttentionBatch(self.embed_dim, self.embed_dim)
    

    # nodes are the items index
    def _build_user_interaction_project_head(self):
        self.user_interaction_project_head_fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.batch_norm1 = nn.BatchNorm1d(self.embed_dim)
        self.user_interaction_project_head_fc2 = nn.Linear(self.embed_dim, self.user_embed_dim)
    
    
    

    def union_along_second_dim(self, tensor1, tensor2):
        result = []
        for row1, row2 in zip(tensor1, tensor2):
            union_row = torch.unique(torch.cat((row1, row2)))
            result.append(union_row)
        # Pad sequences to the same length and stack them into a tensor with padding value -1
        padded_result = torch.nn.utils.rnn.pad_sequence(result, batch_first=True, padding_value=-1) # (bs, max_total_n_item)
        return padded_result # (bs, max_total_n_item)
    



    def get_user_social_rep(self, batch):
        context_items_initiator = batch['context_items_initiator_ids']  # (bs, max_n_item)
        context_items_respondent = batch['context_items_respondent_ids'] # (bs, max_n_item)
        all_context_items_list = self.union_along_second_dim(context_items_initiator, context_items_respondent) # (bs, max_total_n_item)
        social_item_rep, _= self._get_user_social_item_rep(all_context_items_list)  # (bs, dim), 
        project_social_rep = self._user_social_item_project_profile(social_item_rep) # (bs, dim)
        return project_social_rep  # (bs, dim)
    

    

    def get_social_rep_recommendation(self, batch):
        context_items = batch['context_items_ids'] # (bs, items)
        social_item_rep, social_reps= self._get_user_social_item_rep(context_items)  # (bs, dim), (bs, ns_social_profile ,dim)
        project_social_rep = self._user_social_item_project_profile(social_item_rep) # (bs, dim)
        return project_social_rep , social_reps  # (bs, dim),  (bs, ns_social_profile ,dim)
    




    def _get_user_social_item_rep(self, context_items):
        # (bs, n_item)
        social_embeddings = self.social_encoder(None, self.edge_idx, self.edge_type) # (bs, all_ns_item + all_ns_user)
        user_interaction_item_rep, social_reps = self._encode_user_profile(context_items, social_embeddings)  # (bs, dim), (bs, ns_social_profile ,dim)
        return user_interaction_item_rep, social_reps # (bs, dim), (bs, ns_social_profile ,dim)
    



    def _encode_user_profile(self, item_lists, social_embeddings):
        # (bs, n_item), (bs, all_ns_item + all_ns_user)
        user_item_repr_list = []
        social_profile = []
        for item_list in item_lists:
            mask = item_list != -1
            item_list = item_list[mask]
            if  len(item_list) == 0:
                user_item_repr_list.append(torch.zeros(self.user_embed_dim, device=self.device))
                social_profile.append(torch.zeros(self.user_embed_dim, device=self.device).unsqueeze(0))
                continue
            social_repr = []
            for item in item_list:
                #get the embeddings of all user that interacted with the item 
                connected_users = self.edge_idx[1][self.edge_idx[0] == item]
                if connected_users.numel() > 0:
                    embedding = social_embeddings[connected_users]  # (ns_connect_user, dim)
                    # get the mean of all users embeddings 
                    mean_pooled_embedding = torch.mean(embedding, dim=0) # (dim)
                    social_repr.append(mean_pooled_embedding)
            if(len(social_repr) == 0):
                user_item_repr_list.append(torch.zeros(self.user_embed_dim, device=self.device))
                social_profile.append(torch.zeros(self.user_embed_dim, device=self.device).unsqueeze(0))
                continue
            social_repr = torch.nn.utils.rnn.pad_sequence(social_repr, batch_first=True)  # (ns_social_profile, dim)      
            items, social_repr = self.item_attn(social_repr) # (ns_social_profile, dim), (dim)   
            user_item_repr_list.append(social_repr)
            social_profile.append(items)
        return torch.stack(user_item_repr_list, dim=0),  torch.nn.utils.rnn.pad_sequence(social_profile, batch_first=True) # (bs, dim), (bs, ns_social_profile ,dim)
    


    def _user_social_item_project_profile(self, user_interaction_item_user_profile):
        user_interaction_item_user_profile = self.user_interaction_project_head_fc1(user_interaction_item_user_profile) # (bs, dim)
        user_interaction_item_user_profile = F.relu(user_interaction_item_user_profile) # (bs, dim)
        user_interaction_item_user_profile = self.user_interaction_project_head_fc2(user_interaction_item_user_profile) # (bs, user_dim)
        user_interaction_item_user_profile = F.relu(user_interaction_item_user_profile) # (bs, dim)
        return user_interaction_item_user_profile # (bs, user_dim)
    

    def getItemEmbeddings(self):
        social_embeddings = self.social_encoder(None, self.edge_idx, self.edge_type) # (bs, all_ns_item + all_ns_user)
        return social_embeddings[self.num_users:self.num_items + self.num_users]



