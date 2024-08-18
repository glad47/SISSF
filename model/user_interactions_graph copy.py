'''
Author: “glad47” ggffhh3344@gmail.com
Date: 2024-05-05 09:29:08
LastEditors: “glad47” ggffhh3344@gmail.com
LastEditTime: 2024-05-26 10:58:33
Description: user-item graph 
'''
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pickle
from model.UV_Encoders import UV_Encoder
from model.UV_Aggregators import UV_Aggregator
from model.attention import SelfAttentionBatch, SelfAttentionSeq
class UserItemGraph(nn.Module):

    def __init__(self, dir_data, embed_dim, user_embed_dim, weights_loc, items, device):
        super(UserItemGraph, self).__init__()

        self.dir_data = dir_data
        self.embed_dim = embed_dim
        self.user_embed_dim = user_embed_dim
        self.items = items
        
        self.device = device
        self.items = torch.tensor(self.items, dtype=torch.long).to(self.device)
        self.weights_loc= weights_loc
        self.history_u_lists, self.history_ur_lists, self.history_v_lists, self.history_vr_lists,self.train_u,self.train_v, self.train_r, self.test_u, self.test_v, self.test_r, self.social_adj_lists, self.ratings_list = self.initDataset()
        self.num_users = self.history_u_lists.__len__()
        self.num_items = self.history_v_lists.__len__()
        self.num_ratings = self.ratings_list.__len__()
        self.u2e, self.v2e, self.r2e = self.initEmbeddings()
        self._build_model()


    def initDataset(self):
        path_data = self.dir_data + "/redial_interactions.pickle"
        data_file = open(path_data, 'rb')
        return pickle.load(data_file)
    
    def initEmbeddings(self):
        u2e = nn.Embedding(self.num_users, self.embed_dim).to(self.device)
        v2e = nn.Embedding(self.num_items, self.embed_dim).to(self.device)
        r2e = nn.Embedding(self.num_ratings, self.embed_dim).to(self.device)    

        nn.init.constant_(u2e.weight, 0)
        nn.init.constant_(v2e.weight, 0)
        nn.init.constant_(r2e.weight, 0)
        return u2e, v2e, r2e
    
    def _build_model(self):
        self._build_user_interactions_layer()
        self._build_user_interaction_project_head()
        self._build_user_einteraction_project_head()
    
    
    def _build_user_interactions_layer(self):
        # item feature: user * rating
        agg_v_history = UV_Aggregator(self.v2e, self.r2e, self.u2e, self.embed_dim, cuda=self.device, uv=False)
        self.enc_v_history = UV_Encoder(self.v2e, self.embed_dim, self.history_v_lists, self.history_vr_lists, agg_v_history, cuda=self.device, uv=False)
        if self.weights_loc :
            checkpoint = torch.load(self.weights_loc, map_location='cpu')
            self.enc_v_history.load_state_dict(checkpoint)
        self.item_attn = SelfAttentionBatch(self.embed_dim, self.embed_dim)
    

    # nodes are the items index
    def _build_user_interaction_project_head(self):
        self.user_interaction_project_head_fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.batch_norm1 = nn.BatchNorm1d(self.embed_dim)
        self.user_interaction_project_head_fc2 = nn.Linear(self.embed_dim, self.user_embed_dim)

    def _build_user_einteraction_project_head(self):
        self.user_einteraction_project_head_fc1 = nn.Linear(self.user_embed_dim * 2, self.embed_dim)
        self.batch_norm2 = nn.BatchNorm1d(self.embed_dim)
        self.user_einteraction_project_head_fc2 = nn.Linear(self.embed_dim, self.user_embed_dim)    
    
    
    
   
    
    
    def get_user_interaction_project_user_item_profile(self, batch):
        context_items_initiator = [ torch.tensor([entity_id for entity_id in sample if entity_id != -1], dtype=torch.long ).to(self.device)   for sample in batch['context_items_initiator_ids'] ]
        context_items_respondent = [ torch.tensor([entity_id for entity_id in sample if entity_id != -1], dtype=torch.long ).to(self.device)   for sample in batch['context_items_respondent_ids'] ]

        initiator_interaction_user_item_profile = self.get_user_interaction_user_item_profile(context_items_initiator)  # (bs, dim)
        project_initiator_interaction_user_item_profile = self._project_user_interaction_item_user_project_profile(initiator_interaction_user_item_profile)  # (bs, user_dim)
        respondent_interaction_user_item_profile = self.get_user_interaction_user_item_profile(context_items_respondent)  # (bs, dim)
        project_respondent_interaction_user_item_profile = self._project_user_interaction_item_user_project_profile(respondent_interaction_user_item_profile)  # (bs, user_dim)
        user_item_profile = torch.cat([project_initiator_interaction_user_item_profile , project_respondent_interaction_user_item_profile], dim = 1) 
        project_user_item_profile = self._project_user_einteraction_item_user_project_profile(user_item_profile)  # (bs, user_dim)

        
        return project_user_item_profile, project_initiator_interaction_user_item_profile, project_respondent_interaction_user_item_profile  #  (bs, user_dim), (bs, user_dim)
    

    def get_user_interaction_item_rep(self, batch):
        
        context_items_initiator = [ torch.tensor([entity_id for entity_id in sample if entity_id != -1], dtype=torch.long ).to(self.device)  for sample in batch['context_items_initiator_ids'] ]
        context_items_respondent = [ torch.tensor([entity_id for entity_id in sample if entity_id != -1], dtype=torch.long ).to(self.device)   for sample in batch['context_items_respondent_ids'] ]
        

        initiator_interaction_item_rep= self._get_user_interaction_item_rep(context_items_initiator)  # (bs, ns_item * user_dim)
     
        respondent_interaction_item_rep= self._get_user_interaction_item_rep(context_items_respondent)  # (bs, ns_item * user_dim)
        users_interaction_item_rep = []
        for index, itemList in enumerate(initiator_interaction_item_rep):
            users_interaction_item_rep.append(initiator_interaction_item_rep[index] + respondent_interaction_item_rep[index]) 
      

        return users_interaction_item_rep, initiator_interaction_item_rep,respondent_interaction_item_rep  # (bs, ns_item * user_dim * 2), (bs, ns_item * user_dim), (bs, ns_item * user_dim)
    

    def get_user_interaction_user_item_profile(self, context_items):
        user_interaction_user_item_profile = self._get_user_interaction_user_item_profile(context_items)  # (bs, dim)

        return user_interaction_user_item_profile

    def _get_user_interaction_item_rep(self, context_items):
    
        user_interaction_item_rep = self._encode_item(context_items)  # (bs, ns_item * user_dim)
        return user_interaction_item_rep
    

    def _get_user_interaction_user_item_profile(self, context_items):
    
        user_interaction_item_user_profile = self._encode_item_user_profile(context_items)  # (bs, dim)
        return user_interaction_item_user_profile

    def _encode_item_user_profile(self, item_lists):
        user_item_repr_list = []
        for item_list in item_lists:
            if  len(item_list) == 0:
                user_item_repr_list.append(torch.zeros(self.user_embed_dim, device=self.device))
                continue
            item_repr = self.enc_v_history(item_list)
            _ , item_repr = self.item_attn(item_repr)
            user_item_repr_list.append(item_repr)
        return torch.stack(user_item_repr_list, dim=0)  # (bs, dim)
    

    def _encode_item(self, item_lists):
        user_item_repr_list = []
        for item_list in item_lists:
            if  len(item_list) == 0:
                user_item_repr_list.append([torch.zeros(self.user_embed_dim, device=self.device)])
                continue
            item_repr = self.enc_v_history(item_list)
            item_repr, _ = self.item_attn(item_repr)
            project_item_rep = []
            for item_dim in item_repr:
                project_item_rep.append(self._project_user_interaction_item_user_project_profile(item_dim))
            user_item_repr_list.append(project_item_rep)
        return user_item_repr_list  # (bs, ns_item * user_dim)

    def _project_user_interaction_item_user_project_profile(self, user_interaction_item_user_profile):
        user_interaction_item_user_profile = self.user_interaction_project_head_fc1(user_interaction_item_user_profile) # (bs, dim)
        user_interaction_item_user_profile = self.batch_norm1(user_interaction_item_user_profile)
        user_interaction_item_user_profile = F.relu(user_interaction_item_user_profile) # (bs, dim)
        user_interaction_item_user_profile = self.user_interaction_project_head_fc2(user_interaction_item_user_profile) # (bs, user_dim)

        return user_interaction_item_user_profile # (bs, user_dim)
    
    def _project_user_einteraction_item_user_project_profile(self, user_interaction_item_user_profile):
        user_interaction_item_user_profile = self.user_einteraction_project_head_fc1(user_interaction_item_user_profile) # (bs, dim)
        user_interaction_item_user_profile = self.batch_norm2(user_interaction_item_user_profile)
        user_interaction_item_user_profile = F.relu(user_interaction_item_user_profile) # (bs, dim)
        user_interaction_item_user_profile = self.user_einteraction_project_head_fc2(user_interaction_item_user_profile) # (bs, user_dim)

        return user_interaction_item_user_profile # (bs, user_dim)
    



    def get_all_embeddings(self):
        return self.enc_v_history(self.items)