'''
Author: “glad47” ggffhh3344@gmail.com
Date: 2024-05-05 09:29:08
LastEditors: “glad47” ggffhh3344@gmail.com
LastEditTime: 2024-06-06 16:38:53
Description: user-item graph 
'''
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pickle
from model.UV_Encoders import UV_Encoder
from model.UV_Aggregators import UV_Aggregator
from model.Social_Encoders import Social_Encoder
from model.Social_Aggregators import Social_Aggregator
from model.attention import SelfAttentionBatch, SelfAttentionSeq


class UserSocialGraph(nn.Module):
    def __init__(self, dir_data, embed_dim, user_embed_dim, weights_loc, users, device):
        super(UserSocialGraph, self).__init__()

        self.dir_data = dir_data
        self.embed_dim = embed_dim
        self.user_embed_dim = user_embed_dim
        self.users = users
        self.device = device
        self.users = torch.tensor(self.users, dtype=torch.long).to(self.device)
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
        return u2e, v2e, r2e
    def _build_model(self):
        self._build_user_social_layer()
        self._build_user_social_project_head()
        self._build_user_esocial_project_head()
    
    def _build_user_social_layer(self):
        # item feature: user * rating
       # user feature
        # features: item * rating
        agg_u_history = UV_Aggregator(self.v2e, self.r2e, self.u2e, self.embed_dim, self.device, True)
        enc_u_history = UV_Encoder(self.u2e, self.embed_dim, self.history_u_lists, self.history_ur_lists, agg_u_history, cuda=self.device, uv=True)
        # neighobrs
        agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), self.u2e, self.embed_dim, cuda=self.device)
        self.enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), self.embed_dim, self.social_adj_lists, agg_u_social,
                            base_model=enc_u_history, cuda=self.device)
        if self.weights_loc :
            checkpoint = torch.load(self.weights_loc, map_location='cpu')
            self.enc_u.load_state_dict(checkpoint)   
        self.user_attn = SelfAttentionBatch(self.embed_dim, self.embed_dim)
     
    
    def _build_user_social_project_head(self):
        self.user_social_project_head_fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.batch_norm1 = nn.BatchNorm1d(self.embed_dim)
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
        return user_social_rep  # (bs, ns_user * user_dim)
    
    def get_user_social_user_profile(self, context_users):
    
        user_social_user_profile = self._get_user_social_user_profile(context_users)  # (bs, dim)

        return user_social_user_profile

    def _get_user_social_user_rep(self, context_users):
    
        user_social_rep = self._encode_user(context_users)  # (bs, ns_user * user_dim)
        return user_social_rep
    
    def _get_user_social_user_profile(self, context_users):
    
        user_social_user_profile = self._encode_user_profile(context_users)  # (bs, dim)
        return user_social_user_profile


    def _encode_user_profile(self, context_users):
        user_social_repr_list = []
        for user_list in context_users:
            if  user_list.numel() == 0:
                user_social_repr_list.append([torch.zeros( self.user_embed_dim, device=self.device), torch.zeros( self.user_embed_dim, device=self.device)])
                continue
            user_repr = self.enc_u(user_list)
            _ , user_repr = self.user_attn(user_repr)
            user_social_repr_list.append(user_repr)
        return torch.stack(user_social_repr_list, dim=0)  # (bs, dim)
    

    def _encode_user(self, context_users):
        user_social_repr_list = []
        for user_list in context_users:
            if  user_list.numel() == 0:
                user_social_repr_list.append([torch.zeros(self.user_embed_dim, device=self.device), torch.zeros( self.user_embed_dim, device=self.device)])
                continue
            user_repr = self.enc_u(user_list)
            user_repr, _ = self.user_attn(user_repr)
            user_social_repr_list.append(self._project_user_social_user_project_profile(user_repr))
        return user_social_repr_list  # (bs, ns_user * user_dim)

    def _project_user_social_user_project_profile(self, user_social_user_profile):
        user_social_user_profile = self.user_social_project_head_fc1(user_social_user_profile) # (bs, dim)
        if user_social_user_profile.size(0) > 1:
            user_social_user_profile = self.batch_norm1(user_social_user_profile)
        
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
        return self.enc_u(self.users)
