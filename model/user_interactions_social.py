

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import pickle
from model.UV_Encoders import UV_Encoder
from model.UV_Aggregators import UV_Aggregator
from model.attention import SelfAttentionBatch, SelfAttentionSeq
class SocialItemGraph(nn.Module):

    def __init__(self, dir_data, embed_dim, user_embed_dim, items, num_bases, device):
        super(SocialItemGraph, self).__init__()

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
        self._build_user_einteraction_project_head()
    
    
    def _build_user_interactions_layer(self):
        # item feature: user * rating
        self.interaction_encoder = RGCNConv(self.num_items + self.num_users, self.user_embed_dim, self.n_relation, num_bases=self.num_bases)
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
    

    def union_along_second_dim(self, tensor1, tensor2):
        result = []
        for row1, row2 in zip(tensor1, tensor2):
            union_row = torch.unique(torch.cat((row1, row2)))
            result.append(union_row)
        # Pad sequences to the same length and stack them into a tensor with padding value -1
        padded_result = torch.nn.utils.rnn.pad_sequence(result, batch_first=True, padding_value=-1)
        return padded_result

    def get_user_interaction_item_rep(self, batch):
        
        context_items_initiator = batch['context_items_initiator_ids']
        context_items_respondent = batch['context_items_respondent_ids'] 
        

       
        # If you need the result as a list
        all_context_items_list = self.union_along_second_dim(context_items_initiator, context_items_respondent)

        initiator_interaction_item_rep, _= self._get_user_interaction_item_rep(all_context_items_list)  # (bs, ns_item * user_dim)
     
        respondent_interaction_item_rep, _= self._get_user_interaction_item_rep(context_items_respondent)  # (bs, ns_item * user_dim)
        
        initiator_interaction_item_rep = self._project_user_interaction_item_user_project_profile(initiator_interaction_item_rep)
        respondent_interaction_item_rep = self._project_user_interaction_item_user_project_profile(respondent_interaction_item_rep)

        return initiator_interaction_item_rep , initiator_interaction_item_rep,respondent_interaction_item_rep  # (bs, ns_item * user_dim * 2), (bs, ns_item * user_dim), (bs, ns_item * user_dim)
    

    def get_user_interaction_item_rep_recommendation(self, batch):
        
        context_items = batch['context_items_ids']
        # context_items_respondent = batch['context_items_respondent_ids'] 
        

       
        # If you need the result as a list
        # all_context_items_list = self.union_along_second_dim(context_items_initiator, context_items_respondent)

        initiator_interaction_item_rep, social_reps= self._get_user_interaction_item_rep(context_items)  # (bs, ns_item * user_dim)
     
        # respondent_interaction_item_rep= self._get_user_interaction_item_rep(context_items_respondent)  # (bs, ns_item * user_dim)
        
        initiator_interaction_item_rep = self._project_user_interaction_item_user_project_profile(initiator_interaction_item_rep)
        # respondent_interaction_item_rep = self._project_user_interaction_item_user_project_profile(respondent_interaction_item_rep)

        return initiator_interaction_item_rep , social_reps,initiator_interaction_item_rep  # (bs, ns_item * user_dim * 2), (bs, ns_item * user_dim), (bs, ns_item * user_dim)
    

    def get_user_interaction_user_item_profile(self, context_items):
        user_interaction_user_item_profile, _ = self._get_user_interaction_user_item_profile(context_items)  # (bs, dim)

        return user_interaction_user_item_profile
    

    

    def _get_user_interaction_item_rep(self, context_items):
        interaction_embeddings = self.interaction_encoder(None, self.edge_idx, self.edge_type)
        user_interaction_item_rep, social_reps = self._encode_item_user_profile(context_items, interaction_embeddings)  # (bs, ns_item * user_dim)
        return user_interaction_item_rep, social_reps
    

    def _get_user_interaction_user_item_profile(self, context_items):
        interaction_embeddings = self.interaction_encoder(None, self.edge_idx, self.edge_type)
        user_interaction_item_user_profile, social_reps = self._encode_item_user_profile(context_items, interaction_embeddings)  # (bs, dim)
        return user_interaction_item_user_profile, social_reps

    def _encode_item_user_profile(self, item_lists, interaction_embeddings):
        user_item_repr_list = []
        social_items = []
        for item_list in item_lists:
            mask = item_list != -1
            item_list = item_list[mask]
            if  len(item_list) == 0:
                user_item_repr_list.append(torch.zeros(self.user_embed_dim, device=self.device))
                social_items.append(torch.zeros(self.user_embed_dim, device=self.device).unsqueeze(0))
                continue
            item_repr = []
            for item in item_list:
                connected_users = self.edge_idx[1][self.edge_idx[0] == item]
                if connected_users.numel() > 0:
                    embedding = interaction_embeddings[connected_users]
                    mean_pooled_embedding = torch.mean(embedding, dim=0)
                    item_repr.append(mean_pooled_embedding)
            if(len(item_repr) == 0):
                user_item_repr_list.append(torch.zeros(self.user_embed_dim, device=self.device))
                social_items.append(torch.zeros(self.user_embed_dim, device=self.device).unsqueeze(0))
                continue
            item_repr = torch.nn.utils.rnn.pad_sequence(item_repr, batch_first=True)        
            items, item_repr = self.item_attn(item_repr)
            user_item_repr_list.append(item_repr)
            social_items.append(items)
        return torch.stack(user_item_repr_list, dim=0),  torch.nn.utils.rnn.pad_sequence(social_items, batch_first=True) # (bs, dim), (bs, num_user ,dim)
    

    def _encode_item(self, item_lists, interaction_embeddings):
        user_item_repr_list = []
        for item_list in item_lists:
            mask = item_list != -1
            item_list = item_list[mask]
            if  len(item_list) == 0:
                user_item_repr_list.append([torch.zeros(self.user_embed_dim, device=self.device)])
                continue
            item_repr = interaction_embeddings[item_list]
            item_repr, _ = self.item_attn(item_repr)
            project_item_rep = []
            for item_dim in item_repr:
                project_item_rep.append(item_dim)
            user_item_repr_list.append(project_item_rep)
        return user_item_repr_list  # (bs, ns_item * user_dim)

    def _project_user_interaction_item_user_project_profile(self, user_interaction_item_user_profile):
        user_interaction_item_user_profile = self.user_interaction_project_head_fc1(user_interaction_item_user_profile) # (bs, dim)
        # user_interaction_item_user_profile = self.batch_norm1(user_interaction_item_user_profile)
        user_interaction_item_user_profile = F.relu(user_interaction_item_user_profile) # (bs, dim)
        user_interaction_item_user_profile = self.user_interaction_project_head_fc2(user_interaction_item_user_profile) # (bs, user_dim)
        user_interaction_item_user_profile = F.relu(user_interaction_item_user_profile) # (bs, dim)
        return user_interaction_item_user_profile # (bs, user_dim)
    
    def _project_user_einteraction_item_user_project_profile(self, user_interaction_item_user_profile):
        user_interaction_item_user_profile = self.user_einteraction_project_head_fc1(user_interaction_item_user_profile) # (bs, dim)
        user_interaction_item_user_profile = self.batch_norm2(user_interaction_item_user_profile)
        user_interaction_item_user_profile = F.relu(user_interaction_item_user_profile) # (bs, dim)
        user_interaction_item_user_profile = self.user_einteraction_project_head_fc2(user_interaction_item_user_profile) # (bs, user_dim)

        return user_interaction_item_user_profile # (bs, user_dim)
    



    def get_all_embeddings(self):
        interaction_embedding = self.interaction_encoder(None, self.edge_idx, self.edge_type).to(self.device)
        return interaction_embedding