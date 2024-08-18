'''
Author: your name
Date: 2024-06-11 09:17:05
LastEditTime: 2024-06-11 09:17:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Step5\model\entitiy_relationships_graph copy.py
'''

from torch_geometric.nn import RGCNConv
from model.attention import SelfAttentionBatch, SelfAttentionSeq
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import json
import os
from collections import defaultdict



class EntityRelationshipGraph(nn.Module):
    def __init__(self, kg_emb_dim, user_emb_dim, n_entity, num_bases, data_dir, device, SELF_LOOP_ID=185):
        super(EntityRelationshipGraph, self).__init__()
        self.kg_emb_dim = kg_emb_dim 
        self.user_emb_dim = user_emb_dim
        self.n_entity = n_entity
        self.num_bases = num_bases
        self.data_dir = data_dir
        self.device = device
        self.SELF_LOOP_ID = SELF_LOOP_ID
        self._build_model()
        self = self.to(self.device)
        
        self._build_model()
        
    def _build_model(self):
        self.load_data()
        self._build_kg_layer()
        self._build_kg_cl_project_head()
        self._build_ekg_cl_project_head()

    def _build_kg_layer(self):
        # print([self.config.n_entity, self.config.kg_emb_dim, self.config.n_relation, self.config.num_bases])
        # ipdb.set_trace()
        self.kg_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        self.kg_user_rep_dense = nn.Linear(self.kg_emb_dim, self.user_emb_dim)

    def _build_kg_cl_project_head(self):
        self.kg_project_head_fc1 = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        self.kg_project_head_fc2 = nn.Linear(self.kg_emb_dim, self.user_emb_dim)

    def _build_ekg_cl_project_head(self):
        # entity-level kg
        self.ekg_project_head_fc1 = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        self.ekg_project_head_fc2 = nn.Linear(self.kg_emb_dim, self.user_emb_dim)

    def get_project_kg_rep(self, batch):
        context_entities = batch['context_entities']
        entity_ids_in_context = batch['entity_ids_in_context'] # [bs*n_eic]
        eic_conv_ids = batch['eic_conv_ids'] # [bs*n_eic]
        
        duplicate_removal_ekg_ids, duplicate_removal_ekg_conv_ids = self.get_duplicate_removal_ids(entity_ids_in_context, eic_conv_ids) # [<=bs*n_eic], [<=bs*n_eic]

        kg_user_rep, kg_embedding = self._get_kg_user_rep(context_entities)  # (bs, dim), (n_entity, dim)
        project_kg_user_rep = self._project_kg_user_rep(kg_user_rep)  # (bs, dim)

        #project_entity_kg_reps = self._get_project_entity_kg_reps(kg_embedding, duplicate_removal_ekg_ids) # (~bs*n_eic, user_dim) or (1, user_dim)

        return project_kg_user_rep # (bs, dim), (~bs*n_eic, user_dim) or (1, user_dim)
    
    def get_duplicate_removal_ids(self, entity_ids_in_context, eic_conv_ids):
        # [bs*n_eic], [bs*n_eic]
        dr_ekg_ids_set, duplicate_removal_ekg_ids, duplicate_removal_ekg_conv_ids = set(), [], []

        for ekg_id, ekg_conv_id in zip(entity_ids_in_context, eic_conv_ids):
            if ekg_id not in dr_ekg_ids_set:
                dr_ekg_ids_set.add(ekg_id)
                duplicate_removal_ekg_ids.append(ekg_id)
                duplicate_removal_ekg_conv_ids.append(ekg_conv_id)

        return duplicate_removal_ekg_ids, duplicate_removal_ekg_conv_ids # [<=bs*n_eic], [<=bs*n_eic]

    def _get_project_entity_kg_reps(self, kg_embedding, entity_ids):
        entity_kg_reps = self._get_entity_kg_reps(kg_embedding, entity_ids) # (~b*n_eic, kg_dim) or (1, kg_dim)
        project_entity_kg_reps = self._project_entity_kg_reps(entity_kg_reps)  # (~b*n_eic, user_dim) or (1, user_dim)

        return project_entity_kg_reps  # (~b*n_eic, user_dim) or (1, user_dim)

    def _get_entity_kg_reps(self, kg_embedding, entity_ids):
        # (n_entity, kg_dim),

        if len(entity_ids) == 0:
            entity_kg_reps = torch.zeros((1, self.kg_emb_dim))
        else:
            entity_kg_reps = kg_embedding[entity_ids] # (~bs*n_eic, kg_dim)

        return entity_kg_reps # (~bs*n_eic, kg_dim) or (1, kg_dim)
    
    def _project_entity_kg_reps(self, entity_kg_reps):
        entity_kg_reps = self.ekg_project_head_fc1(entity_kg_reps)
        entity_kg_reps = F.relu(entity_kg_reps)
        entity_kg_reps = self.ekg_project_head_fc2(entity_kg_reps)

        return entity_kg_reps

    def get_kg_rep(self, batch, mode):
        context_entities = batch['context_entities']

        kg_user_rep, kg_embedding = self._get_kg_user_rep(context_entities)  # (bs, dim), (n_entity, dim)

        return kg_user_rep, kg_embedding

    def _get_kg_user_rep(self, context_entities):
        # ipdb.set_trace()
        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)
        # print(self.config.edge_idx.shape)
        # ipdb.set_trace()
        user_rep = self._encode_user(context_entities, kg_embedding)  # (bs, dim)

        return user_rep, kg_embedding

    def _encode_user(self, entity_lists, kg_embedding):
        user_repr_list = []
        for entity_list in entity_lists:
            if not entity_list:
                user_repr_list.append(torch.zeros(self.user_emb_dim, device=self.device))
                continue
            user_repr = kg_embedding[entity_list]
            user_repr = self.kg_attn(user_repr)
            user_repr_list.append(user_repr)
        return torch.stack(user_repr_list, dim=0)  # (bs, dim)

    def _project_kg_user_rep(self, kg_user_rep):
        kg_user_rep = self.kg_project_head_fc1(kg_user_rep) # (bs, dim)
        kg_user_rep = F.relu(kg_user_rep) # (bs, dim)
        kg_user_rep = self.kg_project_head_fc2(kg_user_rep) # (bs, dim)

        return kg_user_rep # (bs, dim)
    
    
    def load_data(self):
        self.entity_kg = json.load(open(os.path.join(self.data_dir, 'dbpedia_subkg.json'), 'r', encoding='utf-8'))
        self.entity2id = json.load( open(os.path.join(self.data_dir, 'entity2id.json'), 'r', encoding='utf-8'))
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.entities = [idx for entity, idx in self.entity2id.items()]
        self.entities = torch.tensor(self.entities, dtype=torch.long).to(self.device)
        edge_list = []  # [(entity, entity, relation)]
        entity2neighbor = defaultdict(list)  # {entityId: List[entity]}
        for entity in range(self.n_entity):
            if str(entity) not in self.entity_kg:
                continue
            edge_list.append((entity, entity, self.SELF_LOOP_ID))  # add self loop
            for tail_and_relation in self.entity_kg[str(entity)]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != self.SELF_LOOP_ID:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

        relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 1000:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                edges.add((h, t, relation2id[r]))
                entities.add(self.id2entity[h])
                entities.add(self.id2entity[t])
            entity2neighbor[h].append(t) 
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
        
       
        self.n_relation = len(relation2id)
        self.entity2neighbor = dict(entity2neighbor)



    def get_all_embeddings(self):
        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type).to(self.device)
        return kg_embedding    
        




    