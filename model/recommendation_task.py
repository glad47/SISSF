

import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from util.logconf import logging
from model.rec_trans import RecommenderTransformer
from model.sing_rec_trans import SingleRecommenderTransformer

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class RecommenderModule(nn.Module):
    def __init__(self, user_embed_dim, interaction_embed_dim, social_embed_dim, entity_embed_dim, 
                 n_items, n_entity, n_user, n_heads, ffn_dim, dropout, attention_dropout,n_layers, device ):
        super().__init__()
        self.user_embed_dim = user_embed_dim
        self.interaction_embed_dim = interaction_embed_dim
        self.social_embed_dim = social_embed_dim
        self.entity_embed_dim = entity_embed_dim
        self.n_items = n_items
        self.n_entity = n_entity
        self.n_user = n_user
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.n_layers = n_layers
        self.device = device

        self._build_model()
        
        
    def _build_model(self):
        #transormation layer to user embedding 
        self.transform_items_embeddings = nn.Linear(self.interaction_embed_dim, self.user_embed_dim)
        # self.transform_entity_embeddings = nn.Linear(self.entity_embed_dim, self.user_embed_dim)
        self.transform_social_embeddings = nn.Linear(self.social_embed_dim, self.user_embed_dim)


        #create multiplication layer
        self.rec_item_bias = nn.Parameter(torch.zeros(self.n_items))
        self.rec_entity_bias = nn.Parameter(torch.zeros(self.n_entity)) 
        self.rec_user_bias =nn.Parameter(torch.zeros(self.n_user))


        # transormation layer to items dim
        self.item_entity_socres = nn.Linear(self.n_entity, self.n_items, bias=True)
        self.item_user_socres = nn.Linear(self.n_user, self.n_items, bias=True)


        # self.bn_item = nn.BatchNorm1d(self.user_embed_dim)
        # self.bn_entity = nn.BatchNorm1d(self.user_embed_dim)
        # self.bn_user_bias = nn.BatchNorm1d(self.user_embed_dim)


        self.trans = SingleRecommenderTransformer(embed_dim= self.user_embed_dim,n_items= self.n_items,ffn_dim=self.ffn_dim ,dropout=self.dropout ,n_heads=self.n_heads ,attention_dropout=self.attention_dropout ,n_layers=self.n_layers ,device= self.device)

        self.trans = self.trans.to(self.device)



        self.output_projection_social = nn.Linear(self.n_user, self.user_embed_dim)
        self.output_projection_inter = nn.Linear(self.n_items, self.user_embed_dim)
        
      
        # loss function 
        self.rec_loss = nn.CrossEntropyLoss()

    def metrics_cal_rec(self,scores,labels):
        metrics_rec= {}
        metrics_rec["recall@1"] = 0
        metrics_rec["recall@10"] = 0
        metrics_rec["recall@50"] = 0
        metrics_rec["count"] = 0
        batch_size = len(labels.view(-1).tolist())
        outputs = scores.cpu()
        #outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        _, pred_idx = torch.topk(outputs, k=50, dim=1)
        for b in range(batch_size):
            if labels[b].item()==0:
                continue
            #target_idx = self.movie_ids.index(labels[b].item())
            target_idx = labels[b].item()
            metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            metrics_rec["count"] += 1
        output_dict_rec={key: metrics_rec[key] / metrics_rec['count'] for key in metrics_rec}
        return output_dict_rec["recall@1"], output_dict_rec["recall@10"], output_dict_rec["recall@50"]
            

    
    def forward(self, user_model, batch):
        y = batch['movie_to_rec'].to(self.device)
        # workerIds = batch['worker_ids'].to(self.device)
        # initiatorIds = batch['initiator_ids'].to(self.device) 
        # respondentIds = batch['respondent_ids'].to(self.device)
       
        
        social_embeddings, social_embeddings_initiator, social_embeddings_respondent  = user_model.social_graph.get_user_interaction_item_rep_recommendation(batch)
        #entity_embeddings, kg_initiator_profile , kg_respondent_profile = user_model.entity_graph.get_kg_project_user_profile(batch)  
        interaction_embeddings, interaction_embeddings_initiator, interaction_embeddings_respondent= user_model.interaction_graph.get_user_interaction_item_rep_recommendation(batch)
        
        
        #entity_embs = user_model.entity_graph.get_all_embeddings()
        # user_embs = user_model.social_graph.get_all_embeddings()
        # item_embs = user_model.interaction_graph.get_all_embeddings()

        
        #entity_embs = self.transform_entity_embeddings(entity_embs).to(self.device)
        # user_embs = self.transform_social_embeddings(user_embs).to(self.device)
        # item_embs = self.transform_items_embeddings(item_embs).to(self.device) 


        
        

        #entity_profile = []
        # social_profile = []
        # interaction_profile = []

        # for index, workId in enumerate(workerIds):
        #     if workId ==  initiatorIds[index]:
        #         #entity_profile.append(kg_initiator_profile[index])
        #         social_profile.append(social_embeddings_initiator[index])
        #         interaction_profile.append(interaction_embeddings_initiator[index])
        #     else:
        #         #entity_profile.append(kg_respondent_profile[index])
        #         social_profile.append(social_embeddings_respondent[index])
        #         interaction_profile.append(interaction_embeddings_respondent[index])



        # #entity_profile = torch.stack(entity_profile).to(self.device)  
        # social_profile = torch.stack(social_profile).to(self.device)  
        # interaction_profile = torch.stack(interaction_profile).to(self.device) 
        

        # social_profile = F.dropout(social_profile, p=0.9, training=self.training)
        # interaction_profile = F.dropout(interaction_profile, p=0.9, training=self.training)
        
        
        #interaction_profile = self.bn_item(interaction_profile)
        #entity_profile = self.bn_entity(entity_profile)
        #social_profile = self.bn_user_bias(social_profile)
        
        #entities_score = F.linear(entity_profile, entity_embs, self.rec_entity_bias ).to(self.device)
        # users_score = F.linear(social_profile, user_embs).to(self.device) 
        # items_score = F.linear(interaction_profile, item_embs).to(self.device)
        # users_score = torch.nn.ReLU()(users_score).to(self.device)
        # items_score = torch.nn.ReLU()(items_score).to(self.device)

        # users_score = F.dropout(users_score, p=0.5, training=self.training)
        # items_score = F.dropout(items_score, p=0.5, training=self.training)



        #entities_score = self.item_entity_socres(entities_score).to(self.device)
        # users_score = self.item_user_socres(users_score ).to(self.device)
        
        # Assuming items_score is a tensor from which you want to use the batch size
        # users_score = self.output_projection_social(users_score).to(self.device)
        # items_score = self.output_projection_inter(items_score).to(self.device)

        #combined_embeds_inti = torch.cat((interaction_embeddings, social_embeddings), dim=1)
        # combined_embeds_inti = torch.nn.ReLU()(combined_embeds_inti).to(self.device)
        # # Pass through the encoder
        # encoder_output_init = self.encoder(combined_embeds_inti, None)

        x = self.trans(interaction_embeddings, None)



        

        rec_scores = x

        

        # rec_scores = items_score * users_score 
       
        # l2_lambda = 50  # Lambda for L2 regularization

        # # Compute L2 norm of the user and item biases
        # l2_norm = sum(p.pow(2.0).sum() for p in [self.rec_user_bias, self.rec_item_bias])

       
        rec_scores = rec_scores.to(self.device)
        
        rec_loss = self.rec_loss(rec_scores, y)



        # _, rec_ranks = torch.topk(rec_scores, 50, dim=-1)
        # rec_ranks = rec_ranks.to(self.device)
        r_at_1, r_at_10, r_at_50 = self.metrics_cal_rec(rec_scores, y)
        # r_at_1 = self.recall_at_k(y, rec_ranks, 1)
        # r_at_10 = self.recall_at_k(y, rec_ranks, 10)
        # r_at_50 = self.recall_at_k(y, rec_ranks, 50)


        log.info("****************************")
        log.info(f"{r_at_1} R@1")
        log.info(f"{r_at_10} R@10")
        log.info(f"{r_at_50} R@50")
        log.info(f"{rec_loss} rec_loss")
       
        

        return rec_loss, r_at_1, r_at_10, r_at_50
