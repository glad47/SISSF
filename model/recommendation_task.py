

## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14

import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from util.logconf import logging
from model.rec_trans import RecommenderTransformer
from model.auto_encoder import AutoRec

log = logging.getLogger(__name__)
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
        self.trans = RecommenderTransformer(embed_dim= self.user_embed_dim,n_items= self.n_items,ffn_dim=self.ffn_dim ,dropout=self.dropout ,n_heads=self.n_heads ,attention_dropout=self.attention_dropout ,n_layers=self.n_layers ,device= self.device)
        self.trans = self.trans.to(self.device)
        # loss function 
        self.rec_loss = nn.CrossEntropyLoss()
       


    def metrics_cal_rec(self,scores,labels):
        #(bs,  n_item), bs
        metrics_rec= {}
        metrics_rec["recall@1"] = 0
        metrics_rec["recall@10"] = 0
        metrics_rec["recall@50"] = 0
        metrics_rec["count"] = 0
        batch_size = len(labels.view(-1).tolist())
        outputs = scores.cpu()
        _, pred_idx = torch.topk(outputs, k=50, dim=1)
        for b in range(batch_size):
            if labels[b].item()==0:
                continue
            target_idx = labels[b].item()
            metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            metrics_rec["count"] += 1
        output_dict_rec={key: metrics_rec[key] / metrics_rec['count'] for key in metrics_rec}
        return output_dict_rec["recall@1"], output_dict_rec["recall@10"], output_dict_rec["recall@50"]
            

    
    def forward(self, user_model, batch):
        y = batch['movie_to_rec'].to(self.device) # (bs)
        project_user_resp_rep ,project_user_init_rep, social_reps  = user_model.social_info.get_social_rep_recommendation(batch) # (bs, dim)
        conv_history_embeddings = user_model.conversation_encoder.get_project_context_rep(batch) # (bs, dim)
        
        x = self.trans(project_user_resp_rep,project_user_init_rep, conv_history_embeddings, None)        #(bs,  n_item)
        rec_scores = x                                 #(bs,  n_item)
        rec_scores = rec_scores.to(self.device)        #(bs,  n_item)
        rec_loss = self.rec_loss(rec_scores, y)        
        r_at_1, r_at_10, r_at_50 = self.metrics_cal_rec(rec_scores, y)

        # log.info("****************************")
        # log.info(f"{r_at_1} R@1")
        # log.info(f"{r_at_10} R@10")
        # log.info(f"{r_at_50} R@50")
        # log.info(f"{rec_loss} rec_loss")
       
        

        return rec_loss, r_at_1, r_at_10, r_at_50
