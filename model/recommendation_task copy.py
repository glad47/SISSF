'''
Author: your name
Date: 2024-06-11 09:16:30
LastEditTime: 2024-06-11 09:16:30
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Step5\model\recommendation_task copy.py
'''
'''
Author: “glad47” ggffhh3344@gmail.com
Date: 2024-05-15 16:53:01
LastEditors: Please set LastEditors
LastEditTime: 2024-06-10 18:12:48
FilePath: \Step5\model\recommendation_task.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import logging as lg
import json
import os


# Create a custom logger
logger = lg.getLogger('my_logger_rec')
logger.setLevel(lg.DEBUG)  # Set the desired logging level

# Create a FileHandler to write logs to a file
file_handler = lg.FileHandler('my_logfile_rec.log')
file_handler.setLevel(lg.DEBUG)  # Set the desired logging level for the file

# Create a formatter for the log messages
formatter = lg.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the FileHandler to the logger
logger.addHandler(file_handler)

class RecommenderModule(nn.Module):
    def __init__(self, user_embed_dim, interaction_embed_dim, social_embed_dim, entity_embed_dim
                 , n_entity, n_heads, ffn_dim, dropout, attention_dropout, movie_ids , device ):
        super().__init__()
        self.user_embed_dim = user_embed_dim
        self.interaction_embed_dim = interaction_embed_dim
        self.social_embed_dim = social_embed_dim
        self.entity_embed_dim = entity_embed_dim
        self.n_entity = n_entity
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.movie_ids = movie_ids
        self.device = device

        self._build_model()
        
        
    def _build_model(self):
        
        #transormation layer to user embedding 
        self.transform_items_embeddings = nn.Linear(self.interaction_embed_dim, self.user_embed_dim)
        self.transform_entity_embeddings = nn.Linear(self.entity_embed_dim, self.user_embed_dim)
        self.transform_social_embeddings = nn.Linear(self.social_embed_dim, self.user_embed_dim)


        #create multiplication layer
        # self.rec_item_bias = nn.Parameter(torch.zeros(self.n_items))
        self.rec_entity_bias = nn.Parameter(torch.zeros(self.n_entity)) 
        # self.rec_user_bias =nn.Parameter(torch.zeros(self.n_user))


        # transormation layer to items dim
        # self.item_entity_socres = nn.Linear(self.n_entity, self.n_items, bias=True)
        # self.item_user_socres = nn.Linear(self.n_user, self.n_items, bias=True)
        

        # loss function 
        self.rec_loss = nn.CrossEntropyLoss()

    def recall_at_k(self, labels,ranks, k):
        """
        Calculate the recall at k for tensors.
        
        Args:
            labels (Tensor): A batch of ground truth labels.
            ranks (Tensor): A batch of predicted rankings.
            k (int): The number of top elements to consider for NDCG calculation.
            
        Returns:
            float: The recall at k.
        """
      
        recall_scores = []
        # Calculate hits by checking if the ground truth items are in the top k predictions
        for rank in ranks :
            hits = torch.isin(labels, rank[:k]).sum().float()
            
            # Total number of relevant items
            total_relevant = labels.size(0)
            
            # Recall at k
            recall = hits / total_relevant if total_relevant > 0 else 0
            recall_scores.append(recall)
        return torch.tensor(recall_scores).mean()
    

    def metrics_cal_rec(self,scores,labels):
        metrics_rec= {}
        metrics_rec["recall@1"] = 0
        metrics_rec["recall@10"] = 0
        metrics_rec["recall@50"] = 0
        metrics_rec["count"] = 0
        batch_size = len(labels.view(-1).tolist())
        outputs = scores.cpu()
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        _, pred_idx = torch.topk(outputs, k=50, dim=1)
        for b in range(batch_size):
            if labels[b].item()==0:
                continue
            target_idx = self.movie_ids.index(labels[b].item())
            metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            metrics_rec["count"] += 1
        output_dict_rec={key: metrics_rec[key] / metrics_rec['count'] for key in metrics_rec}
        return output_dict_rec["recall@1"], output_dict_rec["recall@10"], output_dict_rec["recall@50"]
        
    def batch_calculate_metric(self, labels, ranks, k):
        """
        Compute NDCG metric for a batch of labels and predicted ranks.
        
        Args:
            labels (Tensor): A batch of ground truth labels.
            ranks (Tensor): A batch of predicted rankings.
            k (int): The number of top elements to consider for NDCG calculation.
            
        Returns:
            float: The sum of NDCG scores for the batch at k.
            float: The sum of Hit scores for the batch at K.
            float: The sum of MRR scores for the batch at K.
        """
        ndcg_scores = []
        hit_scores = []
        mrr_scores = []
        for label, rank in zip(labels, ranks):
            # Check if the label is within the top-k ranks
            is_label_in_top_k = torch.any(torch.isin(label, rank[:k]))
            
            if is_label_in_top_k:
                # Get the index of the label in the ranks
                label_rank = torch.where(rank == label)[0]
                
                # Check if the label rank is within the top-k
                if label_rank.size(0) > 0 and label_rank[0] < k:
                    label_rank = label_rank[0].float()  # Convert to float for division
                    ndcg_score = 1 / torch.log2(label_rank + 2)
                    mrr_score = 1 / (label_rank + 1)
                    
                    ndcg_scores.append(ndcg_score)
                    hit_scores.append(1.0)
                    mrr_scores.append(mrr_score)
                else:
                    ndcg_scores.append(0.0)
                    hit_scores.append(0.0)
                    mrr_scores.append(0.0)
            else:
                ndcg_scores.append(0.0)
                hit_scores.append(0.0)
                mrr_scores.append(0.0)
        
        # acccumulate the results
        return torch.tensor(ndcg_scores).sum(), torch.tensor(hit_scores).sum(), torch.tensor(mrr_scores).sum()

    def forward(self, user_model, batch):
        y = batch['movie_to_rec'].to(self.device)
        # workerIds = batch['worker_ids'].to(self.device)
        # initiatorIds = batch['initiator_ids'].to(self.device) 
        # respondentIds = batch['respondent_ids'].to(self.device)
       
        
        #social_embeddings, social_embeddings_initiator, social_embeddings_respondent  = user_model.social_graph.get_user_social_user_rep(batch)
        entity_embeddings = user_model.entity_graph.get_project_kg_rep(batch)  
        #interaction_embeddings, interaction_embeddings_initiator, interaction_embeddings_respondent= user_model.interaction_graph.get_user_interaction_item_rep(batch)
        
        
        entity_embs = user_model.entity_graph.get_all_embeddings()
        #user_embs = user_model.social_graph.get_all_embeddings()
        #item_embs = user_model.interaction_graph.get_all_embeddings()

        
        # entity_embs = self.transform_entity_embeddings(entity_embs).to(self.device)
        # user_embs = self.transform_social_embeddings(user_embs).to(self.device)
        # item_embs = self.transform_items_embeddings(item_embs).to(self.device) 
        

        # entity_profile = []
        # # social_profile = []
        # #interaction_profile = []

        # for index, workId in enumerate(workerIds):
        #     if workId ==  initiatorIds[index]:
        #         #entity_profile.append(kg_initiator_profile[index])
        #         #social_profile.append(social_embeddings_initiator[index])
        #         entity_profile.append(kg_initiator_profile[index])
        #     else:
        #         #entity_profile.append(kg_respondent_profile[index])
        #         #social_profile.append(social_embeddings_respondent[index])
        #         entity_profile.append(kg_respondent_profile[index])



        #entity_profile = torch.stack(entity_profile).to(self.device)  
        # social_profile = torch.stack(social_profile).to(self.device)  
        #interaction_profile = torch.stack(interaction_profile).to(self.device) 

        
        entities_score = F.linear(entity_embeddings, entity_embs, self.rec_entity_bias ).to(self.device)
        #users_score = F.linear(social_profile, user_embs, self.rec_user_bias ).to(self.device)
        #items_score = F.linear(interaction_profile, item_embs, self.rec_item_bias ).to(self.device)



        #entities_score = self.item_entity_socres(entities_score).to(self.device)
        #users_score = self.item_user_socres(users_score ).to(self.device)

        # rec_scores = items_score * entities_score * users_score
        # rec_scores = rec_scores.to(self.device)
        
        rec_loss = self.rec_loss(entities_score, y)
        # entities_score= entities_score[:, self.movie_ids]

        # _, rec_ranks = torch.topk(entities_score, 50, dim=-1)
        # rec_ranks = rec_ranks.to(self.device)
        r_at_1, r_at_10, r_at_50 = self.metrics_cal_rec(entities_score, y)
        # r_at_1 = self.recall_at_k(y, rec_ranks, 1)
        # r_at_10 = self.recall_at_k(y, rec_ranks, 10)
        # r_at_50 = self.recall_at_k(y, rec_ranks, 50)


        logger.info("****************************")
        logger.info(f"{r_at_1} R@1")
        logger.info(f"{r_at_10} R@10")
        logger.info(f"{r_at_50} R@50")
        logger.info(f"{rec_loss} rec_loss")
       
        

        return rec_loss, r_at_1, r_at_10, r_at_50
