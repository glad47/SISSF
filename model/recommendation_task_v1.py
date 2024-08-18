'''
Author: “glad47” ggffhh3344@gmail.com
Date: 2024-05-15 16:53:01
LastEditors: “glad47” ggffhh3344@gmail.com
LastEditTime: 2024-05-30 13:33:27
FilePath: \Step5\model\recommendation_task.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import logging as lg
from model.transformer import MultiHeadAttention, _normalize


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
    def __init__(self, user_embed_dim, interaction_embed_dim, social_embed_dim, entity_embed_dim, 
                 n_items, n_entity, n_user, n_heads, ffn_dim, dropout, attention_dropout, device ):
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
        self.device = device

        self._build_model()
        
        
    def _build_model(self):
        # Define convolutional layers for social graph embeddings
        self.social_conv1 = nn.Conv1d(in_channels=1, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.social_conv2 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.social_conv3 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=1, kernel_size=3, padding=1)
        self.social_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.social_adaptive_pool = nn.AdaptiveMaxPool1d(output_size=self.n_user)


        # Define convolutional layers for social graph embeddings
        self.interaction_conv1 = nn.Conv1d(in_channels=1, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.interaction_conv2 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.interaction_conv3 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=1, kernel_size=3, padding=1)
        self.interaction_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.interaction_adaptive_pool = nn.AdaptiveMaxPool1d(output_size=self.n_items)


        # Define convolutional layers for social graph embeddings
        self.relation_conv1 = nn.Conv1d(in_channels=1, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.relation_conv2 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.relation_conv3 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=1, kernel_size=3, padding=1)
        self.relation_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relation_adaptive_pool = nn.AdaptiveMaxPool1d(output_size=self.n_entity)

        #create multiplication layer
        self.rec_item_bias = nn.Parameter(torch.zeros(self.n_items))
        self.rec_entity_bias = nn.Parameter(torch.zeros(self.n_entity)) 
        self.rec_user_bias =nn.Parameter(torch.zeros(self.n_user))


        # transormation layer to items dim
        self.item_entity_socres = nn.Linear(self.n_entity, self.n_items, bias=True)
        self.item_user_socres = nn.Linear(self.n_user, self.n_items, bias=True)




      
        self.dropoutLayer = nn.Dropout(p=self.dropout)

        self.self_items_attention = MultiHeadAttention(
            self.n_heads, self.n_items, dropout=self.attention_dropout
        )
        self.norm_items_attention = nn.LayerNorm(self.n_items)

        self.social_cross_attention = MultiHeadAttention(
            self.n_heads, self.n_items, dropout=self.attention_dropout
        )
        self.norm_social_cross_attention = nn.LayerNorm(self.n_items)

        self.entity_cross_attention = MultiHeadAttention(
            self.n_heads, self.n_items, dropout=self.attention_dropout
        )
        self.norm_entity_cross_attention = nn.LayerNorm(self.n_items)

        # self.get_rec_score = nn.Linear(self.user_embed_dim, self.n_items)

        
        

        

        # loss function 
        self.rec_loss = nn.CrossEntropyLoss()


    def calculate_same_padding(self, kernel_size, stride):
        # Calculate the padding size required for 'same' padding
        return (math.ceil((stride - 1) + kernel_size) // 2)    

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
       
        
        entity_profile, _ , _ = user_model.entity_graph.get_kg_project_user_profile(batch) # (bs, user_dim), (n_entities, user_dim)
        social_profile = user_model.social_graph.get_user_social_project_user_profile(batch)
        interaction_profile, _ , _ = user_model.interaction_graph.get_user_interaction_project_user_item_profile(batch)
       
        entity_embs = user_model.entity_graph.get_all_embeddings()
        user_embs = user_model.social_graph.get_all_embeddings()
        item_embs = user_model.interaction_graph.get_all_embeddings()

       
       
        
        entities_score = F.linear(entity_profile, entity_embs, self.rec_entity_bias ).to(self.device)
        users_score = F.linear(social_profile, user_embs, self.rec_user_bias ).to(self.device)
        items_score = F.linear(interaction_profile, item_embs, self.rec_item_bias ).to(self.device)

        # Apply convolutional layers to social graph embeddings
        users_score = users_score.unsqueeze(1)  # Add channel dimension
        users_score = F.relu(self.social_conv1(users_score))
        users_score = self.social_pool(users_score)
        users_score = F.relu(self.social_conv2(users_score))
        users_score = self.social_pool(users_score)
        users_score = F.relu(self.social_conv3(users_score))
        users_score = self.social_pool(users_score)
        users_score = self.social_adaptive_pool(users_score)
        users_score = users_score.squeeze(1)  # Remove channel dimension


        # Apply convolutional layers to interaction graph embeddings
        items_score = items_score.unsqueeze(1)  # Add channel dimension
        items_score = F.relu(self.interaction_conv1(items_score))
        items_score = self.interaction_pool(items_score)
        items_score = F.relu(self.interaction_conv2(items_score))
        items_score = self.interaction_pool(items_score)
        items_score = F.relu(self.interaction_conv3(items_score))
        items_score = self.interaction_pool(items_score)
        items_score = self.interaction_adaptive_pool(items_score)
        items_score = items_score.squeeze(1)  # Remove channel dimension



        # Apply convolutional layers to relationship graph embeddings
        entities_score = entities_score.unsqueeze(1)  # Add channel dimension
        entities_score = F.relu(self.relation_conv1(entities_score))
        entities_score = self.relation_pool(entities_score)
        entities_score = F.relu(self.relation_conv2(entities_score))
        entities_score = self.relation_pool(entities_score)
        entities_score = F.relu(self.relation_conv3(entities_score))
        entities_score = self.relation_pool(entities_score)
        entities_score = self.relation_adaptive_pool(entities_score)
        entities_score = entities_score.squeeze(1)  # Remove channel dimension

        entities_score = self.item_entity_socres(entities_score).to(self.device)
        users_score = self.item_user_socres(users_score ).to(self.device)

        # Assuming items_score is a tensor from which you want to use the batch size
        bs = items_score.shape[0]

        # Create a 2D array of shape (bs, 1) with all elements set to 1
        mask = torch.ones((bs, 1), dtype=torch.float32).to(self.device)

        residual_item = items_score
        x = self.self_items_attention(query=items_score.unsqueeze(1), mask=mask).squeeze(1)
        x = self.dropoutLayer(x)  # --dropout
        x = x + residual_item
        x = _normalize(x, self.norm_items_attention)

        residual_user =x 
        x = self.social_cross_attention(query=x.unsqueeze(1), mask=mask, key= users_score.unsqueeze(1), value= users_score.unsqueeze(1)).squeeze(1)
        x = self.dropoutLayer(x)  # --dropout
        x = x + residual_user
        x = _normalize(x, self.norm_social_cross_attention)


        residual_entity =x 
        x = self.entity_cross_attention(query=x.unsqueeze(1), mask=mask, key= entities_score.unsqueeze(1), value= entities_score.unsqueeze(1)).squeeze(1)
        x = self.dropoutLayer(x)  # --dropout
        x = x + residual_entity
        x = _normalize(x, self.norm_entity_cross_attention)




        

        rec_scores = x
        
       
        
        rec_loss = self.rec_loss(rec_scores, y)


        _, rec_ranks = torch.topk(rec_scores, 50, dim=-1)
        rec_ranks = rec_ranks.to(self.device)
        hit_at_1, ndcg_at_1, mrr_at_1 = self.batch_calculate_metric(y, rec_ranks, 1)
        hit_at_10, ndcg_at_10, mrr_at_10 = self.batch_calculate_metric(y, rec_ranks, 10)
        hit_at_50, ndcg_at_50, mrr_at_50 = self.batch_calculate_metric(y, rec_ranks, 50)
        r_at_1 = self.recall_at_k(y, rec_ranks, 1)
        r_at_10 = self.recall_at_k(y, rec_ranks, 10)
        r_at_50 = self.recall_at_k(y, rec_ranks, 50)


        logger.info("****************************")
        logger.info(f"{r_at_1} R@1")
        logger.info(f"{r_at_10} R@10")
        logger.info(f"{r_at_50} R@50")
        logger.info(f"{rec_loss} rec_loss")
       
        

        return rec_loss, hit_at_1, hit_at_10, hit_at_50, ndcg_at_1, ndcg_at_10, ndcg_at_50, mrr_at_1, mrr_at_10, mrr_at_50, r_at_1, r_at_10, r_at_50
