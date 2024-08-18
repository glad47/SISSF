'''
Author: “glad47” ggffhh3344@gmail.com
Date: 2024-05-15 16:53:01
LastEditors: Please set LastEditors
LastEditTime: 2024-06-11 17:15:10
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
                 n_items, n_entity, n_user, n_heads, ffn_dim, dropout, attention_dropout, scale_factor, device ):
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
        self.scale_factor = scale_factor
        self.device = device

        self._build_model()
        
        
    def _build_model(self):
        # Define convolutional layers for social graph embeddings
        self.social_conv1 = nn.Conv1d(in_channels=1, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.social_conv2 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.social_conv3 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=1, kernel_size=3, padding=1)
        self.social_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.social_adaptive_pool = nn.AdaptiveMaxPool1d(output_size=self.user_embed_dim)


        # Define convolutional layers for social graph embeddings
        self.interaction_conv1 = nn.Conv1d(in_channels=1, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.interaction_conv2 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.interaction_conv3 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=self.user_embed_dim * 4, kernel_size=3, padding=1)
        self.interaction_conv4 = nn.Conv1d(in_channels=self.user_embed_dim * 4, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.interaction_conv5 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.interaction_conv6 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=1, kernel_size=3, padding=1)
        self.interaction_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.interaction_adaptive_pool = nn.AdaptiveMaxPool1d(output_size=self.user_embed_dim)


        # Define convolutional layers for social graph embeddings
        self.relation_conv1 = nn.Conv1d(in_channels=1, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.relation_conv2 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.relation_conv3 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=self.user_embed_dim * 4, kernel_size=3, padding=1)
        self.relation_conv4 = nn.Conv1d(in_channels=self.user_embed_dim * 4, out_channels=self.user_embed_dim * 6, kernel_size=3, padding=1)
        self.relation_conv5 = nn.Conv1d(in_channels=self.user_embed_dim * 6, out_channels=self.user_embed_dim * 8, kernel_size=3, padding=1)
        self.relation_conv6 = nn.Conv1d(in_channels=self.user_embed_dim * 8, out_channels=self.user_embed_dim * 6, kernel_size=3, padding=1)
        self.relation_conv7 = nn.Conv1d(in_channels=self.user_embed_dim * 6, out_channels=self.user_embed_dim * 4, kernel_size=3, padding=1)
        self.relation_conv8 = nn.Conv1d(in_channels=self.user_embed_dim * 4, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.relation_conv9 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.relation_conv10 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=1, kernel_size=3, padding=1)
        
        self.relation_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relation_adaptive_pool = nn.AdaptiveMaxPool1d(output_size=self.user_embed_dim)

        #create multiplication layer
        self.rec_item_bias = nn.Parameter(torch.zeros(self.n_items))
        self.rec_entity_bias = nn.Parameter(torch.zeros(self.n_entity)) 
        self.rec_user_bias =nn.Parameter(torch.zeros(self.n_user))


        # transormation layer to items dim
        # self.item_entity_socres = nn.Linear(self.n_entity, self.n_items, bias=True)
        # self.item_user_socres = nn.Linear(self.n_user, self.n_items, bias=True)




      
        self.dropoutLayer = nn.Dropout(p=self.dropout)

        self.self_items_attention = MultiHeadAttention(
            self.n_heads, self.user_embed_dim, dropout=self.attention_dropout
        )
        self.norm_items_attention = nn.LayerNorm(self.user_embed_dim)

        self.social_cross_attention = MultiHeadAttention(
            self.n_heads, self.user_embed_dim, dropout=self.attention_dropout
        )
        self.norm_social_cross_attention = nn.LayerNorm(self.user_embed_dim)

        self.entity_cross_attention = MultiHeadAttention(
            self.n_heads, self.user_embed_dim, dropout=self.attention_dropout
        )
        self.norm_entity_cross_attention = nn.LayerNorm(self.user_embed_dim)

        self.get_rec_score1 = nn.Linear(self.user_embed_dim, self.user_embed_dim * 4,bias=True)
        self.get_rec_score2 = nn.Linear(self.user_embed_dim * 4, self.user_embed_dim * 8,bias=True)
        self.get_rec_score3 = nn.Linear(self.user_embed_dim * 8, self.user_embed_dim * 16,bias=True)
        self.get_rec_score4 = nn.Linear(self.user_embed_dim * 16, self.user_embed_dim * 32,bias=True)
        self.get_rec_score5 = nn.Linear(self.user_embed_dim * 32, self.user_embed_dim * 64,bias=True)
        self.get_rec_score6 = nn.Linear(self.user_embed_dim * 64, self.user_embed_dim * 64,bias=True)
        self.get_rec_score7 = nn.Linear(self.user_embed_dim * 64, self.n_items,bias=True)

        
        

        

        # loss function 
        self.rec_loss = nn.CrossEntropyLoss()


    def calculate_same_padding(self, kernel_size, stride):
        # Calculate the padding size required for 'same' padding
        return (math.ceil((stride - 1) + kernel_size) // 2)    

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
        workerIds = batch['worker_ids'].to(self.device)
        initiatorIds = batch['initiator_ids'].to(self.device) 
        respondentIds = batch['respondent_ids'].to(self.device)
       
        
        entity_embeddings, kg_initiator_profile , kg_respondent_profile= user_model.entity_graph.get_kg_project_user_profile(batch) # (bs, user_dim), (n_entities, user_dim)
        social_embeddings, social_embeddings_initiator, social_embeddings_respondent   = user_model.social_graph.get_user_social_user_rep(batch)
        interaction_embeddings, interaction_embeddings_initiator, interaction_embeddings_respondent= user_model.interaction_graph.get_user_interaction_item_rep(batch)
       
        entity_embs = user_model.entity_graph.get_all_embeddings()
        user_embs = user_model.social_graph.get_all_embeddings()
        item_embs = user_model.interaction_graph.get_all_embeddings()

       
       
        entity_profile = []
        social_profile = []
        interaction_profile = []

        for index, workId in enumerate(workerIds):
            if workId ==  initiatorIds[index]:
                entity_profile.append(kg_initiator_profile[index])
                social_profile.append(social_embeddings_initiator[index])
                interaction_profile.append(interaction_embeddings_initiator[index])
            else:
                entity_profile.append(kg_respondent_profile[index])
                social_profile.append(social_embeddings_respondent[index])
                interaction_profile.append(interaction_embeddings_respondent[index])



        entity_profile = torch.stack(entity_profile).to(self.device)  
        social_profile = torch.stack(social_profile).to(self.device)  
        interaction_profile = torch.stack(interaction_profile).to(self.device) 

        
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
        items_score = F.relu(self.interaction_conv4(items_score))
        items_score = self.interaction_pool(items_score)
        items_score = F.relu(self.interaction_conv5(items_score))
        items_score = self.interaction_pool(items_score)
        items_score = F.relu(self.interaction_conv6(items_score))
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
        entities_score = F.relu(self.relation_conv4(entities_score))
        entities_score = self.relation_pool(entities_score)
        entities_score = F.relu(self.relation_conv5(entities_score))
        entities_score = self.relation_pool(entities_score)

        entities_score = F.relu(self.relation_conv6(entities_score))
        entities_score = self.relation_pool(entities_score)
        entities_score = F.relu(self.relation_conv7(entities_score))
        entities_score = self.relation_pool(entities_score)
        entities_score = F.relu(self.relation_conv8(entities_score))
        entities_score = self.relation_pool(entities_score)
        entities_score = F.relu(self.relation_conv9(entities_score))
        entities_score = self.relation_pool(entities_score)
        entities_score = F.relu(self.relation_conv10(entities_score))
        entities_score = self.relation_pool(entities_score)
        entities_score = self.relation_adaptive_pool(entities_score)
        entities_score = entities_score.squeeze(1)  # Remove channel dimension

        # entities_score = self.item_entity_socres(entities_score).to(self.device)
        # users_score = self.item_user_socres(users_score ).to(self.device)

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




        

        rec_scores = self.get_rec_score1(x)
        rec_scores = self.get_rec_score2(rec_scores)
        rec_scores = self.get_rec_score3(rec_scores)
        rec_scores = self.get_rec_score4(rec_scores)
        rec_scores = self.get_rec_score5(rec_scores)
        rec_scores = self.get_rec_score6(rec_scores)
        rec_scores = self.get_rec_score7(rec_scores)
       


        
        rec_loss = self.rec_loss(rec_scores, y)

        # probabilities = F.softmax(rec_scores, dim=1)


        # # Take the argmax to get predictions
        # predictions = torch.argmax(probabilities, dim=1)
        

        # # Find the max probability for each prediction
        # max_probs, preds = torch.max(probabilities, dim=1)

        # # Identify incorrect predictions
        # incorrect_preds = preds != y

        # # Add an additional penalty term for incorrect predictions
        # penalty = incorrect_preds.float() * max_probs * self.scale_factor

        # # Combine the standard loss with the penalty term
        # combined_loss = torch.mean(rec_loss + penalty)


        
      


        # _, rec_ranks = torch.topk(rec_scores, 50, dim=-1)
        # rec_ranks = rec_ranks.to(self.device)
        r_at_1, r_at_10, r_at_50 = self.metrics_cal_rec(rec_scores, y)

        logger.info("****************************")
        logger.info(f"{r_at_1} R@1")
        logger.info(f"{r_at_10} R@10")
        logger.info(f"{r_at_50} R@50")
        logger.info(f"{rec_loss} rec_loss")
        # logger.info(f"{incorrect_preds} incorrect_preds")
        
       
        

        return rec_loss, r_at_1, r_at_10, r_at_50
