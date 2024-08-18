'''
Author: “glad47” ggffhh3344@gmail.com
Date: 2024-05-05 09:29:08
LastEditors: “glad47” ggffhh3344@gmail.com
LastEditTime: 2024-06-06 14:00:47
Description: user-item graph 
'''
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pickle
from model.entitiy_relationships_graph import EntityRelationshipGraph
from model.user_interactions_graph import UserItemGraph
from model.user_social_graph import UserSocialGraph
from model.conversation_history_encoder import ConversationHistoryEncoder
from model.info_nce_loss import info_nce_loss
import logging as lg


# Create a custom logger
logger = lg.getLogger('my_logger')
logger.setLevel(lg.DEBUG)  # Set the desired logging level

# Create a FileHandler to write logs to a file
file_handler = lg.FileHandler('my_logfile.log')
file_handler.setLevel(lg.DEBUG)  # Set the desired logging level for the file

# Create a formatter for the log messages
formatter = lg.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the FileHandler to the logger
logger.addHandler(file_handler)
class SemanticFusion(nn.Module):
    def __init__(self, dir_data, social_embed_dim, interaction_embed_dim, user_embed_dim, users, items, 
                 weights_loc_social, weights_loc_interaction, kg_emb_dim, n_entity, num_bases, graph_dataset,
                 token_embedding_weights, token_emb_dim, n_heads, n_layers, ffn_size, vocab_size, dropout, attention_dropout, 
                 relu_dropout, pad_token_idx, start_token_idx, learn_positional_embeddings, embeddings_scale, reduction, n_positions,
                 tem_social, tem_interaction, tem_entity, hard_positive_scale, confidence_threshold, scale_factor, device):
        super(SemanticFusion, self).__init__()

        self.dir_data = dir_data
        self.social_embed_dim = social_embed_dim
        self.interaction_embed_dim = interaction_embed_dim
        self.weights_loc_social= weights_loc_social
        self.weights_loc_interaction= weights_loc_interaction
        self.user_embed_dim = user_embed_dim
        self.users = users
        self.items = items
        self.kg_emb_dim = kg_emb_dim 
        self.n_entity = n_entity
        self.graph_dataset = graph_dataset
        self.num_bases = num_bases
        self.token_embedding_weights = token_embedding_weights
        self.token_emb_dim = token_emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_size = ffn_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.pad_token_idx = pad_token_idx
        self.start_token_idx = start_token_idx
        self.learn_positional_embeddings = learn_positional_embeddings
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        self.n_positions = n_positions
        self.tem_social = tem_social
        self.tem_interaction = tem_interaction
        self.tem_entity = tem_entity
        self.hard_positive_scale = hard_positive_scale
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        self.device = device
        self._build_model()
    
    def _build_model(self):
        # self.social_graph = UserSocialGraph(self.dir_data, self.social_embed_dim, self.user_embed_dim, self.weights_loc_social,self.users, self.device)
        # self.interaction_graph = UserItemGraph(self.dir_data, self.interaction_embed_dim, self.user_embed_dim, self.weights_loc_interaction, self.items, self.device)
        # for param in self.interaction_graph.parameters():
        #             param.requires_grad = False
        
        
        self.entity_graph = EntityRelationshipGraph(self.kg_emb_dim, self.user_embed_dim, self.n_entity,self.num_bases, self.graph_dataset, self.device )
        self.conversation_encoder = ConversationHistoryEncoder(self.token_embedding_weights, self.token_emb_dim, self.user_embed_dim, self.n_heads,
                                                               self.n_layers, self.ffn_size, self.vocab_size, self.dropout, self.attention_dropout,
                                                               self.relu_dropout, self.pad_token_idx, self.start_token_idx, self.learn_positional_embeddings,
                                                               self.embeddings_scale, self.reduction, self.n_positions, self.device)

        self.cross_entory_loss = nn.CrossEntropyLoss(reduction='none')

    def calculate_info_nce_loss(self, user_rep1, user_rep2,tem):
            # user_rep1 = (bs, dim), user_rep2 = (times * bs, dim)
            features= []
            index_pair = {}
            count = 0
            for index,user in enumerate(user_rep1):
                if len(user_rep2[index]) > 1 :
                    for user2 in user_rep2[index]:
                        if user2.numel() == 0:
                            continue
                        features.append(torch.cat([user, user2], dim=0))
                        if index_pair.get(index) is None:
                            index_pair[index] = []
                        index_pair[index].append(count)
                        count += 1   
            #features = torch.cat([user_rep1, user_rep2], dim=0) # (2*bs, dim)
            
            features = torch.stack(features, dim=0)
            
            batch_size = user_rep1.shape[0]

            logits, labels = info_nce_loss(
                features, 
                bs=batch_size, 
                n_views=2, 
                device=self.device, 
                temperature=tem)
            
            loss = self.cross_entory_loss(logits, labels)
            mean_loss = torch.zeros(user_rep1.shape[0], device=self.device, dtype=torch.float)
            for index, user2_indices in index_pair.items():
                lossValue = 0
                for user2_index in user2_indices:
                    lossValue += loss[user2_index]
                mean_loss[index] = lossValue / len(user2_indices)
            assert mean_loss.shape[0] ==  user_rep1.shape[0] 


            probabilities = F.softmax(logits, dim=1)

            # Take the argmax to get predictions
            predictions = torch.argmax(probabilities, dim=1)
            

            

        
            # True Positives: Predictions that are 0 and match the ground truth
            TP = ((predictions == 0) & (labels == 0)).sum()
            
            # False Positives: Predictions that are not 0 but the ground truth is 0
            FP = ((predictions != 0) & (labels == 0)).sum()
            
            # False Negatives: Ground truth positives that were not predicted correctly
            FN = ((predictions == 0) & (labels != 0)).sum()
            
            return mean_loss.mean(), TP, FP, FN     


    def forward(self, batch):
       

        entity_embeddings, _ , _ = self.entity_graph.get_kg_user_rep_entity(batch)  
        conv_history_embeddings = self.conversation_encoder.get_project_context_rep(batch)

       
        
        
        loss_entity, TP_e, FP_e, FN_e = self.calculate_info_nce_loss(conv_history_embeddings, entity_embeddings, self.tem_entity)

        #TP = TP_s + TP_i + TP_e
        #FP = FP_s + FP_i + FP_e
        #FN = FN_s + FN_i + FN_e
        #loss = (loss_conversational + loss_interaction + loss_entity) / 3.0

        logger.info("****************************")
        logger.info(f"{TP_e} TP")
        logger.info(f"{FP_e} FP")
        logger.info(f"{(TP_e / (TP_e + FP_e)) * 100} precsion")
        logger.info(f"{loss_entity} loss")
        
       

        return loss_entity, TP_e, FP_e, FN_e
    
    
    