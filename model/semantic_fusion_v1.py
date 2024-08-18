'''
Author: “glad47” ggffhh3344@gmail.com
Date: 2024-05-05 09:29:08
LastEditors: “glad47” ggffhh3344@gmail.com
LastEditTime: 2024-05-26 12:01:16
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
        self.social_graph = UserSocialGraph(self.dir_data, self.social_embed_dim, self.user_embed_dim, self.weights_loc_social,self.users, self.device)
        self.interaction_graph = UserItemGraph(self.dir_data, self.interaction_embed_dim, self.user_embed_dim, self.weights_loc_interaction, self.items, self.device)
        self.entity_graph = EntityRelationshipGraph(self.kg_emb_dim, self.user_embed_dim, self.n_entity,self.num_bases, self.graph_dataset, self.device )
        self.conversation_encoder = ConversationHistoryEncoder(self.token_embedding_weights, self.token_emb_dim, self.user_embed_dim, self.n_heads,
                                                               self.n_layers, self.ffn_size, self.vocab_size, self.dropout, self.attention_dropout,
                                                               self.relu_dropout, self.pad_token_idx, self.start_token_idx, self.learn_positional_embeddings,
                                                               self.embeddings_scale, self.reduction, self.n_positions, self.device)

        # Define convolutional layers for social graph embeddings
        self.social_conv1 = nn.Conv1d(in_channels=1, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.social_conv2 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.social_conv3 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=1, kernel_size=3, padding=1)
        self.social_pool = nn.MaxPool1d(kernel_size=2, stride=2)


        # Define convolutional layers for social graph embeddings
        self.interaction_conv1 = nn.Conv1d(in_channels=1, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.interaction_conv2 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.interaction_conv3 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=1, kernel_size=3, padding=1)
        self.interaction_pool = nn.MaxPool1d(kernel_size=2, stride=2)


        # Define convolutional layers for social graph embeddings
        self.relation_conv1 = nn.Conv1d(in_channels=1, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.relation_conv2 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.relation_conv3 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=1, kernel_size=3, padding=1)
        self.relation_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Define convolutional layers for conversation history embeddings
        self.conversation_conv1 = nn.Conv1d(in_channels=1, out_channels=self.user_embed_dim, kernel_size=3, padding=1)
        self.conversation_conv2 = nn.Conv1d(in_channels=self.user_embed_dim, out_channels=self.user_embed_dim * 2, kernel_size=3, padding=1)
        self.conversation_conv3 = nn.Conv1d(in_channels=self.user_embed_dim * 2, out_channels=1, kernel_size=3, padding=1)
        self.conversation_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.cross_entory_loss = nn.CrossEntropyLoss(reduction='none')
        
        

   

    def calculate_info_nce_loss(self, user_rep1, user_rep2, tem):
        features = torch.cat([user_rep1, user_rep2], dim=0) # (2*bs, dim)
        batch_size = user_rep1.shape[0]

        logits, labels = info_nce_loss(
            features, 
            bs=batch_size, 
            n_views=2, 
            device=self.device, 
            temperature=tem,
            hard_positive_scale=self.hard_positive_scale)
        
        loss = self.cross_entory_loss(logits, labels)

        probabilities = F.softmax(logits, dim=1)

        # Take the argmax to get predictions
        predictions = torch.argmax(probabilities, dim=1)
        

        # Find the max probability for each prediction
        max_probs, preds = torch.max(probabilities, dim=1)

        # Identify incorrect predictions
        incorrect_preds = preds != labels

        # Add an additional penalty term for incorrect predictions
        penalty = incorrect_preds.float() * max_probs * self.scale_factor

        # Combine the standard loss with the penalty term
        combined_loss = loss + penalty

      
        # True Positives: Predictions that are 0 and match the ground truth
        TP = ((predictions == 0) & (labels == 0)).sum()
        
        # False Positives: Predictions that are not 0 but the ground truth is 0
        FP = ((predictions != 0) & (labels == 0)).sum()
        
        # False Negatives: Ground truth positives that were not predicted correctly
        FN = ((predictions == 0) & (labels != 0)).sum()
        
        return torch.mean(combined_loss), TP, FP, FN


    def forward(self, batch):
        social_embeddings  = self.social_graph.get_user_social_project_user_profile(batch)
        interaction_embeddings, _, _ = self.interaction_graph.get_user_interaction_project_user_item_profile(batch)
        entity_embeddings, _ , _ = self.entity_graph.get_kg_project_user_profile(batch)  
        conv_history_embeddings = self.conversation_encoder.get_project_context_rep(batch)

        # Apply convolutional layers to social graph embeddings
        social_embeddings = social_embeddings.unsqueeze(1)  # Add channel dimension
        social_embeddings = F.relu(self.social_conv1(social_embeddings))
        social_embeddings = self.social_pool(social_embeddings)
        social_embeddings = F.relu(self.social_conv2(social_embeddings))
        social_embeddings = self.social_pool(social_embeddings)
        social_embeddings = F.relu(self.social_conv3(social_embeddings))
        social_embeddings = self.social_pool(social_embeddings)
        social_embeddings = social_embeddings.squeeze(1)  # Remove channel dimension


        # Apply convolutional layers to interaction graph embeddings
        interaction_embeddings = interaction_embeddings.unsqueeze(1)  # Add channel dimension
        interaction_embeddings = F.relu(self.interaction_conv1(interaction_embeddings))
        interaction_embeddings = self.interaction_pool(interaction_embeddings)
        interaction_embeddings = F.relu(self.interaction_conv2(interaction_embeddings))
        interaction_embeddings = self.interaction_pool(interaction_embeddings)
        interaction_embeddings = F.relu(self.interaction_conv3(interaction_embeddings))
        interaction_embeddings = self.interaction_pool(interaction_embeddings)
        interaction_embeddings = interaction_embeddings.squeeze(1)  # Remove channel dimension



        # Apply convolutional layers to relationship graph embeddings
        entity_embeddings = entity_embeddings.unsqueeze(1)  # Add channel dimension
        entity_embeddings = F.relu(self.relation_conv1(entity_embeddings))
        entity_embeddings = self.relation_pool(entity_embeddings)
        entity_embeddings = F.relu(self.relation_conv2(entity_embeddings))
        entity_embeddings = self.relation_pool(entity_embeddings)
        entity_embeddings = F.relu(self.relation_conv3(entity_embeddings))
        entity_embeddings = self.relation_pool(entity_embeddings)
        entity_embeddings = entity_embeddings.squeeze(1)  # Remove channel dimension

        # Apply convolutional layers to conversation history embeddings
        conv_history_embeddings = conv_history_embeddings.unsqueeze(1)  # Add channel dimension
        conv_history_embeddings = F.relu(self.conversation_conv1(conv_history_embeddings))
        conv_history_embeddings = self.conversation_pool(conv_history_embeddings)
        conv_history_embeddings = F.relu(self.conversation_conv2(conv_history_embeddings))
        conv_history_embeddings = self.conversation_pool(conv_history_embeddings)
        conv_history_embeddings = F.relu(self.conversation_conv3(conv_history_embeddings))
        conv_history_embeddings = self.conversation_pool(conv_history_embeddings)
        conv_history_embeddings = conv_history_embeddings.squeeze(1)  # Remove channel dimension


        
        loss_social, TP_s, FP_s, FN_s = self.calculate_info_nce_loss(conv_history_embeddings, social_embeddings, self.tem_social)
        loss_interaction, TP_i, FP_i, FN_i = self.calculate_info_nce_loss(conv_history_embeddings, interaction_embeddings, self.tem_interaction)
        loss_entity, TP_e, FP_e, FN_e = self.calculate_info_nce_loss(conv_history_embeddings, entity_embeddings, self.tem_entity)

        TP = TP_s + TP_i + TP_e
        FP = FP_s + FP_i + FP_e
        FN = FN_s + FN_i + FN_e
        loss = (loss_social + loss_interaction + loss_entity) / 3.0

        logger.info("****************************")
        logger.info(f"{TP} TP")
        logger.info(f"{FP} FP")
        logger.info(f"{(TP / (TP + FP)) * 100} precsion")
        logger.info(f"{loss} loss")
        
       

        return loss, TP, FP, FN
    
    
    