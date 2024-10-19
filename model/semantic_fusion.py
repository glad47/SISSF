## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pickle
from model.social_information import SocialInformation 
from model.conversation_history_encoder import ConversationHistoryEncoder
from model.info_nce_loss import info_nce_loss


from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class SemanticFusion(nn.Module):

    def __init__(self, dir_data, dtype, social_embed_dim, interaction_embed_dim, user_embed_dim, users, items, 
                 kg_emb_dim, n_entity, num_bases, graph_dataset,
                 token_embedding_weights, token_emb_dim, n_heads, n_layers, ffn_size, vocab_size,tok2ind, dropout, attention_dropout, 
                 relu_dropout, pad_token_idx, start_token_idx, learn_positional_embeddings, embeddings_scale, reduction, n_positions,
                 tem,  sem_dropout, device):
        super(SemanticFusion, self).__init__()

        self.dir_data = dir_data
        self.dtype = dtype
        self.social_embed_dim = social_embed_dim
        self.interaction_embed_dim = interaction_embed_dim
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
        self.tok2ind = tok2ind
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.pad_token_idx = pad_token_idx
        self.start_token_idx = start_token_idx
        self.learn_positional_embeddings = learn_positional_embeddings
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        self.n_positions = n_positions
        self.tem= tem
        self.sem_dropout = sem_dropout
   
        self.device = device
        self._build_model()
    
    def _build_model(self):
        self.social_info = SocialInformation(self.dir_data,self.dtype, self.interaction_embed_dim, self.user_embed_dim, self.items, self.num_bases, self.device)
        
        self.conversation_encoder = ConversationHistoryEncoder(self.token_embedding_weights, self.token_emb_dim, self.user_embed_dim,self.tok2ind,self.dir_data,self.n_heads,self.n_layers, self.ffn_size, self.vocab_size, self.dropout, self.attention_dropout,
                                                               self.relu_dropout, self.pad_token_idx, self.start_token_idx, self.learn_positional_embeddings,
                                                               self.embeddings_scale, self.reduction, self.n_positions, self.device)
        
        self.cross_entory_loss = nn.CrossEntropyLoss()

       
        
        

   

    def calculate_info_nce_loss(self, user_rep1, user_rep2, tem):
        features = torch.cat([user_rep1, user_rep2], dim=0) # (2*bs, dim)
        batch_size = user_rep1.shape[0]

        logits, labels = info_nce_loss(
            features, 
            bs=batch_size, 
            n_views=2, 
            device=self.device, 
            temperature=tem)
        
        loss = self.cross_entory_loss(logits, labels)

        probabilities = F.softmax(logits, dim=1)

        # Take the argmax to get predictions
        predictions = torch.argmax(probabilities, dim=1)
        

        

      
        # True Positives: Predictions that are 0 and match the ground truth
        TP = ((predictions == 0) & (labels == 0)).sum()
        
        # False Positives: Predictions that are not 0 but the ground truth is 0
        FP = ((predictions != 0) & (labels == 0)).sum()
        
        # False Negatives: Ground truth positives that were not predicted correctly
        FN = ((predictions == 0) & (labels != 0)).sum()
        
        return loss, TP, FP, FN


    def forward(self, batch):
        social_information  = self.social_info.get_social_information_rep(batch) # (bs, dim)
        conv_history_embeddings = self.conversation_encoder.get_project_context_rep(batch) # (bs, dim)

       
       

        loss, TP, FP, FN = self.calculate_info_nce_loss(conv_history_embeddings, social_information, self.tem)


        
    
        
      


        
        log.info("****************************")
        log.info(f"{TP} TP")
        log.info(f"{FP} FP")
        log.info(f"{(TP / (TP + FP)) * 100} precsion")
        log.info(f"{loss} loss")
        
       

        return loss, TP, FP, FN
    
    
    