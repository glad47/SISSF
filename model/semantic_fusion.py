
import torch
import torch.nn as nn

from torch.nn import init
import torch.nn.functional as F
import pickle
from model.entitiy_relationships_graph import EntityRelationshipGraph
from model.user_interactions_social import SocialItemGraph
from model.user_interactions_graph import UserItemGraph
from model.user_social_graph import UserSocialGraph
from model.conversation_history_encoder import ConversationHistoryEncoder
from model.info_nce_loss import info_nce_loss
from model.rec_trans import TransformerEncoder

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class SemanticFusion(nn.Module):

    def __init__(self, dir_data, social_embed_dim, interaction_embed_dim, user_embed_dim, users, items, 
                 kg_emb_dim, n_entity, num_bases, graph_dataset,
                 token_embedding_weights, token_emb_dim, n_heads, n_layers, ffn_size, vocab_size,tok2ind, dropout, attention_dropout, 
                 relu_dropout, pad_token_idx, start_token_idx, learn_positional_embeddings, embeddings_scale, reduction, n_positions,
                 tem,  sem_dropout, device):
        super(SemanticFusion, self).__init__()

        self.dir_data = dir_data
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
        #self.social_graph = UserSocialGraph(self.dir_data, self.social_embed_dim, self.user_embed_dim,self.users,self.num_bases, self.device)
        self.interaction_graph = UserItemGraph(self.dir_data, self.interaction_embed_dim, self.user_embed_dim, self.items, self.num_bases, self.device)
        self.social_graph = SocialItemGraph(self.dir_data, self.interaction_embed_dim, self.user_embed_dim, self.items, self.num_bases, self.device)
        # for param in self.interaction_graph.parameters():
        #             param.requires_grad = False
        
        
        #self.entity_graph = EntityRelationshipGraph(self.kg_emb_dim, self.user_embed_dim, self.n_entity,self.num_bases, self.graph_dataset, self.device )
        self.conversation_encoder = ConversationHistoryEncoder(self.token_embedding_weights, self.token_emb_dim, self.user_embed_dim,self.tok2ind,self.dir_data,
                                                               self.n_heads,self.n_layers, self.ffn_size, self.vocab_size, self.dropout, self.attention_dropout,
                                                               self.relu_dropout, self.pad_token_idx, self.start_token_idx, self.learn_positional_embeddings,
                                                               self.embeddings_scale, self.reduction, self.n_positions, self.device)
        

        # self.encoder = TransformerEncoder(
        #             n_layers=4,
        #             embed_dim=self.user_embed_dim,
        #             n_heads=4,
        #             ffn_dim=self.ffn_size,
        #             dropout=self.sem_dropout,
        #             att_dropout= self.attention_dropout
        #         )

        
        
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
        social_embeddings, social_embeddings_initiator, social_embeddings_respondent  = self.social_graph.get_user_interaction_item_rep(batch)
        #entity_embeddings, kg_initiator_profile , kg_respondent_profile = self.entity_graph.get_kg_project_user_profile(batch)  
        # interaction_embeddings, interaction_embeddings_initiator, interaction_embeddings_respondent= self.interaction_graph.get_user_interaction_item_rep(batch)
        conv_history_embeddings = self.conversation_encoder.get_project_context_rep(batch)

       
        # social_embeddings_initiator = F.dropout(social_embeddings_initiator, p=0.5, training=self.training)
        # social_embeddings_respondent = F.dropout(social_embeddings_respondent, p=0.5, training=self.training)
        # interaction_embeddings_initiator = F.dropout(interaction_embeddings_initiator, p=0.5, training=self.training)
        # interaction_embeddings_respondent = F.dropout(interaction_embeddings_respondent, p=0.5, training=self.training)


        # Concatenate embeddings
        # combined_embeds_inti = torch.cat((interaction_embeddings, social_embeddings), dim=1)
        # combined_embeds_inti = torch.nn.ReLU()(combined_embeds_inti).to(self.device)
        # # Pass through the encoder
        # encoder_output_init = self.encoder(combined_embeds_inti, None)

        #print(encoder_output_init.shape)


        # Concatenate embeddings
        # combined_embeds_resp = torch.cat((social_embeddings_respondent, social_embeddings_initiator), dim=1)
        # combined_embeds_resp = torch.nn.ReLU()(combined_embeds_resp).to(self.device)
        # Pass through the encoder
        #encoder_output_resp = self.encoder(interaction_embeddings, None)
    
        # loss_initiator_e, TP_ei, FP_ei, FN_ei = self.calculate_info_nce_loss(conv_history_embeddings, interaction_embeddings , self.tem)


        loss_respondent_e, TP_er, FP_er, FN_er = self.calculate_info_nce_loss(conv_history_embeddings, social_embeddings, self.tem)


        
        # TP = TP_ei + TP_er 
        # FP = FP_ei + FP_er 
        # FN = FN_ei + FN_er  
     
        # final_loss = ( loss_initiator_e +  loss_respondent_e) / 2.0
        
        TP = TP_er 
        FP = FP_er 
        FN = FN_er  
     
        final_loss = loss_respondent_e


        # TP = TP_ei 
        # FP = FP_ei
        # FN = FN_ei
     
        # final_loss = loss_initiator_e 
        log.info("****************************")
        log.info(f"{TP} TP")
        log.info(f"{FP} FP")
        log.info(f"{(TP / (TP + FP)) * 100} precsion")
        log.info(f"{final_loss} loss")
        
       

        return final_loss, TP, FP, FN
    
    
    