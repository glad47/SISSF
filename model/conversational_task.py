## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14
import json
import os
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from util.logconf import logging
from model.conv_trans import ConversationTransformer
import io
import pandas as pd
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

import random


NEAR_INF_FP16 = 65504
NEAR_INF = 1e20 
 
def neginf(dtype):
    """Returns a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


class Handle_Cross_Entropy_Loss(nn.Module):
    """
     from https://github.com/Zyh716/WSDM2022-C2CRS
    """
    def __init__(self, ignore_index=-1, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        if ignore_index != -1 and self.weight is not None:
            self.weight[ignore_index] = 0.0

    def forward(self, output, target):
        # (nb_sample, nb_class), (nb_sample)
        _softmax = F.softmax(output, dim=1)
        neg_log_softmax = -torch.log(_softmax)

        nb_sample = 0
        loss = []

        for y_hat, y in zip(neg_log_softmax, target):
            if y == self.ignore_index:
                continue
            nb_sample += 1
            y_hat_c = self.weight[y] * y_hat[y] if self.weight is not None else y_hat[y]
            loss.append(y_hat_c)
        
        loss = sum(loss) / nb_sample


        return loss
    
class ConversationalModule(nn.Module):
    def __init__(self,dpath, dtype, user_embed_dim, social_embed_dim, n_heads, ffn_dim, dropout,
                     attention_dropout,relu_dropout, n_layers, voc_size, start_token_id,pad_token_id,end_token_id, response_truncate, 
                     decoder_token_prob_weight,embedding,embedding_size, embeddings_scale, learn_positional_embeddings, n_positions, cumulative_prob_th, device ):
        super().__init__()
        self.dpath = dpath
        self.dtype = dtype
        self.user_embed_dim = user_embed_dim
        self.social_embed_dim = social_embed_dim
      
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.n_layers = n_layers
        self.voc_size = voc_size
        self.start_token_id = start_token_id
        self.pad_token_id = pad_token_id
        self.end_token_id = end_token_id
        self.response_truncate = response_truncate
        self.decoder_token_prob_weight = decoder_token_prob_weight
        self.embedding = embedding
        self.embedding_size =embedding_size
        self.embeddings_scale= embeddings_scale
        self.learn_positional_embeddings = learn_positional_embeddings
        self.n_positions = n_positions
        self.tok2ind = json.load(open(os.path.join(self.dpath,self.dtype, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()} 
        self.cumulative_prob_th = cumulative_prob_th   

        self.device = device

        self.ind = 0
        self.initEmbeddings()
        self._build_model()

    def initEmbeddings(self):
        if self.embedding is not None:
                buffer = io.BytesIO()
                torch.save(self.embedding, buffer)
                buffer.seek(0)  # Reset buffer position to the beginning
                pretrained_embeddings = torch.load(buffer)
                self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False,padding_idx=self.pad_token_id)
        else :
            self.embedding = nn.Embedding(self.voc_size, self.ffn_dim, self.pad_token_id)
            nn.init.normal_(self.embedding.weight, mean=0, std=self.user_embed_dim ** -0.5)
            nn.init.constant_(self.embedding.weight[self.pad_token_id], 0)      
        
    
    
    def _build_model(self):
        self.register_buffer('START', torch.tensor([self.start_token_id], dtype=torch.long)) 
        self.trans = ConversationTransformer(embed_dim= self.embedding_size, user_emb= self.user_embed_dim,voc_size= self.voc_size,n_heads=self.n_heads,ffn_dim=self.ffn_dim ,dropout=self.dropout ,
                                             attention_dropout=self.attention_dropout, relu_dropout = self.relu_dropout ,n_layers=self.n_layers 
                                             ,device= self.device,pad_inx=self.pad_token_id, embedding= self.embedding,embeddings_scale= self.embeddings_scale,
                                              learn_positional_embeddings= self.learn_positional_embeddings,n_positions= self.n_positions )

        self.trans = self.trans.to(self.device)
        self.lin1 = nn.Linear(self.user_embed_dim, self.ffn_dim)
        self.lin2 = nn.Linear(self.user_embed_dim, self.ffn_dim)
        self.lin3 = nn.Linear(self.user_embed_dim, self.ffn_dim)
        # loss function ignore_index=self.pad_token_id CustomLoss(self.pad_token_id)
        # self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.conv_loss = Handle_Cross_Entropy_Loss(ignore_index=self.pad_token_id, weight=self.decoder_token_prob_weight.squeeze())
        self.output_copy = nn.Linear(self.ffn_dim, self.voc_size)  # Project to the number of items
        self.copy_projection = nn.Linear(self.ffn_dim  * 2, self.ffn_dim  )
        self.output_projection_social = nn.Linear(self.ffn_dim, self.ffn_dim ) 
    

       


    def metrics_cal_conv(self, preds):
        # (bs, seq_len)  
        bigram_set = set()
        trigram_set = set()
        quagram_set = set()
        for sen in preds:
            # Find the position of the first end token    
            end_pos_tensor = (sen == self.end_token_id).nonzero(as_tuple=False)
            end_pos = end_pos_tensor[0].item() if end_pos_tensor.numel() > 0 else len(sen)
            for start in range(min(len(sen) - 1, end_pos - 1)):
                bg = str(sen[start]) + ' ' + str(sen[start + 1])
                bigram_set.add(bg)
            for start in range(min(len(sen) - 2, end_pos - 2)):
                trg = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                trigram_set.add(trg)
            for start in range(min(len(sen) - 3, end_pos - 3)):
                quag = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                quagram_set.add(quag)

        dis2 = len(bigram_set) / len(preds)  # bigram_count
        dis3 = len(trigram_set) / len(preds)  # trigram_count
        dis4 = len(quagram_set) / len(preds)  # quagram_count
        return dis2, dis3, dis4
    

    

    def top_p_sampling(self, logits):  # (bs, 1, voc_size)
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # (bs, 1, voc_size)
        
        # Calculate cumulative probabilities for each token
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) # (bs, 1, voc_size)
        
        # Remove tokens with cumulative probability above the threshold , create mask
        sorted_indices_to_remove = cumulative_probs > self.cumulative_prob_th  # (bs, 1, voc_size)
        if sorted_indices_to_remove[..., 1:].sum() > 0:
            # ensure that the first token that exceeds the cumulative probability threshold is included in the sampling process
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() # (bs, 1, voc_size)
            # ensure the first toekn cumulative probability is zero 
            sorted_indices_to_remove[..., 0] = 0  # (bs, 1, voc_size)
        
        # Set logits to -inf for tokens to be removed
        sorted_logits[sorted_indices_to_remove] =neginf(torch.float16) # (bs, 1, voc_size)
        
        # Restore the original order of logits
        original_logits = torch.zeros_like(logits).fill_(neginf(torch.float16)) # (bs, 1, voc_size)
        original_logits.scatter_(2, sorted_indices, sorted_logits) # (bs, 1, voc_size)
        
        return original_logits     # (bs, 1, voc_size)
    
       

   
    def force_teaching(self, response,encoder_output, encoder_mask,conv_history_embeddings, social_embeddings,social_reps, user_model, sampling_prob=0.7):
        batch_size, seq_len = response.shape
        inp = response[:, 1:].to(self.device) # (bs, response_truncate -1)
        current_end = response[:, -1:].to(self.device) # (bs, 1)
        new_end = torch.where((current_end == 0) | (current_end == 2), torch.tensor(0), torch.tensor(2)) # (bs, 1)
        start = self.START.detach().expand(batch_size, 1).to(self.device)  # (bs, 1)
        inputs = torch.cat((start, inp), dim=-1).long().to(self.device)    # (bs, response_truncate)
        target = torch.cat((inp, new_end), dim=-1).long().to(self.device)  # (bs, response_truncate)    
      

        latent, _ = self.trans(inputs,encoder_output,encoder_mask, conv_history_embeddings, social_embeddings)   # (bs, seq_len, dim), None
        # implementation of copy
        bs, seq_len, _ = latent.shape
        bs, n_social, _ = social_reps.shape
        mask = (social_reps.sum(dim=-1, keepdim=True) == 0).squeeze(2).to(self.device) # (bs, n_social)
        social_reps_transformed = self.output_projection_social(social_reps).to(self.device)  # (bs, n_social, dim)
      
        

        dot_prod = latent.bmm(social_reps_transformed.transpose(1, 2)).to(self.device) # (bs, seq_len, n_social)
        # where to apply attention 
        # transfor user_emd to ffn_emd
        attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1).to(self.device) # (bs, seq_len, n_social)
        dot_prod.masked_fill_(attn_mask.bool(), neginf(dot_prod.dtype)).to(self.device) # (bs, seq_len, n_social)
        
    

        weight = F.softmax(dot_prod, dim=-1).type_as(social_reps_transformed).to(self.device) # (bs, seq_len, n_social)

        decode_atten_social_reps = weight.bmm(social_reps).to(self.device) # (bs, seq_len, ffn_size)



        copy_latent= self.copy_projection(torch.cat([decode_atten_social_reps ,latent],-1)).to(self.device) # (bs, seq_len, ffn_size)

        #logits = self.output(latent)
        con_logits = self.output_copy(copy_latent).to(self.device)   # (bs, seq_len, voc_size)
        logits = con_logits *  self.decoder_token_prob_weight.to(self.device) # (bs, seq_len, voc_size)
       
        sum_logits = logits   # (bs, seq_len, voc_size)
        _, preds = sum_logits.max(dim=2) # (bs, seq_len)
       

        #calculate loss
        flat_logits = sum_logits.view(-1, sum_logits.shape[-1]).to(self.device) # (bs*response_truncate, voc_size)
        flat_target = target.view(-1).to(self.device) # (bs*response_truncate)
        loss = self.conv_loss(flat_logits, flat_target)
        log.info(f"{preds} inputs_token")
        return loss, preds  # (bs, response_truncate)
    

    def nucleus_sampling(self, response,encoder_output, encoder_mask,conv_history_embeddings, social_embeddings, social_reps, user_model):
        batch_size = response.shape[0]
        inputs = self.START.detach().expand(batch_size, 1).long().to(self.device) # (bs, 1)
        inputs = inputs.to(self.device)
        inp = response[:, 1:].to(self.device)  # (bs, response_truncate -1)
        current_end = response[:, -1:].to(self.device) # (bs, 1)
        new_end = torch.where((current_end == 0) | (current_end == 2), torch.tensor(0), torch.tensor(2)) # (bs, 1) 
        target = torch.cat((inp, new_end), dim=-1).long().to(self.device)   # (bs, response_truncate)
        logits = []
        preds=[]
        for _ in range(self.response_truncate):
            curr_preds, _ = self.trans(inputs,encoder_output, encoder_mask, conv_history_embeddings, social_embeddings) # (bs, seq_len, dim)
            last_latent = curr_preds[:, -1:, :]
            # implementation of copy
            bs, seq_len, _ = last_latent.shape  # (1)
            bs, n_social, _ = social_reps.shape
            
            mask = (social_reps.sum(dim=-1, keepdim=True) == 0).squeeze(2).to(self.device)  # (bs, n_social)
            social_reps_transformed = self.output_projection_social(social_reps).to(self.device) # (bs, n_social, dim)
        
            

            dot_prod = last_latent.bmm(social_reps_transformed.transpose(1, 2)).to(self.device) # (bs, 1, n_social)
            # where to apply attention 
            # transfor user_emd to ffn_emd
            attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1).to(self.device) # (bs, 1, n_social)
            dot_prod.masked_fill_(attn_mask.bool(), neginf(dot_prod.dtype)).to(self.device) # (bs, 1, n_social)
            
        

            weight = F.softmax(dot_prod, dim=-1).type_as(social_reps_transformed).to(self.device) # (bs, 1, n_social)

            decode_atten_social_reps = weight.bmm(social_reps).to(self.device) # (bs, 1, ffn_size)
            copy_latent= self.copy_projection(torch.cat([decode_atten_social_reps ,last_latent],-1)).to(self.device) # (bs, 1, ffn_size)
            #logits = self.output(latent)
            con_logits = self.output_copy(copy_latent).to(self.device)   # (bs, 1, ffn_size)
            logit = con_logits *  self.decoder_token_prob_weight.to(self.device) # (bs, 1, voc_size)
       
            sum_logits = logit  # (bs, 1, voc_size)
            logits.append(sum_logits)
            # _, last_pred = sum_logits.max(dim=-1)
            sum_logits = self.top_p_sampling(sum_logits)  # (bs, 1, voc_size)
            probs = F.softmax(sum_logits, dim=-1)  # (bs, 1, voc_size)
            last_pred = torch.multinomial(probs.squeeze(1), 1).to(self.device) # (bs, 1)
            
            preds.append(last_pred)
            inputs = torch.cat((inputs, last_pred), dim=1) # (bs, gen_response_len)
            
            
           
            finished = ((inputs == self.end_token_id).sum(dim=-1) > 0).all().item() == batch_size
            if finished:
                log.info(f"have break!!")
                break
        log.info(f"{inputs} inputs_token")
        logits = torch.cat(logits, dim=1).to(self.device) # (bs, response_truncate, voc_size)
        preds = torch.cat(preds, dim=1).to(self.device) # (bs, response_truncate)
        flat_logits = logits.view(-1, logits.shape[-1]).to(self.device)  # (bs*response_truncate, voc_size)
       
        
        flat_target = target.view(-1).to(self.device) # (bs*response_truncate)
        loss = self.conv_loss(flat_logits, flat_target) #(1)
        

        return self.printCompareResult(preds)  # (1), (bs, response_truncate)

    
    
   
    
            
    
            
    def printCompareResult(self, preds):
        # sentences = []
        for pred in preds:
            sentence = ' '.join([self.ind2tok[token_id.item()] for token_id in pred])
      

        return sentence
       

           
      
    
    

    
    def forward(self, user_model, batch):
        response = batch['response'] # (bs, seq_len)
        mask = batch['context_mask'] # (bs, context_length)
        mask = mask.to(self.device)  # (bs, context_length)
        social_information  = user_model.social_info.get_social_information_rep(batch) # (bs, dim)
        _, _ , social_reps  = user_model.social_info.get_social_rep_recommendation(batch)  # (bs, user_dim), (bs, user_dim), (bs, ns_social,user_dim)
        encoder_output = user_model.conversation_encoder.get_encoder_rep(batch).to(self.device)   # (bs, context_length, dim)
        conv_history_embeddings = user_model.conversation_encoder.get_project_context_rep(batch)  # (bs, user_dim)
        conv_history_embeddings = self.lin1(conv_history_embeddings).to(self.device)              # (bs, dim)
        social_information = self.lin2(social_information).to(self.device)                          # (bs, dim)
        social_reps = self.lin3(social_reps).to(self.device)                                      # (bs, ns_social, dim)
        
        if self.training:
            loss, preds = self.force_teaching(response,encoder_output, mask,conv_history_embeddings, social_information,social_reps, user_model)    # (1), (bs, seq_len) 
            dist_2, dist_3, dist_4 = self.metrics_cal_conv(preds)    
            return loss, dist_2, dist_3, dist_4
        else:
            sentences = self.nucleus_sampling(response,encoder_output, mask,conv_history_embeddings, social_information,social_reps,user_model) # (1), (bs, seq_len)               
            return sentences
        # log.info("****************************")
        # log.info(f"{dist_2} dist-2")
        # log.info(f"{dist_3} dist-3")
        # log.info(f"{dist_4} dist-4")
        # log.info(f"{loss} conv_loss")
        
    

    

    