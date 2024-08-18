
import json
import os
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from util.logconf import logging
from model.conv_trans import ConversationTransformer
from model.rec_trans import RecommenderTransformer
import io
import pandas as pd
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
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
class CustomLoss(nn.Module):
    def __init__(self, pad_token_id):
        super(CustomLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy(logits, targets)
        predicted_tokens = logits.argmax(dim=-1)
        penalty = torch.sqrt(torch.mean((predicted_tokens != targets).float()))
        penalty.requires_grad_()
        # mse_loss = torch.mean((predicted_tokens.float() - targets.float()) ** 2)
        log.info(f"{penalty} right token loss")
        return ce_loss + penalty
    
class ConversationalModule(nn.Module):
    def __init__(self,user_embed_dim, social_embed_dim, n_heads, ffn_dim, dropout,
                     attention_dropout,relu_dropout, n_layers, voc_size, start_token_id,pad_token_id,end_token_id, response_truncate, 
                     decoder_token_prob_weight,embedding,embedding_size, embeddings_scale, learn_positional_embeddings, n_positions, device ):
        super().__init__()
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
        self.tok2ind = json.load(open(os.path.join('data/dataset', 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}    

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
        self.lin4 = nn.Linear(self.user_embed_dim, self.ffn_dim)
        # loss function ignore_index=self.pad_token_id CustomLoss(self.pad_token_id)
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.output_copy = nn.Linear(self.ffn_dim, self.voc_size)  # Project to the number of items
        self.copy_projection = nn.Linear(self.ffn_dim  * 2, self.ffn_dim  )
        self.output_projection_social = nn.Linear(self.ffn_dim, self.ffn_dim ) 
        self.output_projection = nn.Linear(self.ffn_dim, self.voc_size)
        # self.copyTrans = RecommenderTransformer(embed_dim= self.ffn_dim,n_items= self.voc_size,ffn_dim=self.ffn_dim ,dropout=self.dropout ,n_heads=self.n_heads ,attention_dropout=self.attention_dropout ,n_layers=self.n_layers ,device= self.device)

       

    
      
    # def metrics_cal_conv(self, preds):
    #     bigram_count = 0
    #     trigram_count=0
    #     quagram_count=0
    #     bigram_set = set()
    #     trigram_set=set()
    #     quagram_set=set()
    #     metrics_rec= {}
      
    #     for sen in preds:
    #         for start in range(len(sen) - 1):
    #             bg = str(sen[start]) + ' ' + str(sen[start + 1])
    #             bigram_count += 1
    #             bigram_set.add(bg)
    #         for start in range(len(sen)-2):
    #             trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
    #             trigram_count+=1
    #             trigram_set.add(trg)
    #         for start in range(len(sen)-3):
    #             quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
    #             quagram_count+=1
    #             quagram_set.add(quag)
    #         dis2 = len(bigram_set) / len(preds)  #bigram_count
    #         dis3 = len(trigram_set)/ len(preds)  #trigram_count
    #         dis4 = len(quagram_set)/ len(preds)  #quagram_count
    #     return dis2, dis3, dis4 

    def metrics_cal_conv(self, preds):
        bigram_count = 0
        trigram_count = 0
        quagram_count = 0
        bigram_set = set()
        trigram_set = set()
        quagram_set = set()
        metrics_rec = {}

        for sen in preds:
            # Find the position of the first end token
            
                      
            end_pos_tensor = (sen == self.end_token_id).nonzero(as_tuple=False)
            end_pos = end_pos_tensor[0].item() if end_pos_tensor.numel() > 0 else len(sen)
            for start in range(min(len(sen) - 1, end_pos - 1)):
                bg = str(sen[start]) + ' ' + str(sen[start + 1])
                bigram_count += 1
                bigram_set.add(bg)
            for start in range(min(len(sen) - 2, end_pos - 2)):
                trg = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                trigram_count += 1
                trigram_set.add(trg)
            for start in range(min(len(sen) - 3, end_pos - 3)):
                quag = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                quagram_count += 1
                quagram_set.add(quag)

        dis2 = len(bigram_set) / len(preds)  # bigram_count
        dis3 = len(trigram_set) / len(preds)  # trigram_count
        dis4 = len(quagram_set) / len(preds)  # quagram_count
        return dis2, dis3, dis4
    


    def get_copy_social_force_teaching(self, dialog_latent, social_reps):
        mask = (social_reps.sum(dim=-1, keepdim=True) == 0).squeeze(2)
        social_reps = self.output_projection_social(social_reps)
        # (bs, seq_len, ffn_size), (bs, n_social, ffn_size), (bs, nb_review)
        bs, seq_len, _ = dialog_latent.shape
        bs, n_social, _ = social_reps.shape

        dot_prod = dialog_latent.bmm(social_reps.transpose(1, 2)) # (bs, seq_len, n_social)
        # where to apply attention 
        # transfor user_emd to ffn_emd
        attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1) # (bs, seq_len, nb_review)
        dot_prod.masked_fill_(attn_mask.bool(), neginf(dot_prod.dtype)) # (bs, seq_len, nb_review)
        
    

        weight = F.softmax(dot_prod, dim=-1).type_as(social_reps) # (bs, seq_len, n_social)

        decode_atten_social_reps = weight.bmm(social_reps) # (bs, seq_len, ffn_size)
        decode_atten_social_reps = decode_atten_social_reps.view(bs, seq_len, self.ffn_dim) # (bs, seq_len, ffn_size)

        fusion_latent = torch.cat([dialog_latent, decode_atten_social_reps], dim=-1) # (bs, seq_len, ffn_size*2)
        fusion_latent = self.copy_projection(fusion_latent) # (bs, seq_len, ffn_size)

        return fusion_latent # (bs, seq_len, ffn_size)   
    


   
   
    def force_teaching(self, response,encoder_output, encoder_mask,conv_history_embeddings, social_embeddings,social_reps, user_model, sampling_prob=0.7):
        batch_size, seq_len = response.shape
        inp = response[:, 1:].to(self.device)
        current_end = response[:, -1:].to(self.device)
        new_end = torch.where((current_end == 0) | (current_end == 2), torch.tensor(0), torch.tensor(2))
        start = self.START.detach().expand(batch_size, 1).to(self.device)
        inputs = torch.cat((start, inp), dim=-1).long().to(self.device)   
        target = torch.cat((inp, new_end), dim=-1).long().to(self.device)      
      

        latent, _ = self.trans(inputs,encoder_output,encoder_mask, conv_history_embeddings, social_embeddings)
        # implementation of copy
        bs, seq_len, _ = latent.shape
        bs, n_social, _ = social_reps.shape
        mask = (social_reps.sum(dim=-1, keepdim=True) == 0).squeeze(2).to(self.device)
        social_reps_transformed = self.output_projection_social(social_reps).to(self.device)
      
        

        dot_prod = latent.bmm(social_reps_transformed.transpose(1, 2)).to(self.device) # (bs, seq_len, n_social)
        # where to apply attention 
        # transfor user_emd to ffn_emd
        attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1).to(self.device) # (bs, seq_len, nb_review)
        dot_prod.masked_fill_(attn_mask.bool(), neginf(dot_prod.dtype)).to(self.device) # (bs, seq_len, nb_review)
        
    

        weight = F.softmax(dot_prod, dim=-1).type_as(social_reps_transformed).to(self.device) # (bs, seq_len, n_social)

        decode_atten_social_reps = weight.bmm(social_reps).to(self.device) # (bs, seq_len, ffn_size)
        decode_atten_social_reps = decode_atten_social_reps.view(bs, seq_len, self.ffn_dim).to(self.device) # (bs, seq_len, ffn_size)



        copy_latent= self.copy_projection(torch.cat([decode_atten_social_reps ,latent],-1)).to(self.device)

        #logits = self.output(latent)
        con_logits = self.output_copy(copy_latent).to(self.device)   #F.linear(copy_latent, self.embeddings.weight)
        logits = con_logits *  self.decoder_token_prob_weight.to(self.device)
       
        sum_logits = logits
        _, preds = sum_logits.max(dim=2)
        # sum_logits = self.top_p_sampling(sum_logits)
        # probs = F.softmax(sum_logits, dim=-1)
        # # Reshape to (bs * seq, dim
        # flatpred = torch.multinomial(probs.view(-1, self.voc_size), 1).to(self.device)
        # # Reshape back to (bs, seq, num_samples)
        # preds = flatpred.view(bs, -1)


        # Scheduled Sampling: Replace ground truth tokens with model predictions based on sampling_prob
        # new_inputs = inputs.clone()
        # if self.training:
        #     new_inputs = inputs.clone()
        # for i in range(1, seq_len):
        #     if random.random() < sampling_prob:
        #             new_inputs[:, i] = preds[:, i - 1]
        # inputs = new_inputs

        # inputs = new_inputs

        #calculate loss
        flat_logits = sum_logits.view(-1, sum_logits.shape[-1]).to(self.device) # (bs*seq_len, nb_tok)
        flat_target = target.view(-1).to(self.device) # (bs*seq_len)
        loss = self.conv_loss(flat_logits, flat_target)
        log.info(f"{preds} inputs_token")
        # self.printCompareResult(preds, response)
        return loss, preds
    

    def greedy_selection_v1(self, response,encoder_output, encoder_mask,conv_history_embeddings, social_embeddings, social_reps, user_model):
        batch_size = response.shape[0]
        inputs = self.START.detach().expand(batch_size, 1).long().to(self.device)
        inputs = inputs.to(self.device)
        inp = response[:, 1:].to(self.device)
        current_end = response[:, -1:].to(self.device)
        new_end = torch.where((current_end == 0) | (current_end == 2), torch.tensor(0), torch.tensor(2)) 
        target = torch.cat((inp, new_end), dim=-1).long().to(self.device)  
        logits = []
        preds=[]
        for _ in range(self.response_truncate):
            curr_preds, _ = self.trans(inputs,encoder_output, encoder_mask, conv_history_embeddings, social_embeddings) # (bs, seq_len, dim)
            last_latent = curr_preds[:, -1:, :]
            # implementation of copy
            bs, seq_len, _ = last_latent.shape
            bs, n_social, _ = social_reps.shape
            
            mask = (social_reps.sum(dim=-1, keepdim=True) == 0).squeeze(2).to(self.device)
            social_reps_transformed = self.output_projection_social(social_reps).to(self.device)
        
            

            dot_prod = last_latent.bmm(social_reps_transformed.transpose(1, 2)).to(self.device) # (bs, seq_len, dim)
            # where to apply attention 
            # transfor user_emd to ffn_emd
            attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1).to(self.device) # (bs, seq_len, nb_review)
            dot_prod.masked_fill_(attn_mask.bool(), neginf(dot_prod.dtype)).to(self.device) # (bs, seq_len, nb_review)
            
        

            weight = F.softmax(dot_prod, dim=-1).type_as(social_reps_transformed).to(self.device) # (bs, seq_len, n_social)

            decode_atten_social_reps = weight.bmm(social_reps).to(self.device) # (bs, seq_len, ffn_size)
            # decode_atten_social_reps = decode_atten_social_reps.view(bs, seq_len, self.embedding_size).to(self.device) # (bs, seq_len, ffn_size)
            copy_latent= self.copy_projection(torch.cat([decode_atten_social_reps ,last_latent],-1)).to(self.device)
            #logits = self.output(latent)
            con_logits = self.output_copy(copy_latent).to(self.device)   #F.linear(copy_latent, self.embeddings.weight)
            logit = con_logits *  self.decoder_token_prob_weight.to(self.device)
       
            sum_logits = logit
            logits.append(sum_logits)
            # _, last_pred = sum_logits.max(dim=-1)
            sum_logits = self.top_p_sampling(sum_logits)
            probs = F.softmax(sum_logits, dim=-1)
            last_pred = torch.multinomial(probs.squeeze(1), 1).to(self.device)
            
            preds.append(last_pred)
            inputs = torch.cat((inputs, last_pred), dim=1) # (bs, gen_response_len)
            
            
           
            finished = ((inputs == self.end_token_id).sum(dim=-1) > 0).all().item() == batch_size
            if finished:
                log.info(f"have break!!")
                break
        log.info(f"{inputs} inputs_token")
        logits = torch.cat(logits, dim=1).to(self.device) # (bs, response_truncate, nb_tok)
        preds = torch.cat(preds, dim=1).to(self.device) # (bs, response_truncate)
        flat_logits = logits.view(-1, logits.shape[-1]).to(self.device) # (bs*seq_len, nb_tok)
       
        
        flat_target = target.view(-1).to(self.device) # (bs*seq_len)
        loss = self.conv_loss(flat_logits, flat_target)


        self.printCompareResult(preds, response)
        
      

        return loss, preds

    
    def apply_ngram_penalty(self, logits, generated_tokens, n=3):
        batch_size, seq_len, vocab_size = logits.size()
        for i in range(batch_size):
            for j in range(seq_len - n + 1):
                ngram = tuple(generated_tokens[i, j:j+n].tolist())
                if ngram in generated_tokens[i, :j].tolist():
                    logits[i, j+n-1, :] -= 20.0  # Apply a penalty
        return logits        
    def greedy_selection_v2(self, response,encoder_output, encoder_mask,conv_history_embeddings, social_embeddings, social_reps, user_model, beam_width=3, ngram_penalty=2):
        batch_size = response.shape[0]
        inputs = self.START.detach().expand(batch_size, 1).long().to(self.device)
        inputs = inputs.to(self.device)
        inp = response[:, 1:].to(self.device)
        current_end = response[:, -1:].to(self.device)
        new_end = torch.where((current_end == 0) | (current_end == 2), torch.tensor(0), torch.tensor(2)) 
        target = torch.cat((inp, new_end), dim=-1).long().to(self.device)  
        logits = []
        # preds=[]


        # Initialize beam scores and sequences
        beam_scores = torch.zeros(batch_size, beam_width).to(self.device)
        beam_sequences = inputs.unsqueeze(1).expand(batch_size, beam_width, -1).contiguous().view(batch_size * beam_width, -1)
        # Expand encoder outputs and masks
        encoder_output = encoder_output.repeat_interleave(beam_width, dim=0)
        encoder_mask = encoder_mask.repeat_interleave(beam_width, dim=0)
        conv_history_embeddings = conv_history_embeddings.repeat_interleave(beam_width, dim=0)
        social_embeddings = social_embeddings.repeat_interleave(beam_width, dim=0)
        social_reps = social_reps.repeat_interleave(beam_width, dim=0)
        target = target.repeat_interleave(beam_width, dim=0)
      
        for index in range(self.response_truncate):
            curr_preds, _ = self.trans(beam_sequences,encoder_output, encoder_mask, conv_history_embeddings, social_embeddings) # (bs, seq_len, dim)
            last_latent = curr_preds[:, -1:, :]
            # implementation of copy
            bs, seq_len, _ = last_latent.shape
            bs, n_social, _ = social_reps.shape
            
            mask = (social_reps.sum(dim=-1, keepdim=True) == 0).squeeze(2).to(self.device)
            social_reps_transformed = self.output_projection_social(social_reps).to(self.device)
        
            

            dot_prod = last_latent.bmm(social_reps_transformed.transpose(1, 2)).to(self.device) # (bs, seq_len, dim)
            # where to apply attention 
            # transfor user_emd to ffn_emd
            attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1).to(self.device) # (bs, seq_len, nb_review)
            dot_prod.masked_fill_(attn_mask.bool(), neginf(dot_prod.dtype)).to(self.device) # (bs, seq_len, nb_review)
            
        

            weight = F.softmax(dot_prod, dim=-1).type_as(social_reps_transformed).to(self.device) # (bs, seq_len, n_social)

            decode_atten_social_reps = weight.bmm(social_reps).to(self.device) # (bs, seq_len, ffn_size)
            # decode_atten_social_reps = decode_atten_social_reps.view(bs, seq_len, self.embedding_size).to(self.device) # (bs, seq_len, ffn_size)
            copy_latent= self.copy_projection(torch.cat([decode_atten_social_reps ,last_latent],-1)).to(self.device)
            #logits = self.output(latent)
            con_logits = self.output_copy(copy_latent).to(self.device)   #F.linear(copy_latent, self.embeddings.weight)
            logit = con_logits *  self.decoder_token_prob_weight.to(self.device)
       
            sum_logits = logit

            sum_logits = self.apply_ngram_penalty(sum_logits, beam_sequences, n=ngram_penalty)  # Apply n-gram penalty
            sum_logits = sum_logits.view(batch_size, beam_width, -1)
            logits.append(sum_logits)
            beam_scores = beam_scores.unsqueeze(-1) + sum_logits  # Update beam scores


            # _, last_pred = sum_logits.max(dim=-1)
            # sum_logits = self.top_p_sampling(sum_logits)
            # probs = F.softmax(sum_logits, dim=-1)
            # last_pred = torch.multinomial(probs.squeeze(1), 1).to(self.device)

             # Flatten beam scores and select top-k
            beam_scores_flat = beam_scores.view(batch_size, -1)
            topk_scores, topk_indices = torch.topk(beam_scores_flat, beam_width, dim=-1)
            # Update beam scores
            beam_scores = topk_scores

            # Update beam sequences
            beam_indices = topk_indices // sum_logits.size(-1)
            token_indices = topk_indices % sum_logits.size(-1)
            if index != self.response_truncate -1:
                beam_sequences = beam_sequences.view(batch_size, beam_width, -1)
                beam_sequences = beam_sequences.gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, beam_sequences.size(-1)))
                beam_sequences = torch.cat([beam_sequences, token_indices.unsqueeze(-1)], dim=-1)
                


            
            
                # preds.append(beam_sequences)
                # inputs = torch.cat((inputs, last_pred), dim=1) # (bs, gen_response_len)
                
                # Select the best beam for each batch
                best_beam_indices = beam_scores.argmax(dim=-1)
                best_beam_sequences = beam_sequences.gather(1, best_beam_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, beam_sequences.size(-1))).squeeze(1)
                beam_sequences =   beam_sequences.view(batch_size * beam_width, -1)
                finished = ((best_beam_sequences == self.end_token_id).sum(dim=-1) > 0).all().item() == batch_size
                if finished:
                    log.info(f"have break!!")
                    break
        
        # Select the best beam for each batch
        best_beam_indices = beam_scores.argmax(dim=-1)
        best_beam_sequences = beam_sequences.view(batch_size, beam_width, -1).gather(1, best_beam_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, beam_sequences.size(-1))).squeeze(1)
        logits = torch.cat(logits, dim=1).to(self.device) # (bs, response_truncate, nb_tok)
        preds = best_beam_sequences # (bs, response_truncate)
        log.info(f"{preds} inputs_token")
        flat_logits = logits.view(-1, logits.shape[-1]).to(self.device) # (bs*seq_len, nb_tok)
       
        
        flat_target = target.view(-1).to(self.device) # (bs*seq_len)
        loss = self.conv_loss(flat_logits, flat_target)


        self.printCompareResult(preds, response)
        
      

        return loss, preds 
    
            
    
            

      
    def printCompareResult(self, preds, response):
        sentences = []
        for pred in preds:
            sentence = ' '.join([self.ind2tok[token_id.item()] for token_id in pred])
            sentences.append(sentence)

        sentencesRes = []
        for pred in response:
            sentence = ' '.join([self.ind2tok[token_id.item()] for token_id in pred])
            sentencesRes.append(sentence)    

        df = pd.DataFrame({
            'Generated Response': sentences,
            'Real Response': sentencesRes
        })

        file_path = 'responses.xlsx'
    
        if os.path.exists(file_path):
            existing_data = pd.read_excel(file_path)
            combined_data = pd.concat([existing_data,df], ignore_index=True)
        else:
            combined_data = df

        combined_data.to_excel(file_path, index=False)   
    

    
    def forward(self, user_model, batch):
        response = batch['response']
        mask = batch['context_mask']
        mask = mask.to(self.device)
       
        social_embeddings, social_reps, social_embeddings_respondent  = user_model.social_graph.get_user_interaction_item_rep_recommendation(batch)
        encoder_output = user_model.conversation_encoder.get_encoder_rep(batch).to(self.device)
        conv_history_embeddings = user_model.conversation_encoder.get_project_context_rep(batch)
        conv_history_embeddings = self.lin1(conv_history_embeddings).to(self.device)
        social_embeddings = self.lin2(social_embeddings).to(self.device)
        social_reps = self.lin4(social_reps).to(self.device)
        
        if self.training:
            # if self.ind:
            #     loss, preds = self.force_teaching(response,social_embeddings,conv_history_embeddings,social_reps, user_model)         
            #     self.ind = 0
            # else:
            #     loss, preds = self.greedy_selection_v1(response,social_embeddings,conv_history_embeddings,social_reps, user_model)         
            #     self.ind = 1 
            loss, preds = self.force_teaching(response,encoder_output, mask,conv_history_embeddings, social_embeddings,social_reps, user_model)         
        else:
            loss, preds = self.greedy_selection_v1(response,encoder_output, mask,conv_history_embeddings, social_embeddings,social_reps,user_model)         
        # loss, preds = self.greedy_selection_v1(response,social_embeddings,conv_history_embeddings, user_model) 
      
        dist_2, dist_3, dist_4 = self.metrics_cal_conv(preds)
       


        log.info("****************************")
        log.info(f"{dist_2} dist-2")
        log.info(f"{dist_3} dist-3")
        log.info(f"{dist_4} dist-4")
        log.info(f"{loss} conv_loss")
       
        

        return loss, dist_2, dist_3, dist_4
    

    def top_p_sampling(self, logits, p=0.95):
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        if sorted_indices_to_remove[..., 1:].sum() > 0:
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
        
        # Set logits to -inf for tokens to be removed
        sorted_logits[sorted_indices_to_remove] =neginf(torch.float16)
        
        # Restore the original order of logits
        original_logits = torch.zeros_like(logits).fill_(neginf(torch.float16))
        original_logits.scatter_(2, sorted_indices, sorted_logits)
        
        return original_logits

    # def force_teaching_v2(self, response, social_embeddings, conv_history_embeddings,social_reps, user_model):
    #     batch_size, seq_len = response.shape
    #     inp = response[:, 1:].to(self.device)
    #     start = self.START.detach().expand(batch_size, 1).to(self.device)
    #     inputs = torch.cat((start, inp), dim=-1).long().to(self.device)  
    #     input =inputs[:, 0].unsqueeze(-1)
    #     inputs_token = input
    #     input = user_model.conversation_encoder.embedding(input)
    #     inputs = inputs.to(self.device)
    #     input = input.to(self.device)
        
    #     inputs_token = inputs_token.to(self.device)
    #     logits = []
    #     preds=[]

    #     for index in range(self.response_truncate):
    #         curr_preds = self.trans(input,conv_history_embeddings, social_embeddings,social_reps, None) # (bs, seq_len, dim)
    #         last_latent = curr_preds[:, -1:, :]
    #         # implementation of copy
    #         bs, seq_len, _ = last_latent.shape
    #         bs, n_social, _ = social_reps.shape
    #         mask = (social_reps.sum(dim=-1, keepdim=True) == 0).squeeze(2).to(self.device)
    #         social_reps_transformed = self.output_projection_social(social_reps).to(self.device)
        
            

    #         dot_prod = last_latent.bmm(social_reps_transformed.transpose(1, 2)).to(self.device) # (bs, seq_len, n_social)
    #         # where to apply attention 
    #         # transfor user_emd to ffn_emd
    #         attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1).to(self.device) # (bs, seq_len, nb_review)
    #         dot_prod.masked_fill_(attn_mask.bool(), neginf(dot_prod.dtype)).to(self.device) # (bs, seq_len, nb_review)
            
        

    #         weight = F.softmax(dot_prod, dim=-1).type_as(social_reps_transformed).to(self.device) # (bs, seq_len, n_social)

    #         decode_atten_social_reps = weight.bmm(social_reps_transformed).to(self.device) # (bs, seq_len, ffn_size)
    #         decode_atten_social_reps = decode_atten_social_reps.view(bs, seq_len, self.ffn_dim).to(self.device) # (bs, seq_len, ffn_size)
    #         copy_latent= self.copy_projection(torch.cat([decode_atten_social_reps ,last_latent],-1)).to(self.device)
    #         #logits = self.output(latent)
    #         con_logits = self.output_copy(copy_latent).to(self.device)   #F.linear(copy_latent, self.embeddings.weight)
    #         logit = self.output_projection(last_latent).to(self.device) * self.decoder_token_prob_weight.to(self.device)
       
    #         sum_logits = logit
    #         _, last_pred = sum_logits.max(dim=-1)
    #         # sum_logits = self.top_p_sampling(sum_logits)
    #         # probs = F.softmax(sum_logits, dim=-1)
    #         # last_pred = torch.multinomial(probs.squeeze(1), 1).to(self.device)
    #         logits.append(sum_logits)
    #         preds.append(last_pred)
    #         if index != self.response_truncate - 1:
    #             inputs_token = torch.cat((inputs_token,  inputs[:, index + 1 ].unsqueeze(-1)), dim=1) # (bs, gen_response_len)
    #             input = user_model.conversation_encoder.embedding(inputs_token).to(self.device)



            
            
            
           
    #         finished = ((inputs_token == self.end_token_id).sum(dim=-1) > 0).all().item() == batch_size
    #         if finished:
    #             log.info(f"have break!!")
    #             break
        
    #     logits = torch.cat(logits, dim=1).to(self.device) # (bs, response_truncate, nb_tok)
    #     preds = torch.cat(preds, dim=1).to(self.device) # (bs, response_truncate)
    #     log.info(f"{preds} inputs_token")
    #     flat_logits = logits.view(-1, logits.shape[-1]).to(self.device) # (bs*seq_len, nb_tok)
       
        
    #     target = response.view(-1).to(self.device) # (bs*seq_len)
    #     loss = self.conv_loss(flat_logits, target)

      

    #     return loss, preds  

    # def greedy_selection_v2(self, response, social_embeddings, conv_history_embeddings,social_reps, user_model, beam_width= 3):
    #     batch_size = response.shape[0]
    #     inputs = self.START.detach().expand(batch_size, 1).long().to(self.device)
    #     inputs_token = inputs
    #       # Convert list to tensor
        
    #     # Initialize the beam with the start token
    #     logits = []
    #     preds=[]
    #     for _ in range(self.response_truncate):
    #         inputs = user_model.conversation_encoder.embedding(inputs)
    #         inputs = inputs.to(self.device)
    #         score = [0] * batch_size
    #         score = torch.tensor(score, dtype=torch.float32).to(self.device)
            
    #         curr_preds = self.trans(inputs, conv_history_embeddings, social_embeddings,social_reps, None)  # (bs, seq_len, dim)
    #         last_pred = curr_preds[:, -1, :]  # (bs, 1, dim)
    #         log_probs = F.log_softmax(last_pred, dim=-1).to(self.device)  # (bs, 1, vocab_size)

    #         # Get top beam_width predictions
    #         topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
    #         topk_log_probs = topk_log_probs.to(self.device)
    #         topk_indices = topk_indices.to(self.device)
    #         # Expand dimensions to match batch size and beam width
    #         next_tokens = user_model.conversation_encoder.embedding(topk_indices).to(self.device)  # (bs, beam_width, dim)
    #         next_tokens = next_tokens.unsqueeze(2).expand(-1, -1, inputs.size(1), -1).to(self.device)  # (bs, beam_width, seq_len, dim)

          
    #         # Concatenate input tokens with topk indices for each beam
    #         expanded_inputs_token = inputs_token.unsqueeze(1).expand(-1, beam_width, -1).to(self.device)  # (bs, beam_width, seq_len)
    #         new_inputs_token = torch.cat((expanded_inputs_token, topk_indices.unsqueeze(2)), dim=2).to(self.device)  # (bs, beam_width, seq_len + 1)

    #         # Calculate new scores for each beam
    #         new_scores = score.unsqueeze(1) + topk_log_probs.to(self.device)  # (bs, beam_width)


    #         # Select the best beam for each item in the batch
    #         best_beam_indices = torch.argmax(new_scores, dim=1).to(self.device)  # (bs)
    #         best_new_inputs_token = new_inputs_token[torch.arange(new_inputs_token.size(0)), best_beam_indices].to(self.device)  # (bs, seq_len + 1)

    #         # Update inputs and inputs_token for the next iteration
    #         inputs = best_new_inputs_token

            
           
            
    #         # Check if all sequences have generated the end token
    #         finished = ((inputs == self.end_token_id).sum(dim=-1) > 0).sum().item() == batch_size
    #         if finished:
    #             log.info(f"have break!!")
    #             break
            
    #         last_tokens = best_new_inputs_token[:, -1].to(self.device)  # Shape: (bs,)

    #         # Initialize a zero tensor of shape (bs, vocab_size)
    #         one_hot_tensor = torch.zeros((batch_size, self.voc_size), device=self.device)

    #         # Set the positions specified in last_tokens to 1
    #         one_hot_tensor.scatter_(1, last_tokens.unsqueeze(1), 1)
            
    #         # Calculate loss
    #         preds.append(one_hot_tensor.unsqueeze(1).to(self.device))   # Remove the start token

            
    #     preds = torch.cat(preds, dim=1).to(self.device)    
    #     flat_preds = preds.view(-1, preds.shape[-1]).to(self.device)  # (bs*seq_len, nb_tok)
    #     target = response.view(-1).to(self.device)  # (bs*seq_len)
    #     loss = self.conv_loss(flat_preds, target)
    #     loss.requires_grad_(True)
    #     preds = torch.argmax(preds, dim=-1)
            
    #     return loss, preds  



  # 