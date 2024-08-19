## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14
import torch
import fasttext
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pickle
import numpy as np
import os
from model.attention import SelfAttentionBatch, SelfAttentionSeq
from model.transformer import TransformerEncoder
from tqdm import tqdm
from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class ConversationHistoryEncoder(nn.Module):
    def __init__(self, embedding_weights, token_emb_dim, user_emb_dim,token2Ids,dir_data,  n_heads, 
                 n_layers, ffn_size, vocab_size, dropout, attention_dropout, relu_dropout, pad_token_idx,
                 start_token_idx, learn_positional_embeddings, embeddings_scale, reduction, n_positions, device):
        super(ConversationHistoryEncoder, self).__init__()
        self.embedding_weights = embedding_weights
        self.token_emb_dim = token_emb_dim
        self.user_emb_dim = user_emb_dim
        self.token2Ids = token2Ids
        self.dir_data =dir_data
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
        self.device = device
        self= self.to(self.device)

        self._build_model()

    def _build_model(self):
        self.initEmbeddings()
        self._build_context_transformer_layer()
        self._build_context_cl_project_head()
        self._build_atten()
    
    
    def initEmbeddings(self):
        if self.embedding_weights is not None:
                pretrained_embeddings = torch.load(self.embedding_weights)
                self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False,padding_idx=self.pad_token_idx)
        else :
            self.embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.embedding.weight, mean=0, std=self.user_emb_dim ** -0.5)
            nn.init.constant_(self.embedding.weight[self.pad_token_idx], 0)   
            self.load_fasttext_embeddings() 

    def load_fasttext_embeddings(self):
        path_data = self.dir_data + "/cc.en.300.bin"
        # Load the pre-trained FastText model
        log.info("Load embedding")
        fasttext_model = fasttext.load_model(path_data)
        for word, idx in tqdm(self.token2Ids.items()):
            if word in fasttext_model:
                self.embedding.weight.data[idx] = torch.tensor(fasttext_model.get_word_vector(word))         
               
    def _build_context_transformer_layer(self):
        self.register_buffer('C_START', torch.tensor([self.start_token_idx], dtype=torch.long))

        self.context_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )



    def _build_context_cl_project_head(self):
        self.context_project_head_fc1 = nn.Linear(self.token_emb_dim, self.token_emb_dim)
        self.context_project_head_fc2 = nn.Linear(self.token_emb_dim, self.user_emb_dim)

    def _build_atten(self):
        self.ContextHiddenStateAttenFunc = SelfAttentionSeq(self.token_emb_dim, self.token_emb_dim)

    

    def get_project_context_rep(self, batch):
        context = batch['context'].to(self.device)  # (bs, seq_len) it might be context_tokens
        context_mask = batch['context_mask'].to(self.device)  # (bs, seq_len)
        context_pad_mask = batch['context_pad_mask'].to(self.device)  # (bs, seq_len)    this is maybe the entites_mask_in_context

        context_user_rep = self._get_context_user_rep(context, context_mask, context_pad_mask) # (bs, dim), (bs, seq_len, dim)
        project_context_user_rep = self._project_context_user_rep(context_user_rep)  # (bs, user_dim)

        return project_context_user_rep

    def get_encoder_rep(self, batch):
        context = batch['context'].to(self.device)  # (bs, seq_len) it might be context_tokens
        context_mask = batch['context_mask'].to(self.device)  # (bs, seq_len)
        context_pad_mask = batch['context_pad_mask'].to(self.device)  # (bs, seq_len)    this is maybe the entites_mask_in_context

        encoder_rep = self._get_encoder_rep(context, context_mask, context_pad_mask) # (bs, dim), (bs, seq_len, dim)
        return encoder_rep # (bs, user_dim)
    
    def _get_encoder_rep(self, context, context_mask, context_pad_mask):
        cls_state, state = self._get_hidden_state_context_transformer(context)  # (bs, dim), (bs, seq_len, dim)

       
        return state  # (bs, dim)
       
    def _get_context_user_rep(self, context, context_mask, context_pad_mask):
        cls_state, state = self._get_hidden_state_context_transformer(context)  # (bs, dim), (bs, seq_len, dim)
        atten_last_state = self.ContextHiddenStateAttenFunc(state, context_pad_mask)  # (bs, dim)

        assert len(atten_last_state.shape) == 2
        return atten_last_state  # (bs, dim)

    def _get_hidden_state_context_transformer(self, context):
        state, mask = self.context_encoder(context)
        cls_state = state[:, 0, :] # (bs, dim)

        return cls_state, state

    def _project_context_user_rep(self, context_user_rep):
        # context_user_rep = (bs, dim)
        context_user_rep = self.context_project_head_fc1(context_user_rep) # (bs, dim)
        context_user_rep = F.relu(context_user_rep) # (bs, dim)
        context_user_rep = self.context_project_head_fc2(context_user_rep) # (bs, user_dim)

        return context_user_rep # (bs, user_dim)