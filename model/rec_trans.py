
## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14

import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn





    



class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # (bs, dim), (bs, dim), (bs, dim), None
        batch_size = query.size(0)

        # Linear projections in batch from d_model => h x d_k
        query = self.query_projection(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #(bs, n_heads, 1, d_k)
        key = self.key_projection(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #(bs, n_heads, 1, d_k)
        value = self.value_projection(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #(bs, n_heads, 1, d_k)

        # Apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k) #(bs, n_heads, 1, 1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)      #(bs, n_heads, 1, 1)
        p_attn = self.dropout(p_attn)           #(bs, n_heads, 1, 1)
        attn = torch.matmul(p_attn, value)      #(bs, n_heads, 1, d_k)

        # Concatenate heads and put through final linear layer
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k) #(bs, 1,  dim)
        output = self.out_projection(attn)                                                     #(bs, 1,  dim)

        return output  #(bs, 1,  dim)



class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout, att_dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn2 = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.cross_attn3 = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm4 = nn.LayerNorm(embed_dim)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, src,embs2, conv_embs, src_mask):
        # (bs, dim), None
        # Self-attention
        src2 = self.self_attn(src, src, src, mask=src_mask).squeeze(1)    #(bs,  dim)
        src = src + self.dropout1(src2)                                   #(bs,  dim)
        src = self.norm1(src)                                             #(bs,  dim)

        src2 = self.cross_attn2(src, embs2, embs2, mask=src_mask).squeeze(1)    #(bs,  dim)
        src = src + self.dropout2(src2)                                   #(bs,  dim)
        src = self.norm2(src)                                             #(bs,  dim)

        src2 = self.cross_attn3(src, conv_embs, conv_embs, mask=src_mask).squeeze(1)    #(bs,  dim)
        src = src + self.dropout3(src2)                                   #(bs,  dim)
        src = self.norm3(src)                                             #(bs,  dim)


        # Feed-forward
        src2 = self.ffn(src)                                              #(bs,  dim)
        src = src + self.dropout4(src2)                                   #(bs,  dim)
        src = self.norm4(src)                                             #(bs,  dim)

        return src  #(bs,  dim)

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, embed_dim, n_heads, ffn_dim, dropout, att_dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, n_heads, ffn_dim, dropout, att_dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
       

    def forward(self, src,embs2, conv_embs, src_mask):
        for layer in self.layers:
            src = layer(src, embs2, conv_embs,src_mask)  #(bs,  dim)
        src = self.norm(src)            #(bs,  dim)
        return src                      #(bs,  dim)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout, att_dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)
        self.cross_attn = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm_cross = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        
        self.cross_attn2 = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)

        self.norm_cross2 = nn.LayerNorm(embed_dim)
        self.dropout_cross2 = nn.Dropout(dropout)


        self.cross_attn3 = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)

        self.norm_cross3 = nn.LayerNorm(embed_dim)
        self.dropout_cross3 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, embs,embs2, conv_embs, tgt_mask, memory_mask):
        # (bs,  dim), (bs,  dim), None, None
        # Self-attention on the encoder output
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask).squeeze(1)              # (bs,  dim)
        tgt = tgt + self.dropout1(tgt2)                                             # (bs,  dim)
        tgt = self.norm1(tgt)                                                       # (bs,  dim)

        # Cross-attention with the recommeender user repesentation 
        tgt2 = self.cross_attn(tgt, embs, embs, mask=memory_mask).squeeze(1)        # (bs,  dim)
        tgt = tgt + self.dropout_cross(tgt2)                                             # (bs,  dim)
        tgt = self.norm_cross(tgt)                                                  # (bs,  dim)


        # Cross-attention with the seeker user repesentation 
        tgt2 = self.cross_attn2(tgt, embs2, embs2, mask=memory_mask).squeeze(1)        # (bs,  dim)
        tgt = tgt + self.dropout_cross2(tgt2)                                             # (bs,  dim)
        tgt = self.norm_cross2(tgt)    


        # Cross-attention with the conversation hsitory representation 
        tgt2 = self.cross_attn3(tgt, conv_embs, conv_embs, mask=memory_mask).squeeze(1)        # (bs,  dim)
        tgt = tgt + self.dropout_cross3(tgt2)                                             # (bs,  dim)
        tgt = self.norm_cross3(tgt)      
        


        # Feed-forward network
        tgt2 = self.ffn(tgt)                                                        # (bs,  dim)
        tgt = tgt + self.dropout3(tgt2)                                             # (bs,  dim)
        tgt = self.norm3(tgt)                                                       # (bs,  dim)

        return tgt # (bs,  dim)

class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, embed_dim, n_heads, ffn_dim, dropout, n_items, att_dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, n_heads, ffn_dim, dropout, att_dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, n_items)  # Project to the number of items

    def forward(self, tgt, embs,embs2, conv_embs, tgt_mask, memory_mask):
        #(bs,  dim), (bs,  dim), None, None
        for layer in self.layers:
            tgt = layer(tgt, embs,embs2, conv_embs ,tgt_mask, memory_mask) # (bs,  dim)
        tgt = self.norm(tgt)  # (bs,  dim)
        
        # Project the output to the number of items
        output = self.output_projection(tgt)  #(bs,  n_item)
        return output    #(bs,  n_item)


class RecommenderTransformer(nn.Module):
    def __init__(self, embed_dim, 
                 n_items, n_heads, ffn_dim, dropout, attention_dropout, n_layers, device):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_items = n_items
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.n_layers = n_layers
        self.device = device

       
        # Transformer Encoder
        self.encoder = TransformerEncoder(
            n_layers=self.n_layers,
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            att_dropout= attention_dropout
        )

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            n_layers=self.n_layers,
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            n_items=self.n_items,
            att_dropout= attention_dropout
        )

    def forward(self, user_embs,user_embs2, conv_embs, mask=None):
        #(bs,  dim), None
        # Pass through the encoder
        encoder_output = self.encoder(user_embs,user_embs2, conv_embs, mask) #(bs,  dim)

        # Pass through the decoder
        decoder_output = self.decoder(encoder_output, user_embs,user_embs2, conv_embs, mask, None)  #(bs,  n_item)

        return decoder_output   #(bs,  n_item)               