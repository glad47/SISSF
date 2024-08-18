

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
        batch_size = query.size(0)

        # Linear projections in batch from d_model => h x d_k
        query = self.query_projection(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.key_projection(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.value_projection(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        attn = torch.matmul(p_attn, value)

        # Concatenate heads and put through final linear layer
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.out_projection(attn)

        return output



class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout, att_dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Self-attention
        src2 = self.self_attn(src, src, src, mask=src_mask).squeeze(1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, embed_dim, n_heads, ffn_dim, dropout, att_dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, n_heads, ffn_dim, dropout, att_dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
       

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        src = self.norm(src)
        return src   


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout, att_dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)
        self.cross_attn = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm_cross = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, embs, tgt_mask, memory_mask):
        # Self-attention on the target sequence
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask).squeeze(1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention with the encoder's output (memory)
        tgt2 = self.cross_attn(tgt, embs, embs, mask=memory_mask).squeeze(1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm_cross(tgt)

        


        # Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, embed_dim, n_heads, ffn_dim, dropout, n_items, att_dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, n_heads, ffn_dim, dropout, att_dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, n_items)  # Project to the number of items

    def forward(self, tgt, embs, tgt_mask, memory_mask):
        for layer in self.layers:
            tgt = layer(tgt, embs ,tgt_mask, memory_mask)
        tgt = self.norm(tgt)
        
        # Project the output to the number of items
        output = self.output_projection(tgt)
        return output     


class SingleRecommenderTransformer(nn.Module):
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

    def forward(self, embs, mask):
        # Embedding lookups
        

        # Concatenate embeddings
     
        #embs = torch.nn.ReLU()(embs).to(self.device)
        # Pass through the encoder
        encoder_output = self.encoder(embs, None)

        # Pass through the decoder
        decoder_output = self.decoder(encoder_output, embs, None, None)  # Assuming no masks for simplicity

        return decoder_output                    