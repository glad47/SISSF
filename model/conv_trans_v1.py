


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
        seq_len = query.size(1)

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
        if seq_len == 1:
            attn = attn.transpose(1, 2).contiguous().view(batch_size, self.n_heads * self.d_k)
            attn = attn.unsqueeze(1)  # Add the sequence length dimension back
        else:
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_k)
        output = self.out_projection(attn)

        return output

class MultiHeadAttentionSocial(nn.Module):
    def __init__(self, n_heads, d_model, user_dim,  dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(user_dim, d_model)  # Project key to d_model
        self.value_projection = nn.Linear(user_dim, d_model)  # Project value to d_model
        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        mask_key = (key.sum(dim=-1, keepdim=True) == 0)
        mask_value = (value.sum(dim=-1, keepdim=True) == 0)

        # Linear projections in batch from d_model => h x d_k
        query = self.query_projection(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.key_projection(key)
        key = torch.where(mask_key, torch.zeros_like(key), key)
        key = key.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.value_projection(value)
        value = torch.where(mask_value, torch.zeros_like(value), value)
        value = value.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        attn = torch.matmul(p_attn, value)

        # Concatenate heads and put through final linear layer
        if seq_len == 1:
            attn = attn.transpose(1, 2).contiguous().view(batch_size, self.n_heads * self.d_k)
            attn = attn.unsqueeze(1)  # Add the sequence length dimension back
        else:
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_k)
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
    def __init__(self, user_embed_dim, token_dim, n_heads, ffn_dim, dropout, att_dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, token_dim, dropout=att_dropout)
        self.cross_attn = MultiHeadAttentionSocial(n_heads, token_dim, user_embed_dim, dropout=att_dropout)
        self.cross_attn_social = MultiHeadAttentionSocial(n_heads, token_dim,user_embed_dim , dropout=att_dropout)
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm_cross = nn.LayerNorm(token_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, token_dim)
        )
        self.norm3 = nn.LayerNorm(token_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, conv_embs, social_embs, tgt_mask, memory_mask):
        # combined_embeds = torch.cat((conv_embs, social_embs), dim=1)
        # combined_embeds = torch.nn.ReLU()(combined_embeds)
        # combined_embeds = self.output_projection(combined_embeds)
        # Self-attention on the target sequence
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # # Cross-attention with the encoder's output (memory)
        # tgt2 = self.cross_attn_social(tgt, social_embs, social_embs, mask=memory_mask)
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm_cross(tgt)

        # # Cross-attention with the encoder's output (memory)
        # tgt2 = self.cross_attn(tgt, conv_embs, conv_embs, mask=memory_mask)
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm_cross(tgt)



        # Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, user_embed_dim,token_dim, n_heads, ffn_dim, dropout, voc_size, att_dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(user_embed_dim,token_dim, n_heads, ffn_dim, dropout, att_dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(token_dim)
        
   
   
      

    def forward(self, tgt,conv_embs, social_embs,social_reps, tgt_mask, memory_mask):
        for layer in self.layers:
            tgt = layer(tgt, conv_embs, social_embs ,tgt_mask, memory_mask)
        tgt = self.norm(tgt)
        

      
        return tgt     


class ConversationTransformerOLD(nn.Module):
    def __init__(self, embed_dim, user_emb, 
                 voc_size, n_heads, ffn_dim, dropout, attention_dropout, n_layers, device, pad_inx,
                 embedding,  embeddings_scale=True, 
                 learn_positional_embeddings = True, n_positions= 1024):
        super().__init__()
        self.user_embed_dim = user_emb
        self.voc_size = voc_size
        self.dropout = dropout
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.n_layers = n_layers
        self.device = device
        self.embedding_size = embed_dim
        self.ffn_size = ffn_dim
     

        self.dim = embed_dim
        self.embeddings_scale = embeddings_scale
        self.dropout1 = nn.Dropout(dropout)  # --dropout
        self.out_dim = embed_dim
        assert embed_dim % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = self.decoder_token_embedding = nn.Embedding.from_pretrained(
                embedding.weight,
                freeze=False,
                padding_idx=pad_inx)
        self.position_embeddings = self._init_postision_embeddings(n_positions, embed_dim, learn_positional_embeddings)

       
        # Transformer Encoder
        # self.encoder = TransformerEncoder(
        #     n_layers=self.n_layers,
        #     embed_dim=self.embed_dim,
        #     n_heads=self.n_heads,
        #     ffn_dim=self.ffn_dim,
        #     dropout=self.dropout,
        #     att_dropout= attention_dropout
        # )

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            n_layers=self.n_layers,
            user_embed_dim=self.user_embed_dim,
            token_dim=self.embedding_size,
            n_heads=self.n_heads,
            ffn_dim=self.ffn_size,
            dropout=self.dropout,
            voc_size=self.voc_size,
            att_dropout= attention_dropout
        )
        
    def _init_postision_embeddings(self, n_positions, embedding_size, learn_positional_embeddings):
        # create the positional embeddings
        position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            self.create_position_codes(
                n_positions, embedding_size, out=position_embeddings.weight
            )
        else:
            nn.init.normal_(position_embeddings.weight, 0, embedding_size ** -0.5)

        return position_embeddings
    
    def create_position_codes(self, n_pos, dim, out):
        position_enc = np.array([
            [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
            for pos in range(n_pos)
        ])
        out.detach_()
        out.requires_grad = False
        out[:, 0::2] = torch.tensor(np.sin(position_enc)).type_as(out)
        out[:, 1::2] = torch.tensor(np.cos(position_enc)).type_as(out)


    def embed_input(self, input):
        tensor = self.embeddings(input)  # (bs, seq_len, embed_dim)

        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        
        positions_embedding = self.get_postition_embeddings(input, tensor)
        tensor = tensor + positions_embedding
        
        tensor = self.dropout1(tensor)  # --dropout

        return tensor
    
    def get_postition_embeddings(self, input, tensor):
        seq_len = input.size(1)
        positions = input.new(seq_len).long()  # (seq_len)
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)  # (1, seq_len)
        positions_embedding = self.position_embeddings(positions).expand_as(tensor)

        return positions_embedding  

    def forward(self, inputx,encoder_embeddings,social_embeddings, social_reps, mask):
        inputs = self.embed_input(inputx)  # (bs, seq_len2, dim)

        # Pass through the decoder
        decoder_output = self.decoder(inputs, encoder_embeddings, social_embeddings, social_reps,None, None)  # Assuming no masks for simplicity

        return decoder_output                    