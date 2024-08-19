## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14


import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def neginf(dtype):
    """Returns a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF
def create_position_codes(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
        for pos in range(n_pos)
    ])

    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)
    out.detach_()
    out.requires_grad = False
class MultiHeadAttentionUser(nn.Module):
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
        query = self.query_projection(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #(bs, n_heads, seq_length, d_k)
        key = self.key_projection(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  #(bs, n_heads, 1, d_k)
        value = self.value_projection(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #(bs, n_heads, 1, d_k)

        # Apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k) #(bs, n_heads, seq_length, 1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)  #(bs, n_heads, seq_length, 1)
        p_attn = self.dropout(p_attn)  #(bs, n_heads, seq_length, 1)
        attn = torch.matmul(p_attn, value)  #(bs, n_heads, 1, d_k)

        # Concatenate heads and put through final linear layer
        if seq_len == 1:
            attn = attn.transpose(1, 2).contiguous().view(batch_size, self.n_heads * self.d_k) #(bs, dim)
            attn = attn.unsqueeze(1)  # Add the sequence length dimension back   #(bs, 1, dim)
        else:
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_k) #(bs, seq_length, dim)
        output = self.out_projection(attn)  #(bs, seq_length, dim),  #(bs, 1, dim)

        return output



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

        
        query = self.query_projection(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #(bs, n_heads, seq_length, d_k)
        key = self.key_projection(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #(bs, n_heads, seq_length, d_k)
        value = self.value_projection(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #(bs, n_heads, seq_length, d_k)

        # Apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k) #(bs, n_heads, seq_length, seq_length)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  #(bs, n_heads, seq_length, seq_length)
        p_attn = F.softmax(scores, dim=-1)   #(bs, n_heads, seq_length, seq_length)
        p_attn = self.dropout(p_attn)  #(bs, n_heads, seq_length, seq_length)
        attn = torch.matmul(p_attn, value) #(bs, n_heads, seq_length, d_k)

        # Concatenate heads and put through final linear layer
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k) #(bs, seq_length, dim)
        output = self.out_projection(attn) #(bs, seq_length, dim)

        return output
def _normalize(tensor, norm_layer):
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)    




class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.n_heads = n_heads
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size)

        self.encoder_db_attention = MultiHeadAttentionUser(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2_db = nn.LayerNorm(embedding_size)

        self.encoder_kg_attention = MultiHeadAttentionUser(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2_kg = nn.LayerNorm(embedding_size)
  
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, ffn_size),
            nn.ReLU(),
            nn.Dropout(relu_dropout),
            nn.Linear(ffn_size, embedding_size)
        )
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask, social_embeddings,conv_user_rep):  
        #(bs, seq_length, dim), (bs, context_length, dim), (bs, context_length), (bs, dim), (bs, dim)

        decoder_mask = self._create_selfattn_mask(x)  #(bs, n_heads, seq_length, seq_length)
        # first self attn
        residual = x    #(bs, seq_length, dim)
        # don't peak into the future!
        x = self.self_attention(x, x, x, mask=decoder_mask) #(bs, seq_length, dim)
        x = self.dropout(x)             #(bs, seq_length, dim)
        x = x + residual                #(bs, seq_length, dim)
        x = _normalize(x, self.norm1)   #(bs, seq_length, dim)
         
        residual = x                    #(bs, seq_length, dim)
        x = self.encoder_db_attention(
            query=x,
            key=conv_user_rep,
            value=conv_user_rep
        )                                   #(bs, seq_length, dim)
        x = self.dropout(x)                 #(bs, seq_length, dim)
        x = residual + x                    #(bs, seq_length, dim)
        x = _normalize(x, self.norm2_db)    #(bs, seq_length, dim)

        residual = x                        #(bs, seq_length, dim)
        x = self.encoder_kg_attention(
            query=x,
            key=social_embeddings,
            value=social_embeddings
        )                                   #(bs, seq_length, dim)
        x = self.dropout(x)                 #(bs, seq_length, dim)
        x = residual + x                    #(bs, seq_length, dim)
        x = _normalize(x, self.norm2_kg)    #(bs, seq_length, dim)
        encoder_mask = self._create_crossattn_mask(x,encoder_output, encoder_mask )  #(bs, n_heads, seq_length, context_length)
        residual = x                        #(bs, seq_length, dim)
        x = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )                                   #(bs, seq_length, dim)
        x = self.dropout(x)                 #(bs, seq_length, dim)
        x = residual + x
        x = _normalize(x, self.norm2)       #(bs, seq_length, dim)
        
       
        
        x = self.ffn(x)         #(bs, seq_length, dim)
        x = self.dropout(x)     #(bs, seq_length, dim)
        x = residual + x        #(bs, seq_length, dim)
        x = _normalize(x, self.norm3)     #(bs, seq_length, dim)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).unsqueeze(1).expand(bsz, self.n_heads, -1, -1) #(bs, n_heads, time, time)
        return mask

    def _create_crossattn_mask(self, query, key, existing_mask):
        bsz = query.size(0)
        seq_q = query.size(1)
        seq_k = key.size(1)
        
        mask = existing_mask.unsqueeze(1).unsqueeze(2).expand(bsz, self.n_heads, seq_q, seq_k) #(bs, n_heads, seq_q, seq_k)
        
        return mask    

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param int vocabulary_size: the number of all tokens in the dataset 
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
   
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        n_positions=1024,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerDecoderLayer(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, input, encoder_output, encoder_mask, social_embeddings,conv_user_rep, incr_state=None):
        for layer in self.layers:
            tensor = layer(input, encoder_output, encoder_mask, social_embeddings,conv_user_rep)

        return tensor, None

class ConversationTransformer(nn.Module):
    def __init__(self, embed_dim, user_emb, 
                 voc_size, n_heads, ffn_dim, dropout, attention_dropout,relu_dropout, n_layers, device, pad_inx,
                 embedding,  embeddings_scale=True, 
                 learn_positional_embeddings = True, n_positions= 1024):
        super().__init__()
        self.user_embed_dim = user_emb
        self.voc_size = voc_size
        self.dropout = dropout
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.relu_dropout= relu_dropout
        self.n_layers = n_layers
        self.device = device
        self.embedding_size = embed_dim
        self.ffn_size = ffn_dim
     

        self.dim = embed_dim
        self.embeddings_scale = embeddings_scale
        self.dropout1 = nn.Dropout(dropout)  # --dropout
        self.out_dim = embed_dim
        self.n_positions =n_positions
        assert embed_dim % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = self.decoder_token_embedding = nn.Embedding.from_pretrained(
                embedding.weight,
                freeze=False,
                padding_idx=pad_inx)
        self.position_embeddings = self._init_postision_embeddings(n_positions, embed_dim, learn_positional_embeddings)
        # self.lin1 = nn.Linear(300, self.ffn_size)
       
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
        n_heads=self.n_heads,
        n_layers=self.n_layers,
        embedding_size=self.embedding_size,
        ffn_size=self.ffn_size,
        vocabulary_size=self.voc_size,
        embedding=self.embeddings,
        dropout=self.dropout,
        attention_dropout=self.attention_dropout,
        relu_dropout=self.relu_dropout,
        learn_positional_embeddings=self.position_embeddings,
        embeddings_scale=self.embeddings_scale,
        n_positions=self.n_positions)
    
        
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

    def forward(self, inputx,encoder_output,encoder_mask, conv_user_rep,social_embeddings):
        #(bs, seq_length), (bs, context_length, dim ), (bs, context_length), (bs, dim), (bs, dim)


        inputs = self.embed_input(inputx)  # (bs, seq_len, dim)
        
        # Pass through the decoder
        decoder_output = self.decoder(inputs, encoder_output, encoder_mask, social_embeddings,conv_user_rep)  # Assuming no masks for simplicity

        return decoder_output    # (bs, seq_len, dim)