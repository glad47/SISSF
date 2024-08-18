


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
# class MultiHeadAttention(nn.Module):
#     def __init__(self, n_heads, dim, dropout=0):
#         super(MultiHeadAttention, self).__init__()
#         self.n_heads = n_heads
#         self.dim = dim

#         self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
#         self.q_lin = nn.Linear(dim, dim)
#         self.k_lin = nn.Linear(dim, dim)
#         self.v_lin = nn.Linear(dim, dim)
#         # TODO: merge for the initialization step
#         nn.init.xavier_normal_(self.q_lin.weight)
#         nn.init.xavier_normal_(self.k_lin.weight)
#         nn.init.xavier_normal_(self.v_lin.weight)
#         # and set biases to 0
#         self.out_lin = nn.Linear(dim, dim)

#         nn.init.xavier_normal_(self.out_lin.weight)

#     def forward(self, query, key=None, value=None, mask=None):
#         # Input is [B, query_len, dim]
#         # Mask is [B, key_len] (selfattn) or [B, key_len, key_len] (enc attn)
#         batch_size, query_len, dim = query.size()
#         assert dim == self.dim, \
#             f'Dimensions do not match: {dim} query vs {self.dim} configured'
#         assert mask is not None, 'Mask is None, please specify a mask'
#         n_heads = self.n_heads
#         dim_per_head = dim // n_heads
#         scale = math.sqrt(dim_per_head)

#         def prepare_head(tensor):
#             # input is [batch_size, seq_len, n_heads * dim_per_head]
#             # output is [batch_size * n_heads, seq_len, dim_per_head]
#             bsz, seq_len, _ = tensor.size()
#             tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
#             tensor = tensor.transpose(1, 2).contiguous().view(
#                 batch_size * n_heads,
#                 seq_len,
#                 dim_per_head
#             )
#             return tensor

#         # q, k, v are the transformed values
#         if key is None and value is None:
#             # self attention
#             key = value = query
#         elif value is None:
#             # key and value are the same, but query differs
#             # self attention
#             value = key
#         _, key_len, dim = key.size()

#         q = prepare_head(self.q_lin(query))
#         k = prepare_head(self.k_lin(key))
#         v = prepare_head(self.v_lin(value))

#         dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
#         # [B * n_heads, query_len, key_len]
#         attn_mask = (
#             (mask == 0)
#             .view(batch_size, 1, -1, key_len)
#             .repeat(1, n_heads, 1, 1)
#             .expand(batch_size, n_heads, query_len, key_len)
#             .view(batch_size * n_heads, query_len, key_len)
#         )
#         assert attn_mask.shape == dot_prod.shape
#         dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

#         attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
#         attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

#         attentioned = attn_weights.bmm(v)
#         attentioned = (
#             attentioned.type_as(query)
#             .view(batch_size, n_heads, query_len, dim_per_head)
#             .transpose(1, 2).contiguous()
#             .view(batch_size, query_len, dim)
#         )

#         out = self.out_lin(attentioned)

#         return out


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
def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)    

class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, relu_dropout=0):
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 2048, kernel_size=3, padding=1),  # Added layer
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(2048, 1024, kernel_size=3, padding=1),  # Adjusted input channels
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
        self.unet = UNet(embedding_size, embedding_size)
        # self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, ffn_size),
            nn.ReLU(),
            nn.Dropout(relu_dropout),
            nn.Linear(ffn_size, embedding_size)
        )
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask, social_embeddings,conv_user_rep):
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(x, x, x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm1)
        
        residual = x
        x = self.encoder_db_attention(
            query=x,
            key=conv_user_rep,
            value=conv_user_rep
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2_db)

        residual = x
        x = self.encoder_kg_attention(
            query=x,
            key=social_embeddings,
            value=social_embeddings
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2_kg)
        encoder_mask = self._create_crossattn_mask(x,encoder_output, encoder_mask )
        residual = x
        x = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2)
        
        # finally the ffn
        # residual = x
        # x = x.permute(0, 2, 1)  # (batch_size, embedding_size, seq_len)
        # x = self.unet(x)
        # x = x.permute(0, 2, 1) 
        
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).unsqueeze(1).expand(bsz, self.n_heads, -1, -1)
        return mask

    def _create_crossattn_mask(self, query, key, existing_mask):
        bsz = query.size(0)
        seq_q = query.size(1)
        seq_k = key.size(1)
        
        # Expand the existing mask to match the dimensions (bsz, 1, seq_q, seq_k)
        mask = existing_mask.unsqueeze(1).unsqueeze(2).expand(bsz, self.n_heads, seq_q, seq_k)
        
        return mask    

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
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
        inputs = self.embed_input(inputx)  # (bs, seq_len2, dim)
        
        # Pass through the decoder
        decoder_output = self.decoder(inputs, encoder_output, encoder_mask, social_embeddings,conv_user_rep)  # Assuming no masks for simplicity

        return decoder_output                    