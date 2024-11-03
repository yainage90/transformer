import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_block) for _ in range(n_layer)
        ])
            
    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        return out

        
class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff, d_model):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = nn.ModuleList([
            ResidualConnectionLayer(d_model=d_model) for _ in range(2)
        ])

    def forward(self, src, src_mask):
        out = src
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residuals[1](out, self.position_ff)
        return out


class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(n_layer)])

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, position_ff, d_model):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residuals = nn.ModuleList([
            ResidualConnectionLayer(d_model=d_model) for _ in range(3)
        ])

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residuals[1](out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)
        return out

        
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.out_fc = out_fc
        self.dropout = nn.Dropout(p=dropout)

    def calculate_attention(self, query, key, value, mask):
        # query, key, value: (n_batch, h, seq_len, d_k)
        # mask: (n_batch, 1, seq_len, seq_len)
        d_k = key.size(-1)
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # (n_batch, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        p_attn = F.softmax(attention_score, dim=-1)
        # p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value) # (n_batch, h, seq_len, d_k)
        return out

    def forward(self, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, seq_len, d_embed)
        n_batch = query.size(0)

        def transform(x, fc):
            # x: (n_batch, seq_len, d_embed)
            out = fc(x) # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model // self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query, self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc) # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        return out



class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc1, fc2, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1 # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = fc2 # (d_ff, d_embed)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

        
class ResidualConnectionLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sub_layer):
        out = x
        out = self.dropout(sub_layer(self.norm(out)))
        out = out + x
        return out


class TransformerEmbedding(nn.Module):
    def __init__(self, token_embed, pos_embed=None):
        super(TransformerEmbedding, self).__init__()
        embeddings = [token_embed]
        if pos_embed is not None:
            embeddings.append(pos_embed)
        self.embedding = nn.Sequential(*embeddings)

    def forward(self, x):
        out = self.embedding(x)
        return out


class TokenEmbedding(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out

        
class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_len=256, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[: , :seq_len, :].to(x.device)
        out = x + pos_embed
        out = self.dropout(out)
        return out


class PositionEmbedding(nn.Module):
    def __init__(self, d_embed, max_len=256, dropout=0.1):
        super(PositionEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_embed)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_embed, eps=1e-6)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.int64, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)
        out = self.embedding(position_ids)
        out = self.layer_norm(self.dropout(out))
        return out