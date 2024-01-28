import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
import torch.nn.functional as F


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int=120):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(1, max_seq_len, d_model)

        self.register_buffer('pe', pe, False)

    def forward(self, x: torch.Tensor):
        n, seq_len, d_model = x.shape
        pe: torch.Tensor = self.pe
        assert seq_len <= pe.shape[1]
        assert d_model == pe.shape[2]
        rescaled_x = x * d_model**0.5
        return rescaled_x + pe[:, 0:seq_len, :]

def attention(query, key, value, mask=None):
    '''The dtype of mask must be bool
    query shape: [n, heads, q_len, d_k]
    key shape: [n, heads, k_len, d_k]
    value shape: [n, heads, k_len, d_v]
    '''
    MY_INF = 1e12

    assert query.shape[-1] == key.shape[-1]
    d_k = key.shape[-1]
    # tmp shape: [n, heads, q_len, k_len]
    tmp = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        tmp = tmp.masked_fill(mask, float(-MY_INF))
    tmp = F.softmax(tmp, dim=-1)
    # now tmp shape: [n, heads, q_len, d_v]
    return torch.matmul(tmp, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()

        assert d_model % n_head == 0
        # d_k = d_v = d_model // n_head
        self.d_k = self.d_v = d_model // n_head
        self.n_head = n_head
        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        assert q.shape[0] == k.shape[0] == v.shape[0]  # batch should be the same
        assert k.shape[1] == v.shape[1]  # the sequence length of k and v should be the same
        
        n, q_len = q.shape[:2]
        k_len = k.shape[1]
        q_ = self.Q(q).reshape(n, q_len, self.n_head, self.d_k).transpose(1, 2)
        k_ = self.K(k).reshape(n, k_len, self.n_head, self.d_k).transpose(1, 2)
        v_ = self.V(v).reshape(n, k_len, self.n_head, self.d_v).transpose(1, 2)

        attention_res = attention(q_, k_, v_, mask)
        concat_res = attention_res.transpose(1, 2).reshape(n, q_len, self.d_model)
        concat_res = self.dropout(concat_res)

        return self.out(concat_res)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = x + self.dropout1(self.attn(x, x, x, src_mask))
        x = self.ln1(x)
        x = x + self.dropout2(self.ff(x))
        x = self.ln2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.attn1 = MultiHeadAttention(n_head, d_model, dropout)
        self.attn2 = MultiHeadAttention(n_head, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, src_tgt_mask):
        x = x + self.dropout1(self.attn1(x, x, x, tgt_mask))
        x = self.ln1(x)
        x = x + self.dropout2(self.attn2(x, memory, memory, src_tgt_mask))
        x = self.ln2(x)
        x = x + self.dropout3(self.ff(x))
        x = self.ln3(x)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, n_layer, n_head, d_model, d_ff, dropout=0.1, max_seq_len=120):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(n_head, d_model, d_ff, dropout) for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = self.embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, n_layer, n_head, d_model, d_ff, dropout=0.1, max_seq_len=120):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(n_head, d_model, d_ff, dropout) for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_tgt_mask=None):
        x = self.embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_tgt_mask)
        return x

# Seq2Seq Network

