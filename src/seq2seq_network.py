import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
import torch.nn.functional as F


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
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

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

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

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
