import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.norm1(tgt)
        q = k = tgt2

        tgt2, _ = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        # 调整 memory 的形状以匹配 tgt
        memory = memory.repeat(1, tgt.size(1), 1)
        tgt2, _ = self.cross_attn(tgt2, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=120+2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab, vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.layers = nn.ModuleList([TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, captions, memory):
        captions = self.embed(captions) * math.sqrt(self.embed.embedding_dim)
        captions = self.pos_encoder(captions)

        for layer in self.layers:
            captions = layer(captions, memory)

        captions = self.norm(captions)
        output = self.fc_out(captions)
        return output
