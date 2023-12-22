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

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        
        # print("tgt shape:", tgt.shape)
        # print("memory shape:", memory.shape)

        # 确保 key_padding_mask 和 attn_mask 类型一致
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.bool()
        
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.bool()

        if tgt_mask is not None:
            tgt_mask = tgt_mask.bool()

        if memory_mask is not None:
            memory_mask = memory_mask.bool()

        memory = memory.repeat(1, tgt.size(1), 1)

        tgt2 = self.norm1(tgt)
        q = k = tgt2

        # print("q, k shape:", q.shape)

        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt2, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=120):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab,vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.layers = nn.ModuleList([TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, captions, memory):
        # 确保 captions 是二维的，形状为 [batch_size, seq_len]
        # 假设 <pad> 词在 vocab 中的索引是 pad_token_id
        pad_token_id = self.vocab['<pad>']  # 根据您的 vocab 获取 <pad> 词的索引
        tgt_padding_mask = (captions == pad_token_id).transpose(0, 1)

        # Embedding 和 Positional Encoding
        captions = self.embed(captions) * math.sqrt(self.embed.embedding_dim)
        captions = self.pos_encoder(captions)

        # 生成序列掩码
        tgt_mask = self.generate_square_subsequent_mask(captions.size(0)).to(captions.device)

        memory_key_padding_mask = None  

        # 应用 Transformer 层
        for layer in self.layers:
            captions = layer(captions, memory, tgt_mask=tgt_mask, memory_mask=None,
                             tgt_key_padding_mask=tgt_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)

        captions = self.norm(captions)
        output = self.fc_out(captions)
        return output

