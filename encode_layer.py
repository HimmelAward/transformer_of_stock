import torch
import numpy as np
import torch.nn as nn

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q ,_= seq_q.size()
    batch_size, len_k ,_= seq_k.size()
    pad_attn_mask = torch.ones(batch_size,len_q,len_k)
    return pad_attn_mask.cuda()

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_len)
        ])
        pos_table[1:,0::2] = np.sin(pos_table[1:, 0::2])
        pos_table[1:,1::2] = np.cos(pos_table[1:, 1::2])
        self.pos_table = torch.FloatTensor(pos_table).cuda()

    def forward(self, enc_inputs):
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)

class Attention(nn.Module):
    def __init__(self,d_k):
        super(Attention, self).__init__()
        self.attn = nn.Softmax(dim=-1)
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask.byte(), -1e9)
        attn = self.attn(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,d_head,n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.attention = Attention(d_k=d_head)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_head).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,1)

        context, attn = self.attention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,self.n_heads * self.d_head)
        output = self.fc(context)

        return self.norm(output + residual), attn


class FeedForward(nn.Module):
    def __init__(self,d_model,d_ffn):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model, bias=False))
        self.norm = nn.LayerNorm(d_model)


    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_head,n_heads,d_ffn):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model,d_head,n_heads)
        self.pos_ffn = FeedForward(d_model,d_ffn)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn