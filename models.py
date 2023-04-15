import torch.nn as nn
import torch
import encode_layer

class Transformer(nn.Module):
    def __init__(self,d_model,d_head,n_heads,d_ffn,feature,layers):
        super(Transformer, self).__init__()
        self.src_emb = nn.Linear(feature, d_model)
        self.pos_emb = encode_layer.PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            encode_layer.EncoderLayer(d_model,d_head,n_heads,d_ffn)
            for _ in range(layers)
        ])
        self.linear = nn.Linear(d_model, 1, bias=False)

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs)
        enc_self_attn_mask = encode_layer.get_attn_pad_mask(enc_inputs, enc_inputs)

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)                                                              # enc_self_attn : [batch_size, n_heads, src_len, src_len]
        output = self.linear(enc_outputs)

        return output