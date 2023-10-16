import os
import torch
import torch.nn as nn

from src.attention import scaled_dot_product, multihead_attention
from src.normalization import LayerNormalization
from src.feed_forward import FeedForward

class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(Encoder, self).__init__()

        self.W_q = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.W_q)

        self.W_k = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.W_k)

        self.W_v = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.W_v)

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        self.norm1 = LayerNormalization(embedding_dim)
        self.norm2 = LayerNormalization(embedding_dim)
        
        self.ffn = FeedForward(embedding_dim, 2048)
    
    def forward(self, x):

        query = torch.bmm(x, self.W_q.unsqueeze(0).expand(x.size(0), -1, -1))
        key = torch.bmm(x, self.W_k.unsqueeze(0).expand(x.size(0), -1, -1))
        value = torch.bmm(x, self.W_v.unsqueeze(0).expand(x.size(0), -1, -1))
        
        attention_output = multihead_attention(query, key, value, self.num_heads)
        
        output1 = self.norm1.forward(query + attention_output)
        
        ffn_output = self.ffn(output1)
        
        output2 = self.norm2.forward(torch.cat([output1, ffn_output], dim=1))
        
        return output2