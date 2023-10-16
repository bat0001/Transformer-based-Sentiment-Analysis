import torch.nn as nn
import torch

class LayerNormalization(nn.Module):
    
    def __init__(self, embedding_dim, epsilon=1e-5):
        super(LayerNormalization, self).__init__() 
        
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.epsilon = epsilon

    def forward(self, x):  
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        
        x_norm = (x - mean) / torch.sqrt(var + self.epsilon)
        out = x_norm * self.gamma + self.beta
        return out
