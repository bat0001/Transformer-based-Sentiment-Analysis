import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_size):
        super(FeedForward, self).__init__()
        
        self.layer1 = nn.Linear(embed_size, ff_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(ff_size, embed_size)
        
    def forward(self, x):
        z1 = self.layer1(x)
        a1 = self.relu(z1)
        a2 = self.layer2(a1)
        return a2
