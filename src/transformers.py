import torch.nn as nn
import torch

from src.encoder import Encoder
from src.positional_encoding import get_positional_encoding

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_length, embedding_dim, num_classes=4, num_heads=8, N=1):
        super(Transformer, self).__init__()

        # Embedding layer
        self.encoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        pos_encoding = get_positional_encoding(max_seq_length, embedding_dim)
        self.encoder_positional_encoding = pos_encoding.clone().detach().requires_grad_(False)
        
        # Stacked encoders
        self.encoders = nn.ModuleList([Encoder(embedding_dim, num_heads) for _ in range(N)])

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_indices):
        # Convert token indices to embeddings
        input_embeddings = self.encoder_embedding(input_indices)

        # Add positional encoding to embeddings
        input_with_pos = input_embeddings + self.encoder_positional_encoding

        # Pass through each encoder in the stack
        for encoder in self.encoders:
            input_with_pos = encoder(input_with_pos)  # Nous passons juste les embeddings

        pooled_output = torch.mean(input_with_pos, dim=1)
        logits = self.classifier(pooled_output)

        return logits
