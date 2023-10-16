import torch

def get_positional_encoding(max_seq_length, embedding_dim):
    """
    Calcule le positional encoding pour une dimension d'embedding et une longueur de séquence donnée.
    """
    pos_enc = torch.zeros((max_seq_length, embedding_dim))
    
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                         (-torch.log(torch.tensor(10000.0)) / embedding_dim))
    
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
                
    return pos_enc
