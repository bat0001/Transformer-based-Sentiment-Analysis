import torch
from src.transformers import Transformer

def save_model(model, optimizer, path="model_checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    
def load_model(vocab_size, max_seq_length, embedding_dim, path="model_checkpoint.pth"):
    checkpoint = torch.load(path)
    
    model = Transformer(vocab_size, max_seq_length, embedding_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return model, optimizer
