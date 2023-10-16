import torch

from src.transformers import Transformer

def training(dataloader, vocab_size, embedding_dim, loss_function, epochs=1, patience=5):
    
    max_seq_length = dataloader.dataset.tensors[0].size(1)
    model = Transformer(vocab_size, max_seq_length, embedding_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # early stopping
    best_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(epochs):
        epoch_losses = []

        for batch_X, batch_y in dataloader:
            # Forward Pass
            predictions = model(batch_X)
            
            batch_y = batch_y.view(-1).long()
            # Calculez la perte
            loss = loss_function(predictions, batch_y)
            epoch_losses.append(loss.item())
            
            # Zero-out gradients
            optimizer.zero_grad()
            
            # Backpropagation
            loss.backward()
            
            # Update weights
            optimizer.step()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs == patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break
    
    save_model(model, optimizer, "my_model.pth")
    return model

