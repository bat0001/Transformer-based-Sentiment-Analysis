import numpy as np
import torch
import torch.nn as nn
import os

from src.preprocessing import load_data, preprocess_data, split_data, prepare_data, convert_to_tensors, create_dataloader
from src.transformers import Transformer
from src.utils import save_model, load_model
from src.evaluate import evaluate
from src.training import training

def main():
    print("[INFO] Starting data preparation...")
    X_train, X_test, y_train, y_test, tokenizer = prepare_data('./data/twitter_training.csv')
    print(f"[INFO] Loaded {len(X_train)} training samples and {len(X_test)} testing samples.")

    vocab_size = len(tokenizer.lex_to_index) + 1
    embedding_dim = 32
    print(f"[INFO] Vocabulary size: {vocab_size}")
    print(f"[INFO] Embedding dimension: {embedding_dim}")

    print("[INFO] Converting training and testing data to tensors...")
    X_train_tensor, y_train_tensor = convert_to_tensors(X_train, y_train)
    X_test_tensor, y_test_tensor = convert_to_tensors(X_test, y_test)
    
    batch_size = 128 
    print(f"[INFO] Creating dataloaders with batch size: {batch_size}")
    train_dataloader = create_dataloader(X_train_tensor, y_train_tensor, batch_size)
    test_dataloader = create_dataloader(X_test_tensor, y_test_tensor, batch_size)
    
    loss_function = nn.CrossEntropyLoss()
    print("[INFO] Initialized CrossEntropy loss function.")
    
    model_path = "my_model.pth"

    if os.path.exists(model_path):
        print("[INFO] Loading existing model from disk...")
        max_seq_length = train_dataloader.dataset.tensors[0].size(1)
        model, optimizer = load_model(vocab_size, max_seq_length, embedding_dim,model_path)
    else:
        print("[INFO] Training model from scratch...")
        model = training(train_dataloader, vocab_size, embedding_dim, loss_function)
        print("[INFO] Model training completed.")

    print("[INFO] Evaluating model on test data...")
    test_loss, test_accuracy = evaluate(model, test_dataloader, loss_function, tokenizer)

    print(f"[RESULT] Test Loss: {test_loss:.4f}")
    print(f"[RESULT] Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
