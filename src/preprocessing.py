import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.tokenizer import SpacyTokenizer

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df.columns = ["idk", "key", "sentiment", "text"]
    return df['text'].tolist(), df['sentiment'].tolist()

def preprocess_data(text_data, tokenizer):
    tokenized_text_data = tokenizer.tokenize_data(text_data)
    token_ids_text_data = [tokenizer.to_ids(sentence) for sentence in tokenized_text_data]
    token_ids_text_data = pad_sequence([torch.tensor(seq) for seq in token_ids_text_data], batch_first=True).tolist()

    return token_ids_text_data

def split_data(text_data, sentiment_data, test_size=0.2):
    return train_test_split(text_data, sentiment_data, test_size=test_size, random_state=42)

def prepare_data(path):
    text_data, sentiment_data = load_data(path)

    label_to_index = {
        "Negative": 0,
        "Positive": 1,
        "Neutral": 2,
        "Irrelevant": 3
    }
    
    sentiment_data = [label_to_index[label] for label in sentiment_data]

    tokenizer = SpacyTokenizer()
    text_data = preprocess_data(text_data, tokenizer)
    X_train, X_test, y_train, y_test = split_data(text_data, sentiment_data)
    
    return X_train, X_test, y_train, y_test, tokenizer

def convert_to_tensors(X_train, y_train):
    X_train_tensor = torch.LongTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    return X_train_tensor, y_train_tensor

def create_dataloader(X_train_tensor, y_train_tensor, batch_size):
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
