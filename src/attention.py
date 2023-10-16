import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1))
    dimension_keys = torch.tensor(key.size(-1)).float()
    scaled_attention_scores = scores / torch.sqrt(dimension_keys)
    softmax_scores = F.softmax(scaled_attention_scores, dim=-1)    
    attention_output = torch.matmul(softmax_scores, value)
    return attention_output

def multihead_attention(query, key, value, num_heads):
    batch_size, seq_length, embedding_dim = query.size()
    
    if embedding_dim % num_heads != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) should be divisible by num_heads ({num_heads})")

    dim_per_head = embedding_dim // num_heads
    
    query_transform = nn.Linear(embedding_dim, embedding_dim)
    key_transform = nn.Linear(embedding_dim, embedding_dim)
    value_transform = nn.Linear(embedding_dim, embedding_dim)
    output_transform = nn.Linear(embedding_dim, embedding_dim)

    query = query_transform(query)
    key = key_transform(key)
    value = value_transform(value)
    
    query = query.view(batch_size, seq_length, num_heads, dim_per_head)
    key = key.view(batch_size, seq_length, num_heads, dim_per_head)
    value = value.view(batch_size, seq_length, num_heads, dim_per_head)

    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    
    heads = []
    for i in range(num_heads):
        head_i = scaled_dot_product(query[:, i], key[:, i], value[:, i])
        heads.append(head_i)
    
    concat_heads = torch.cat(heads, dim=-1)
    concat_heads = concat_heads.view(-1, embedding_dim)
    output = output_transform(concat_heads)
    output = output.view(batch_size, seq_length, embedding_dim)

    return output
