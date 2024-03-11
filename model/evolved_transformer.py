import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
import math

class TokenEmbedding(nn.Module):
    ''' helper Module to convert tensor of input indices into corresponding tensor of token embeddings'''
    
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class PositionalEncoding(nn.Module):
    ''' helper Module that adds positional encoding to the token embedding to introduce a notion of word order.'''
    
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

    
class LinearPointEmbedder(nn.Module):
    def __init__(self, vocab_size: int, input_emb_size, emb_size, max_input_points):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_emb_size)
        self.emb_size = emb_size
        self.input_size = max_input_points*input_emb_size
        self.fc1 = nn.Linear(self.input_size, emb_size)
        self.fc2 = nn.Linear(emb_size, emb_size)
        self.activation = nn.ReLU()

    def forward(self, tokens):
        out = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        print(out.shape)
        bs, n = out.shape[0], out.shape[1]
        out = out.view(bs, n, -1)
        out = self.activation(self.fc1(out))
        out = self.fc2(out)
        return out
    
