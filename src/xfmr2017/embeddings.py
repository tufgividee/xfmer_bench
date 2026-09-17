import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # position canvus of max seq length and embeddintg size (d_model)

        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # adding dim at idx1

        div_term = torch.exp(torch.arange(0, d_model, 2).float() \
                             * (-torch.log(torch.tensor(10000))) / d_model )
        # x = exp(ln(x)), 1/x^y = x^-y

        # use of the same div_term saves multiply
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        # Now, `pe` will have shape (max_len, d_model).
        # We unsqueeze it to (1, max_len, d_model) for broadcasting with (batch_size, sequence_length, d_model)
        pe = pe.unsqueeze(0) # adds a dimension at the beginning
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to have shape (batch_size, sequence_length, d_model)
        # self.pe has shape (1, max_len, d_model)
        # We slice self.pe along the sequence_length dimension
        return x + self.pe[:, :x.size(1), :]