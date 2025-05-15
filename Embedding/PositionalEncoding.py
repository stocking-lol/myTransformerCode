import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding Module
    """
    def __init__(self, input_n, input_d):
        super(PositionalEncoding, self).__init__()
        self.P = torch.zeros((1, input_n, input_d))
        
        # Pi,2j = sin(i/10000^(2j)/d) Pi,2j+1=sin(i/10000^(2j+1)/d)
        i = torch.arange(input_n, dtype=torch.float).reshape((-1, 1))
        j = torch.arange(0, input_d, dtype=torch.float)
        x = i / torch.pow(10000, j)  # 广播
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        x = x + self.P
        return x
