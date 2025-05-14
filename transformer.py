import math
import torch
from torch import nn
import pandas as pd
from multi_head import MultiHeadAttention

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_output,**kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.densel1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.densel2 = nn.Linear(ffn_num_hiddens, ffn_num_output)

    def forward(self, x):
        return self.densel2(self.relu(self.densel1(x)))

'''ffn = PositionWiseFFN(4,4,8)
ffn.eval()
ffn(torch.ones((2,3,4)))[0]'''

'''ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = torch.tensor([[1 , 2],[2 , 3]], dtype=torch.float32)
print('layer norm:',ln(X),'batch norm:',bn(X))'''


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    # 残差链接
    def forward(self, X,Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    def __init__(self, key_size, value_size, query_size, num_hiddens,norm_shape, ffn_num_input,ffn_num_hiddens,num_heads,dropout, use_bias=False,**kwargs):
        super(EncoderBlock,self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size,)