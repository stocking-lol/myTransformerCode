import math
import torch
from torch import nn

'''class MultiHeadAttention(nn.Module):


    def __init__(self, key_size, query_size,value_size,
                 num_hiddens,
                 num_heads,dropout,bias=False,**kwargs):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.query_size = query_size
        #定义线性变换矩阵
        self.linear_q = nn.Linear(query_size, num_hiddens,bias=False)
        self.linear_k = nn.Linear(dim_in, key_size,bias=False)
        self.linear_v = nn.Linear(dim_in, value_size,bias=False)
        self._norm_fact = 1 / math.sqrt(key_size // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.key_size // nh
        dv = self.dim_v // nh

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact
        dist = dist.softmax(dim=-1)

        att = torch.matmul(dist, v)
        att = att.transpose(1,2).reshape(batch, n, self.dim_v )
        return att'''