import torch
from torch import nn
import math





class MultiHeadAttention(nn.Module):
    def __init__(self, dim , num_heads, dropout, *args, **kwargs):
        super(MultiHeadAttention, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(dropout)

        # Wq,Wk,Wv
        """
        Q,K,V:[bs,n,d] , W:[d,d/h]
        Qi = Q*W_qi , Ki = K*W_ki , Vi = V*W_vi
        Output=concat(Output_1,...,Output_h)
        Output:[bs,n,d]
        Output_i:[bs,n,d/h]
        Output_i = Qi*Ki_t*Vi
        """
        self.Wq = nn.Linear(dim, dim, bias=False)  # weigh:[d,q_size]
        self.Wk = nn.Linear(dim, dim, bias=False)  # weigh:[d,q_size]
        self.Wv = nn.Linear(dim, dim, bias=False)  # weigh:[d,q_size]
        self.Wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask):
        """Q,K,V:[bs,n,d]"""
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        q = transpose_qkv(q, self.num_heads)
        k = transpose_qkv(k, self.num_heads)
        v = transpose_qkv(v, self.num_heads)
        output = self.attention(q, k, v)
        if mask is not None:
            mask = mask.unsqueeze(1)
        output = transpose_o(output, self.num_heads)
        output = self.Wo(output)
        return output

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def transpose_qkv(x, num_heads):
    """
    X[bs,n,h*dm] -> [bs,n,h,dm] -> [bs,h,n,dm] -> [bs*h,n,dm]
    """
    bs, n, _ = x.shape
    x = x.reshape(bs, n, num_heads, -1)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bs * num_heads, n, -1)
    return x


"""# test
X = torch.randn((8, 3, 32))
num_heads = 2
multi_head_attention = MultiHeadAttention(32, 32, 32, 32, 2, Dropout=0)
Output = multi_head_attention(X, X, X)
print(Output.shape)"""

'''
李沐版MultiHeadAttention

class MultiHeadAttention(nn.Module):


    def __init__(self, key_size, query_size,value_size,
                 num_hiddens,
                 num_heads,Dropout,bias=False,**kwargs):
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
        return att
'''
