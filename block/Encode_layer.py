from torch import nn
from layers.Attention import MultiHeadAttention
from layers.FFN import FFN
from layers.AddNorm import AddNorm
from Embedding.Token_Embedding import Token_Embedding
from Embedding.PositionalEncoding import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(
            self,
            key_size,
            query_size,
            value_size,
            num_hiddens,
            norm_shape,
            ffn_num_input,
            ffn_num_hiddens,
            num_heads,
            dropout,
            *args,
            **kwargs):
        super(EncoderLayer, self).__init__()
        self.mht = MultiHeadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout)
        self.dropout1 = dropout(dropout)
        self.addnorm1 = AddNorm(norm_shape)
        self.ffn = FFN(ffn_num_input, ffn_num_hiddens)
        self.dropout2 = dropout(dropout)
        self.addnorm2 = AddNorm(norm_shape)

    def forward(self, x, mask=None):
        # 自注意力
        _x = x
        x = self.mht(x)

        # 残差连接
        x = self.dropout1(x)
        _y = self.addnorm1(x, _x)

        # fnn
        _y = self.ffn(_y)
        y = self.dropout2(_y)
        self.addnorm2(_y, y)
        return y


class TransformerEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 ffn_num_input,
                 ffn_num_hiddens,
                 num_heads,
                 dropout,
                 *args,
                 **kwargs):
        super.__init__(*args, **kwargs)
        self.embedding = Token_Embedding(vocab_size,)
        self.positional_encoding = PositionalEncoding()
        self.dropout = dropout(dropout)
        self.encblocks = nn.Sequential(*[EncoderLayer(*args, **kwargs) for _ in range(4)])

    def forward(self):
        self.embedding()
        self.positional_encoding()
        self.dropout()
        for _, block in self.encblocks:
            block()
