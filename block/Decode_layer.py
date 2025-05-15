import torch
from torch import nn
from layers.Attention import MultiHeadAttention
from layers.FFN import FFN
from layers.Dropout import Dropout
from layers.AddNorm import AddNorm
from Embedding.Token_Embedding import Token_Embedding
from Embedding.PositionalEncoding import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(
        self,
        i,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.i = i
        self.mht1 = MultiHeadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout)
        self.dropout1 = dropout(dropout)
        self.addnorm1 = AddNorm(norm_shape)
        self.mht2 = MultiHeadAttention(query_size, key_size, value_size,num_hiddens, num_heads, dropout)
        self.dropout2 = dropout(dropout)
        self.addnorm2 = AddNorm(norm_shape)
        self.ffn = FFN(ffn_num_input, ffn_num_hiddens)
        self.dropout3 = dropout(dropout)
        self.addnorm3 = AddNorm(norm_shape)


    def forward(self,x,state):
        enc_outputs,enc_valid_lens = state[0],state[1]
        """
        训练阶段：输出序列的词元在同一时间处理
        预测阶段：输出序列一个一个解码直到[eos]
        """
        if state[2][self.i] is None:
            key_value = x #[bos]
        else:
            key_value = torch.cat((state[2][self.i],x))
        state[2][self.i] = key_value
        if self.training:
            batch_size,num_steps,_ = x.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).repeat(batch_size,1)
        else:
            dec_valid_lens = None
        #自注意力
        _x2 = self.mht1(x,key_value)
        x2 = self.dropout1(_x2)
        _y =  self.addnorm1(_x2,x2)
        #编码器-解码器注意力 enc_output的开头(batch_size,num_steps,num_hiddens)
        y = self.dropout2(_y)
        _z = self.addnorm2(_y,y)
        _z = self.ffn(_z)
        z = self.dropout3(_z)
        outputs = self.addnorm3(_z,z)
        return outputs,state

class TransformerDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.embedding = Token_Embedding()
        self.positional_encoding = PositionalEncoding()
        self.dropout = Dropout()
        self.encblocks = nn.Sequential(*[DecoderLayer(*args, **kwargs) for _ in range(4)])

    def forward(self):
        self.embedding()
        self.positional_encoding()
        self.dropout()
        for _,block in self.encblocks:
            block()