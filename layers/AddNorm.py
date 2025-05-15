import torch
from torch import nn

class AddNorm(nn.Module):
    def __init__(self,norm_shape,*args,**kwargs):
        super(AddNorm, self).__init__(*args,**kwargs)
        """
        :param norm_shape:model dimension
        """
        self.ln = nn.LayerNorm(normalized_shape=norm_shape,eps=1e-6)

    def forward(self,x,y):
        """
        残差连接
        :param x:上一个多头注意力的输入
        :param y:上一个多头注意力的输出
        :return:两个输入层归一化的结果
        """
        y = x+y
        y = self.ln(y)
        return y