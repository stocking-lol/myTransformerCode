from torch import nn

from layers.Attention import MultiHeadAttention
from layers.FFN import FFN
from layers.AddNorm import AddNorm
from layers.Dropout import Dropout


class Block(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = AddNorm(dim, eps=1e-6)
        self.pwff = FFN(dim, ff_dim)
        self.norm2 = AddNorm(dim, eps=1e-6)
        self.drop = Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.Block = Block(dim, num_heads, ff_dim, dropout)
        self.blocks = nn.ModuleList([
            self.Block for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
