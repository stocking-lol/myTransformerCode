import torch
from torch import nn

Dropout = nn.Dropout(p=0.1)
'''
inputs = torch.randn(10000, dtype=torch.float)
outputs = Dropout(inputs)
print((outputs==0).sum().item)
'''