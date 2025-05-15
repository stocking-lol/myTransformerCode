import torch
from torch import nn


class FFN(nn.Module):
    def __init__(self, d, dm, *args, **kwargs):
        """
        :param d:dimensions of model
        :param dm: dimensions of dense1
        """
        super(FFN, self).__init__(*args, **kwargs)
        self.dense1 = nn.Linear(d, dm)  # weight:[dm,d]
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(dm, d)  # weight:[d,dm]

    def forward(self, x):
        y = self.dense1(x)
        y = self.relu(y)
        y = self.dense2(y)
        return y
