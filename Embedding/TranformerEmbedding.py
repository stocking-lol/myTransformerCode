from torch import nn

from Embedding.Token_Embedding import Token_Embedding
from Embedding.PositionalEncoding import PositionalEncoding

class Transformer_Embedding(nn.Module):
    """
    token_embedding + positional_encoding
    """

    def __init__(self, d_model, vocab_size, max_len, dropout):
        """

        :param d_model: dimension of model
        :param vocab_size: vocabulary size:
        :param max_len: max sequence length:
        :param dropout: Dropout rate
        """
        super(Transformer_Embedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        token_embed = self.token_embedding(x)
        position_embed = self.positional_encoding(x)
        return self.dropout(token_embed) + position_embed
