from torch import nn

class Token_Embedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        """

        :param vocab_size: size of vocabulary
        :param d_model:dimensions of model
        """
        super(Token_Embedding, self).__init__(vocab_size, d_model)
