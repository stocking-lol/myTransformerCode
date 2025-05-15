import torch

# GPU device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model parameter setting
image_size = 224,
patch_size = 16,
num_classes = 1000,
dim = 768,
depth = 6,
heads = 8,
mlp_dim = 768 * 4,
dropout = 0.1,
emb_dropout = 0.1,