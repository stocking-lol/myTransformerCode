import torch
def mask_qk(x, valid):
    bs, _, n = x.shape
    x = x.reshape(-1, n)
    valid = torch.repeat_interleave(valid, n)
    mask = torch.arange(n)[None, :] < valid[:, None]
    x[~mask] = -1e6
    x = x.reshape(bs, -1, n)
    return x