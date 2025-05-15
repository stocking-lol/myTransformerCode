import torch
from torch import nn
from transformer import Transformer


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size,tuple) else (image_size, image_size)  # 224*224
        patch_height, patch_width = patch_size if isinstance(patch_size,tuple) else (patch_size, patch_size)  # 16 * 16

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_height, stride=patch_height),
            # Rearrange('b c (h p1) (w p2) -&gt; b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # flatten: [B, C, H, W] -> [B, C, HW]   从指定维度开始展平（flatten）
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.to_patch_embedding(img).flatten(2).transpose(1, 2)  ## img 1 3 224 224  输出形状x : 1 196 768
        b, n, _ = x.shape  ##
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # cls_tokens = repeat(self.cls_token, '() n d -&gt; b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)