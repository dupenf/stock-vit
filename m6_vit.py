from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
import torch
from einops import repeat


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout
        )
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(x, x, x)
        return attn_output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )


ff = FeedForward(dim=128, hidden_dim=256)
ff(torch.ones((1, 5, 128))).shape


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class ViT3D(nn.Module):
    def __init__(
        self,
        features_len=8,
        seq_len=20,
        # img_size=144,
        # patch_size=4,
        emb_dim=512,
        n_layers=24,
        num_classes=600,
        dropout=0.1,
        heads=8,
    ):
        super(ViT3D, self).__init__()

        # self.feature_embedding = nn.Sequential(
        #     # break-down the image in s1 x s2 patches and flat them
        #     # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #     # nn.Linear(patch_size * patch_size * in_channels, emb_size)
        #     nn.Linear(features_len, emb_dim)
        # )
        self.feature_embedding = nn.Linear(features_len, emb_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        # Transformer Encoder
        self.n_layers = n_layers
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(
                    PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))
                ),
                ResidualAdd(
                    PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout=dropout))
                ),
            )
            self.layers.append(transformer_block)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim), 
            nn.Linear(emb_dim, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Get patch embedding vectors
        x = self.feature_embedding(x)
        b, n, _ = x.shape

        # Add cls token to inputs
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, : (n + 1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on classification token
        x =  self.head(x[:, 0, :])
        x = self.softmax(x)
        return x


# model = ViT3D()
# print(model)
# print(model(torch.ones((1,50,8))))
