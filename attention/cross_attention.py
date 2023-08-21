'''
This code is adapted from timm version 0.9.5,
authored by Ross Wightman under Apache Software License 2.0.
Current version of the code can be found at
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/crossvit.py
Original license is included in the file "Apache License" in the root directory of this source tree.
'''

import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, enc, dec):
        '''
        In this implementation of cross-attention, the vectors in "dec" (i.e. decoder) pay attention
        to the vectors in "enc" (i.e. encoder). Both sets of vectors are assumed to have the same
        batch size (B) and embedding depth (or "channels", C), but may have different numbers of
        elements (N_enc, N_dec). This is changed from the original code.
        '''
        B, N_dec, C = dec.shape
        N_enc = enc.shape[1]
        # B N_dec C -> B N_dec H (C/H) -> B H N_dec (C/H)
        q = self.wq(dec).reshape(B, N_dec, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # B N_enc C -> B N_enc H (C/H) -> B H N_enc (C/H)
        k = self.wk(enc).reshape(B, N_enc, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # B N_enc C -> B N_enc H (C/H) -> B H N_enc (C/H)
        v = self.wv(enc).reshape(B, N_enc, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B H N_dec (C/H) @ B H (C/H) N_enc -> B H N_dec N_enc
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_dec, C)  # (B H N_dec N_enc @ B H N_enc (C/H)) -> B H N_dec (C/H) -> B N_dec H (C/H) -> B N_dec C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = nn.Dropout(drop_path)

    def forward(self, enc, dec):
        x = dec + self.drop_path(self.attn(self.norm1(enc), self.norm2(dec)))
        return x