'''
This code is adapted from Masked Autoencoders: A PyTorch Implementation,
attributed to Meta Platforms, Inc. and affiliates under CC BY-NC 4.0 International. 
Current version of the code can be found at
https://github.com/facebookresearch/mae/blob/main/models_mae.py
Original license is included in the file "CC BY-NC License" in the root directory of this source tree.

# --------------------------------------------------------
References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
'''

import numpy as np
import torch
import torch.nn as nn

from layers.self_attention import Block
from layers.cross_attention import CrossAttentionBlock
from util.patch_embed import PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class _MAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, pos_dim=64, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(_MAE, self).__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.in_chans = in_chans
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim + pos_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim + pos_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim + pos_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim + pos_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim + pos_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim + pos_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, pos_dim), requires_grad=True)  # learnable sin-cos embedding
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (but don't freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, building_mask, min_samples, max_samples):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = np.random.choice(np.arange(min_samples,max_samples+1))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        noise = noise + self.patchify(building_mask).squeeze()
        '''
        Free space on building_mask is 1, buildings are 0. Adding the building_mask ensures that the noise values at
        indices corresponding to free space are larger than the noise values at indices corresponding to buildings.
        This allows us to stratify them below so that we are only drawing samples from free space indices.
        '''

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1, descending=True)  # descend: large is keep, small is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1, descending=False)  # ascend: we still want indices to go from small to large

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_shuffle, ids_restore

    def forward_encoder(self, x, building_mask, min_samples, max_samples, pre_sampled=False):
        if pre_sampled:
            mask = (x[:, 1:] == 1)
            mask = self.patchify(mask).squeeze(-1)
            x = x[:,:1]

        # embed patches
        x = self.patch_embed(x)

        # append pos embed w/o cls token
        x = torch.cat((x, self.pos_embed[:,1:,:].expand(x.shape[0],-1,-1)), dim=-1)

        if pre_sampled:
            ids_shuffle = torch.argsort(mask * 1, dim=1, descending=True)
            ids_restore = torch.argsort(ids_shuffle, dim=1, descending=False)
            x = x[mask].unsqueeze(0)
            mask = 1 - (1 * mask)

        else:
            # masking: length -> length * mask_ratio
            x, mask, ids_shuffle, ids_restore = self.random_masking(x, building_mask, min_samples=min_samples, max_samples=max_samples)

        # append cls token
        cls_token = torch.cat((self.cls_token, self.pos_embed[:, :1, :]), dim=-1)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_shuffle, ids_restore

    def forward_decoder(self, x, ids_shuffle, ids_restore):
        # embed encoder tokens
        enc = self.decoder_embed(x)

        # generate mask tokens and append positional embeddings to both encoder tokens and decoder mask tokens
        dec = self.mask_token.expand(enc.shape[0], ids_restore.shape[1] + 1 - enc.shape[1], -1)
        pos_embed = torch.gather(self.pos_embed[:,1:,:].expand(dec.shape[0], -1, -1), dim=1, 
                                 index=ids_shuffle.unsqueeze(-1).repeat(1,1,self.pos_embed.shape[-1]))
        pos_embed = torch.cat((self.pos_embed[:,:1,:].expand(dec.shape[0], -1, -1), pos_embed),dim=1)

        enc = torch.cat((enc, pos_embed[:,:enc.shape[1],:].expand(enc.shape[0],-1,-1)), dim=-1)
        dec = torch.cat((dec, pos_embed[:,enc.shape[1]:,:].expand(enc.shape[0],-1,-1)), dim=-1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            dec = blk(enc, dec)
        dec = self.decoder_norm(dec)

        x = torch.cat((enc[:, 1:, :], dec), dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1, x.shape[2]))

        # predictor projection
        x = self.decoder_pred(x)

        return x


    def forward(self, imgs, building_mask, min_samples, max_samples, pre_sampled=False):
        latent, mask, ids_shuffle, ids_restore = self.forward_encoder(imgs, building_mask, min_samples, max_samples, pre_sampled)
        pred = self.forward_decoder(latent, ids_shuffle, ids_restore)  # [N, L, p*p*3]

        # replace sampled locations in prediction with sampled measurements
        mask = mask.unsqueeze(-1)
        samples = self.patchify(imgs) * (1-mask) # mask has 0 for sampled locations, 1 for unsampled locations
        pred *= mask
        pred += samples
        return self.unpatchify(pred), self.unpatchify(mask)