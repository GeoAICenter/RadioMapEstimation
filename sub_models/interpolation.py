import torch
import numpy as np
from scipy.interpolate import griddata

class Interpolation:

    def __init__(self, img_size=224, patch_size=16, in_chans=3):
        self.patch_size = patch_size
        self.in_chans = in_chans

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_size
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
        p = self.patch_size
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

    def interpolate_map(self, sampled_map, mask, method):
        grid_x, grid_y = np.mgrid[0:sampled_map.shape[0], 0:sampled_map.shape[1]]
        points = (mask == 1).nonzero()
        values = sampled_map[sampled_map == 1] 
        map = griddata(points.cpu(), values.cpu(), (grid_x, grid_y), method=method, fill_value=np.median(values.cpu()))
        map = torch.Tensor(map).unsqueeze(0)
        return map
    
    def forward(self, x, building_mask, min_samples, max_samples, method, pre_sampled=False):
        if pre_sampled:
            mask = (x[:, 1:] != 1)
            x = x[:,:1]

        else:
            x = self.patchify(x)
            N, L, D = x.shape
            x_masked, mask, ids_shuffle, ids_restore = self.random_masking(x, building_mask, min_samples, max_samples)
            mask_token = torch.zeros((N, L - x_masked.shape[1], D), device=x.device)
            x = torch.cat((x_masked, mask_token), dim=1)
            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1, x.shape[2]))
            x, mask = self.unpatchify(x), self.unpatchify(mask)

        map = torch.stack([self.interpolate_map(x[i,0], mask[i,0], method) for i in range(x.shape[0])])

        return map, mask