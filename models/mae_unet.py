import os
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sub_models.mae import MAE
from sub_models.unet import UNet

class MAE_UNet(nn.Module):
    def __init__(self,
                 # transformer params 
                 img_size=64, 
                 patch_size=1, 
                 in_chans=1,
                 embed_dim=64,
                 pos_dim=16, 
                 depth=8, 
                 num_heads=4,
                 decoder_embed_dim=32, 
                 decoder_depth=4, 
                 decoder_num_heads=4,
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False,
                 # UNet params
                 latent_channels=64,
                 out_channels=1,
                 features=[32,32,32]):
        
        super(MAE_UNet, self).__init__()
        
        self.transformer = MAE(img_size, patch_size, in_chans, embed_dim, pos_dim, depth, num_heads,
                 decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio, norm_layer, norm_pix_loss)
        
        self.unet = UNet(in_channels=3, latent_channels=latent_channels, out_channels=out_channels, features=features)


    def forward(self, x, building_mask, min_samples, max_samples, pre_sampled=False):
      # building_mask has 1 for free space, 0 for buildings (may change this for future datasets).
      inv_building_mask = 1-building_mask

      # sample_mask has 1 for non-sampled locations, 0 for sampled locations.
      map1, sample_mask = self.transformer(x, building_mask, min_samples, max_samples, pre_sampled)

      x = torch.cat((map1, sample_mask, inv_building_mask), dim=1)

      map2 = self.unet(x)

      return map1, sample_mask, map2


    def step(self, batch, optimizer, min_samples, max_samples, train=True, free_space_only=False, mae_regularization=False):
        with torch.set_grad_enabled(train):
            sampled_map, complete_map, building_mask, complete_map, path, tx_loc = batch
            complete_map, building_mask = complete_map.to(torch.float32).to(device), building_mask.to(torch.float32).to(device)

            map1, _, pred_map = self.forward(complete_map, building_mask, min_samples, max_samples).to(torch.float32)
            
            # building_mask has 1 for free space, 0 for buildings (may change this for future datasets)
            # RadioUNet also calculates loss over buildings, whereas our previous models did not. I have included both options here.
            if free_space_only:
                loss_ = nn.functional.mse_loss(pred_map * building_mask, complete_map * building_mask).to(torch.float32)
                if mae_regularization:
                    loss_ += nn.functional.mse_loss(map1 * building_mask, complete_map * building_mask).to(torch.float32)
            else:
                loss_ = nn.functional.mse_loss(pred_map, complete_map).to(torch.float32)
                if mae_regularization:
                    loss_ += nn.functional.mse_loss(map1, complete_map).to(torch.float32)

            if train:
                loss_.backward()
                optimizer.step()
                optimizer.zero_grad()

        return loss_
    

    def fit(self, train_dl, test_dl, optimizer, scheduler, min_samples, max_samples, dB_max=-47.84, dB_min=-147,
            free_space_only=False, epochs=100, save_model_epochs=25, eval_model_epochs=5, save_model_dir ='/content', mae_regularization=False):
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for i, batch in enumerate(train_dl):
                loss = self.step(batch, optimizer, min_samples, max_samples, train=True, free_space_only=free_space_only, mae_regularization=mae_regularization)
                running_loss += loss.detach().item()
                print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}')

            if eval_model_epochs > 0:
                if (epoch + 1) % eval_model_epochs == 0:
                    test_loss = self.evaluate(test_dl, min_samples, max_samples, dB_max, dB_min, free_space_only=free_space_only)
                    print(f'{test_loss}, [{epoch + 1}]')

            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'epoch_{epoch}.pth')
                self.save_model(filepath)
            
            if scheduler:
              scheduler.step()    


    def fit_wandb(self, train_dl, test_dl, optimizer, scheduler, min_samples, max_samples, project_name, run_name, 
                  dB_max=-47.84, dB_min=-147, free_space_only=False, epochs=100, save_model_epochs=25, save_model_dir='/content', mae_regularization=False):
        import wandb
        wandb.init(project=project_name, name=run_name)
        for epoch in range(epochs):
            self.train()
            train_running_loss = 0.0
            for i, batch in enumerate(train_dl):
                loss = self.step(batch, optimizer, min_samples, max_samples, train=True, free_space_only=free_space_only, mae_regularization=mae_regularization)
                train_running_loss += loss.detach().item()
                train_loss = train_running_loss/(i+1)
                print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {train_loss}')
            test_loss = self.evaluate(test_dl, min_samples, max_samples, dB_max, dB_min, free_space_only=free_space_only)
            print(f'{test_loss}, [{epoch + 1}]')
                                    
            wandb.log({'train_loss': train_loss, 'test_loss': test_loss})

            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'{run_name}, {epoch+1} epochs.pth')
                self.save_model(filepath)

            if scheduler:
                scheduler.step()


    def evaluate(self, test_dl, min_samples, max_samples, dB_max=-47.84, dB_min=-147, free_space_only=False, pre_sampled=False):
        self.eval()
        losses = 0
        pixels = 0
        with torch.no_grad():
            for i, batch in enumerate(test_dl):
                    sampled_map, complete_map, building_mask, complete_map, path, tx_loc = batch
                    building_mask = building_mask.to(torch.float32).to(device)
                    if pre_sampled:
                        sampled_map = sampled_map.to(torch.float32).to(device)
                        _, _, pred_map = self.forward(sampled_map, building_mask, min_samples, max_samples, pre_sampled=True)
                        pred_map = pred_map.detach().cpu()

                    else:
                        complete_map = complete_map.to(torch.float32).to(device)
                        _, _, pred_map = self.forward(complete_map, building_mask, min_samples, max_samples, pre_sampled=False)
                        pred_map = pred_map.detach().cpu()

                    complete_map = complete_map.to(torch.float32).cpu()
                    building_mask = building_mask.cpu()

                    # building_mask has 1 for free space, 0 for buildings (may change this for future datasets)
                    # RadioUNet also calculates loss over buildings, whereas our previous models did not. I have included both options here.
                    if free_space_only:
                        loss = nn.functional.mse_loss(self.scale_to_dB(pred_map * building_mask, dB_max, dB_min),
                                                      self.scale_to_dB(complete_map * building_mask, dB_max, dB_min),
                                                      reduction='sum')
                        pix = building_mask.sum()
                    else:
                        loss = nn.functional.mse_loss(self.scale_to_dB(pred_map, dB_max, dB_min),
                                                      self.scale_to_dB(complete_map, dB_max, dB_min),
                                                      reduction='sum')
                        pix = pred_map.numel()

                    losses += loss
                    pixels += pix
                    print(f'{torch.sqrt(loss / pix).item()}')

            return math.sqrt(losses / pixels)
        

    def visualize_maps(self, x, building_mask, min_samples, max_samples, pre_sampled):
        self.eval()
        map1, sample_mask, map2 = self.forward(x, building_mask, min_samples, max_samples, pre_sampled)
        if pre_sampled:
            x = x[:,0]
        fig, axs = plt.subplots(1,4, figsize=(15,5))
        axs[0].imshow(x[0,0,:,:].cpu())
        axs[1].imshow(sample_mask[0,0,:,:].detach().cpu())
        axs[2].imshow(map1[0,0,:,:].detach().cpu())
        axs[3].imshow(map2[0,0,:,:].detach().cpu())
        return fig, x.cpu(), sample_mask.detach().cpu(), map1.detach().cpu(), map2.detach().cpu()


    def scale_to_dB(self, value, dB_max, dB_min):
        range_dB = dB_max - dB_min
        dB = value * range_dB + dB_min
        return torch.Tensor(dB)
    

    def save_model(self, out_path):
        torch.save(self, out_path)