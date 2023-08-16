import os
import json
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from util.random_sample import RandomSample
from sub_models._unet import _UNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class UNet(nn.Module):
    def __init__(self,
                 model_name,
                 model_type='UNet',
                 # map sample params 
                 img_size=64, 
                 patch_size=1, 
                 in_chans=1,
                 # UNet params
                 latent_channels=64,
                 out_channels=1,
                 features=[32,32,32]):
        
        self.config = dict(model_name=model_name,
                           model_type='UNet',
                           img_size=img_size, 
                           patch_size=patch_size, 
                           in_chans=in_chans,
                           latent_channels=latent_channels,
                           out_channels=out_channels,
                           features=features)
        
        super(UNet, self).__init__()
                
        self.model1 = RandomSample(img_size=img_size, patch_size=patch_size, in_chans=in_chans)
        
        self.model2 = _UNet(in_channels=3, latent_channels=latent_channels, out_channels=out_channels, features=features)


    def forward(self, x, building_mask, min_samples, max_samples, pre_sampled=False):
      # building_mask has 1 for free space, 0 for buildings (may change this for future datasets).
      inv_building_mask = 1-building_mask

      # sample_mask has 1 for non-sampled locations, 0 for sampled locations.
      map1, sample_mask = self.model1.forward(x, building_mask, min_samples, max_samples, pre_sampled)

      x = torch.cat((map1, sample_mask, inv_building_mask), dim=1)

      map2 = self.model2(x)

      return map1, sample_mask, map2
    


    def step(self, batch, optimizer, min_samples, max_samples, train=True, free_space_only=False):
        with torch.set_grad_enabled(train):
            sampled_map, complete_map, building_mask, complete_map, path, tx_loc = batch
            complete_map, building_mask = complete_map.to(torch.float32).to(device), building_mask.to(torch.float32).to(device)

            map1, _, pred_map = self.forward(complete_map, building_mask, min_samples, max_samples)
            map1, pred_map = map1.to(torch.float32), pred_map.to(torch.float32)
            
            # building_mask has 1 for free space, 0 for buildings (may change this for future datasets)
            # RadioUNet also calculates loss over buildings, whereas our previous models did not. I have included both options here.
            if free_space_only:
                loss_ = nn.functional.mse_loss(pred_map * building_mask, complete_map * building_mask).to(torch.float32)
            else:
                loss_ = nn.functional.mse_loss(pred_map, complete_map).to(torch.float32)

            if train:
                loss_.backward()
                optimizer.step()
                optimizer.zero_grad()

        return loss_
    

    def fit(self, train_dl, val_dl, optimizer, scheduler, min_samples, max_samples, run_name=None, dB_max=-47.84, dB_min=-147,
            free_space_only=False, epochs=100, save_model_epochs=25, eval_model_epochs=5, save_model_dir ='/content'):
        
        if run_name is None:
            run_name = f'{min_samples}-{max_samples} samples'
            if free_space_only:
                run_name += ', free space only'

        self.training_config = dict(train_batch=train_dl.batch_size, val_batch=val_dl.batch_size, min_samples=min_samples,
                                    max_samples=max_samples, run_name=run_name, dB_max=dB_max, dB_min=dB_min, 
                                    free_space_only=free_space_only)
        
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for i, batch in enumerate(train_dl):
                loss = self.step(batch, optimizer, min_samples, max_samples, train=True, free_space_only=free_space_only, mae_regularization=mae_regularization)
                running_loss += loss.detach().item()
                print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}')

            if eval_model_epochs > 0:
                if (epoch + 1) % eval_model_epochs == 0:
                    test_loss = self.evaluate(val_dl, min_samples, max_samples, dB_max, dB_min, free_space_only=free_space_only)
                    print(f'{test_loss}, [{epoch + 1}]')
            
            if scheduler:
              scheduler.step()   

            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                self.save_model(epoch+1, optimizer, scheduler, out_dir=save_model_dir)


    def fit_wandb(self, train_dl, val_dl, optimizer, scheduler, min_samples, max_samples, project_name, run_name=None, 
                  dB_max=-47.84, dB_min=-147, free_space_only=False, epochs=100, save_model_epochs=25, save_model_dir='/content'):
        
        if run_name is None:
            run_name = f'{min_samples}-{max_samples} samples'
            if free_space_only:
                run_name += ', free space only'
      
        self.training_config = dict(train_batch=train_dl.batch_size, val_batch=val_dl.batch_size, min_samples=min_samples,
                                    max_samples=max_samples, project_name=project_name, run_name=run_name, dB_max=dB_max,
                                    dB_min=dB_min, free_space_only=free_space_only)

        import wandb
        config = {**self.config, **self.training_config}
        wandb.init(project=project_name, group=config['model_name'], name=run_name, config=config)

        for epoch in range(epochs):
            self.train()
            train_running_loss = 0.0
            for i, batch in enumerate(train_dl):
                loss = self.step(batch, optimizer, min_samples, max_samples, train=True, free_space_only=free_space_only, mae_regularization=mae_regularization)
                train_running_loss += loss.detach().item()
                train_loss = train_running_loss/(i+1)
                print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {train_loss}')
            test_loss = self.evaluate(val_dl, min_samples, max_samples, dB_max, dB_min, free_space_only=free_space_only)
            print(f'{test_loss}, [{epoch + 1}]')
                                    
            wandb.log({'train_loss': train_loss, 'test_loss': test_loss})

            if scheduler:
                scheduler.step()

            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                self.save_model(epoch+1, optimizer, scheduler, out_dir=save_model_dir)


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
    

    def save_model(self, epoch=0, optimizer=None, scheduler=None, out_dir='/content'):
        # First time model is saved (as indicated by not having a pre-existing model directory),
        # create model folder and save model config.
        model_name = self.config['model_name']
        model_dir = os.path.join(out_dir, model_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

            # Save model config (i.e. params called with __init__).
            config_path = os.path.join(model_dir, f'{model_name} config.json')
            with open(config_path, 'w') as config_file:
                json.dump(self.config, config_file, indent=2)

        # If called with fit or fit_wandb (hence having "self.training_config"), make a new run
        # directory and save training_config, optimizer, scheduler, and trained weights.
        if hasattr(self, 'training_config'):
            run_name = self.training_config['run_name']
            run_dir = os.path.join(model_dir, run_name)
            if not os.path.isdir(run_dir):
                os.mkdir(run_dir)

                # Save training_config first time for new run
                train_config_path = os.path.join(run_dir, 'training_config.json')
                with open(train_config_path, 'w') as train_config_file:
                    json.dump(self.training_config, train_config_file, indent=2)

            # Save optimizer (if specified in fit or fit_wandb)
            if optimizer:
                opt_path = os.path.join(run_dir, f'{epoch} epochs optimizer.pth')
                torch.save(optimizer.state_dict(), opt_path)

            # Save scheduler (if specified in fit or fit_wandb)
            if scheduler:
                sched_path = os.path.join(run_dir, f'{epoch} epochs scheduler.pth')
                torch.save(scheduler.state_dict(), sched_path)

            # Save state dict
            model_path = os.path.join(run_dir, f'{epoch} epochs state dict.pth')
            torch.save(self.state_dict(), model_path)