import os
import argparse

import torch
from torch.utils.data import DataLoader

from models.mae_cbam import MAE_CBAM
from models.unet import UNet
from util.dataset import RadioMapDataset

ROOT_DIR = '/home/UNT/tjt0147/research_projects/rme/'
DATASET_DIR = os.path.join(ROOT_DIR, 'datasets')

IMG_SIZE_64_TRAIN = os.path.join(
    DATASET_DIR, 'RadioMapSeer_64x64_Train')
IMG_SIZE_64_VAL = os.path.join(
    DATASET_DIR, 'RadioMapSeer_64x64_Val')

CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'archive', 'checkpoints')

parser = argparse.ArgumentParser()
parser.add_argument('--img_size',
                    type=int,
                    default=64,
                    choices=[32, 64])
parser.add_argument('--samples', 
                    nargs=2, 
                    type=int, 
                    metavar=('min_samples', 'max_samples'),
                    required=True)
args = parser.parse_args()

batch_size = 64
learning_rate = 1e-3

def get_train_dataset():
    train_dataset = RadioMapDataset(IMG_SIZE_64_TRAIN)
    return train_dataset

def get_train_dataloader():
    train_dataset = get_train_dataset()
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  drop_last=True)
    return train_dataloader

def get_val_dataset():
    val_dataset = RadioMapDataset(IMG_SIZE_64_VAL)
    return val_dataset

def get_val_dataloader():
    val_dataset = get_val_dataset()
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False)    
    return val_dataloader

def get_optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer

img_size = args.img_size
min_samples, max_samples = sorted(args.samples)
patch_size = 1
in_chans = 1
embed_dim = 128
pos_dim = 64
depth = 6
num_heads = 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_mae_cbam_model():
    model = MAE_CBAM(
        f'mae-cbam_img_{img_size}_samples_{min_samples}-{max_samples}',
        img_size=img_size, 
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        pos_dim=pos_dim,
        depth=depth,
        num_heads=num_heads,    
        decoder_embed_dim=embed_dim,
        decoder_depth=depth,
        decoder_num_heads=num_heads)
    return model

def get_unet_model():
    model = UNet(
        f'unet_img_{img_size}_samples_{min_samples}-{max_samples}',
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans)
    return model

model = get_unet_model().to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Number of Parameters: {count_parameters(model)}')

model.fit(
    get_train_dataloader(),
    get_val_dataloader(),
    get_optimizer(model),
    scheduler=None,
    min_samples=min_samples,
    max_samples=max_samples,
    run_name=None,
    dB_max=1,
    dB_min=0,
    free_space_only=False,
    #mae_regularization=False,
    epochs=20,
    save_model_epochs=20,
    eval_model_epochs=1,
    save_model_dir=CHECKPOINT_DIR
)