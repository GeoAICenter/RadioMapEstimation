import os

import torch
from torch.utils.data import DataLoader

from models.mae_cbam import MAE_CBAM
from util.dataset import RadioMapDataset

ROOT_DIR = '/home/UNT/tjt0147/research_projects/rme/'
DATASET_DIR = os.path.join(ROOT_DIR, 'datasets')

IMG_SIZE_64_TRAIN = os.path.join(
    DATASET_DIR, 'RadioMapSeer_64x64_Train')
IMG_SIZE_64_VAL = os.path.join(
    DATASET_DIR, 'RadioMapSeer_64x64_Val')

CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'archive', 'checkpoints')

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
    optimizer = torch.optim.AdamW(model, lr=learning_rate)
    return optimizer

img_size = 64
min_samples, max_samples = 3, 41
patch_size = 1
in_chans = 1
embed_dim = 128
pos_dim = 64
depth = 6
num_heads = 4

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Parameters: {count_parameters(model)}')

model.fit(
    get_train_dataloader(),
    get_val_dataloader(),
    get_optimizer(model),
    scheduler=None,
    min_samples=min_samples,
    max_samples=max_samples,
    run_name=None,
    epochs=20,
    eval_model_epochs=1,
    save_model_dir=CHECKPOINT_DIR
)