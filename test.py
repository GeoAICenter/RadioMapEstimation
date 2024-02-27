import os
import torch
import numpy as np

from util.dataset import RadioMapDataset
from models.mae_cbam import MAE_CBAM

model_name = 'mae-cbam_3-41_samples_fso_mae'
root_dir = '/home/UNT/tjt0147/research_projects/rme/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model():    
    model = MAE_CBAM(model_name).to(device)
    model_path = os.path.join(root_dir, f'pretrained_models/{model_name}/')
    model.load_state_dict(torch.load(os.path.join(model_path, '50 epochs state dict.pth')))
    return model

def load_dataset():
    dataset_path = os.path.join(root_dir, 'datasets/RadioMapSeer_64x64_Val')
    val_ds = RadioMapDataset(dataset_path)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1024, shuffle=False)
    return val_dl

if __name__ == '__main__':
    model = load_model()
    val_dl = load_dataset()
    min_samples, max_samples = 3, 41
    loss = model.evaluate(
        val_dl, 
        min_samples, 
        max_samples, 
        free_space_only=True, 
        pre_sampled=False
    )