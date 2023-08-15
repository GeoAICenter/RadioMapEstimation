import os
import glob
import pickle
import torch
import numpy as np


class RadioMapDataset(torch.utils.data.Dataset):
  '''
  This RadioMapDataset class is based on a class used for another project with a different dataset, 
  then (partially) adapted to a new dataset derived from the RadioMapSeer dataset. To make the outputs
  of the two datasets compatible (because we were testing some models on both), we return "complete_map"
  twice in the __getitem__ method, even though this is obviously redundant. 
  
  In addition to this, "sampled_map" (which is a pre-sampled version of "complete_map") is also mostly 
  redundant at this point, because the current models usually take the "complete_map" as input and then 
  sample that map within the forward loop. It is possible to use the "sampled_map" for evaluation by
  setting pre_sampled=True in the evaluate method, but if we change the dataset further or decide we don't
  need to compare with earlier models that used pre-sampled maps, we could eliminate this output as well.

  The specific datasets this class is designed to work with are RadioMapSeer_32x32 and RadioMapSeer_64x64
  in the "RadioMapSeer" folder under "Datasets" in the shared Google Drive:
  https://drive.google.com/drive/folders/1JKXDvh76m-GJnlTzb3Crx46qRz2QXrTK?usp=sharing

  It should also work on any of the datasets stored in the "Deep Completion Autoencoder" folder under
  "Datasets" in the shared Google Drive, but I haven't tested this:
  https://drive.google.com/drive/folders/1G16KbTCQ7flF49zVj8-E2S2LUBLDb_eI?usp=sharing

  The original RadioMapSeer Dataset (which this will not work on) is stored as a zip file in the "RadioMapSeer"
  folder in the shared Google Drive, and can also be downloaded here:
  https://radiomapseer.github.io/
  '''

  def __init__(self, data):
    super().__init__()
    if isinstance(data, str):
        self.samples = glob.glob(os.path.join(data, '*'))
    elif isinstance(data, list):
        self.samples = data
    else:
        raise ValueError('Argument "Data" should either be a string showing the path to the data directory' \
                         'or a list of strings showing the paths to individual samples')

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    '''
    returns
        sampled_map:    2xHxW torch.Tensor. First channel is sampled radio power measurements, with all unsampled
                        positions filled with 0. Second channel is ternary mask, with 1 for sampled locations, 0
                        for unsampled locations, and -1 for building locations.
        
        complete_map:   1xHxW torch.Tensor. Radio power at all positions.

        building_mask:  1xHxW torch.Tensor. Binary mask, with 1 for building locations and 0 for free space
                        (non-building) locations.

        path:           String. Full path to saved maps.

        tx_loc:         Size 2 torch.Tensor. X and Y coordinates of transmitter, scaled to between 0 and 1 as a
                        percentage of map width / height.
         
    '''
    path = self.samples[idx]
    with open(path, 'rb') as f:
      sampled_map, complete_map, building_mask, tx_loc = pickle.load(f)
    return sampled_map, complete_map, building_mask, complete_map, path, tx_loc