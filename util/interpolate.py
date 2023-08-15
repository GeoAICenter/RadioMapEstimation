import numpy as np
from scipy.interpolate import griddata

def interpolate_map(sampled_map, method):
  grid_x, grid_y = np.mgrid[0:sampled_map.shape[1]:1, 0:sampled_map.shape[2]:1]
  points = (sampled_map[1] == 1).nonzero()
  values = sampled_map[0][sampled_map[1] == 1] 
  grid_z0 = griddata(points, values, (grid_x, grid_y), method=method, fill_value=np.median(values))
  return grid_z0