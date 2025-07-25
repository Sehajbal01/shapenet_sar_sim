import numpy as np
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings, 
    MeshRasterizer,  
)


obj_filename = '/workspace/data/srncars/02958343/7dac31838b627748eb631ba05bd8dfe/models/model_normalized.obj'
device = 'cuda'

mesh = load_objs_as_meshes([obj_filename], device=device)

# Set up the camera
azimuths = torch.tensor([0, 30, 60])  # in degrees
elevations = torch.tensor([15, 45, 60])  # in degrees
R, T = look_at_view_transform(1.8, elevations, azimuths) # distance, elevation, azimuth
print('R: ', R)
print('T: ', T)
cameras = FoVOrthographicCameras(device=device, R=R, T=T, 
                                 min_x = -0.9, max_x = 0.9,
                                 min_y = -0.9, max_y = 0.9,)

# Set up the rasterizer
raster_settings = RasterizationSettings(
    image_size=128, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
rasterizer = MeshRasterizer(
    cameras=cameras,
    raster_settings=raster_settings
)


# get depth map
fragments = rasterizer(mesh)
depth_map = fragments.zbuf[0, ..., 0]  # (H, W) # missed rays are -1.0

# normalize all non-negative depth values to [0, 1]
positive_mask = depth_map >= 0
depth_map[positive_mask] = (depth_map[positive_mask] - depth_map[positive_mask].min()) / (depth_map[positive_mask].max() - depth_map[positive_mask].min())  # normalize to [0, 1]
depth_map[~positive_mask] = 0  # set negative values to 0

# plot it with pillow
depth_map_image = Image.fromarray((depth_map.cpu().numpy() * 255).astype(np.uint8))
depth_map_image.save('figures/depth_map.png')