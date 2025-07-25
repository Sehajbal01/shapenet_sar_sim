import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)


obj_filename = '/workspace/data/srncars/02958343/7dac31838b627748eb631ba05bd8dfe/models/model_normalized.obj'
device = 'cuda'

mesh = load_objs_as_meshes([obj_filename], device=device)
plt.figure(figsize=(7,7))
texture_image=mesh.textures.maps_padded()
plt.imshow(texture_image.squeeze().cpu().numpy())
plt.axis("off")
plt.savefig('figures/texture_image.png', bbox_inches='tight', pad_inches=0)


# 2

# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
R, T = look_at_view_transform(1.8, 45, 45) 
cameras = FoVOrthographicCameras(device=device, R=R, T=T, 
                                 min_x = -0.9, max_x = 0.9,
                                 min_y = -0.9, max_y = 0.9,)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=128, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
rasertizer = MeshRasterizer(
    cameras=cameras,
    raster_settings=raster_settings
)
shader = SoftPhongShader(
    device=device,
    cameras=cameras,
    lights=lights,
)
renderer = MeshRenderer(
    rasterizer=rasertizer,
    shader=shader
)


# 3

images = renderer(mesh)
print('images.shape: ', images.shape)  # (1, H, W, 4) where 4 is RGBA
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.savefig('figures/rendered_image.png', bbox_inches='tight', pad_inches=0)
plt.close()


# 4

# get depth map
fragments = rasertizer(mesh)
depth_map = fragments.zbuf[0, ..., 0]  # (H, W)

sorted_depth_map = torch.sort(depth_map.flatten())[0]  # sort depth values
print('unsorted depth map first 10: ', depth_map.flatten()[:10])  # print first 10
print('sorted_depth_map first 10: ', sorted_depth_map[:10])  # print first 10
print('sorted_depth_map  last 10: ', sorted_depth_map[-10:])  # print last 10

# normalize all non-negative depth values to [0, 1]
positive_mask = depth_map >= 0
depth_map[positive_mask] = (depth_map[positive_mask] - depth_map[positive_mask].min()) / (depth_map[positive_mask].max() - depth_map[positive_mask].min())  # normalize to [0, 1]
depth_map[~positive_mask] = 0  # set negative values to 0

plt.figure(figsize=(10, 10))
plt.imshow(depth_map.cpu().numpy(), cmap='gray')
plt.axis("off")
plt.savefig('figures/depth_map.png', bbox_inches='tight', pad_inches=0)
plt.close()