import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings, 
    MeshRasterizer,  
)

def extract_pose_info(target_poses):
    '''
    Extracts camera position and orientation from the target poses.
    inputs:
        target_poses (T,4,4): the camera poses in world coordinates
    outputs:
        see the return statement below
    '''
    cam_center    = target_poses[:, :3, 3] # (T,3)
    cam_right     = target_poses[:, :3, 0]  # (T,3)
    cam_up        = target_poses[:, :3, 1]     # (T,3)
    cam_forward   = target_poses[:, :3, 2] # (T,3)
    cam_distance  = torch.norm(cam_center, dim=-1, keepdim=True) # (T,)
    cam_elevation = torch.asin(cam_center[:, 2:3] / cam_distance) # (T,)
    cam_azimuth   = torch.acos(cam_center[:, 0:1] / (cam_distance * torch.cos(cam_elevation))) # (T,)
    cam_azimuth   = torch.where(cam_center[:, 1:2] < 0, 2 * torch.pi - cam_azimuth, cam_azimuth) # (T,)
    return cam_center, cam_right, cam_up, cam_forward, cam_distance, cam_elevation, cam_azimuth

def accumulate_scatters(target_poses, z_near, z_far, object_filename,
               azimuth_spread=15, n_pulses=30, n_rays_per_side=4):
    '''
    returns the energy and range for a bunch of rays for each pulse

    inputs:
        target_poses (T,4,4): the rgb pose for which we want to get a sar image from
        z_near (int): near distance of the obj
        z_far (int): far distance of the obj
        object_filename (str): path to the .obj file
        azimuth_spread (float): the range of azimuth angles for sar rendering
        n_pulses (int): the number of pulses for sar rendering
        n_rays_per_side (int): the number of rays on each side of the triangle, yielding num_ray_side**2 total rays

    outputs:
        range (T,P,R'): the range of all the rays that hit the object
        energy (T,P,R'): the simulated energy of all the rays that hit the object

    '''
    device = target_poses.device
    T = target_poses.shape[0]  # no. of camera views
    P = n_pulses               # no. of pulses per view

    # Pull out camera positions info
    cam_center, _, _, _, cam_distance, cam_elevation, cam_azimuth = extract_pose_info(target_poses)

    # Spread the pulses across a small range of azimuth angles
    azimuth_offsets = torch.linspace(-azimuth_spread / 2, azimuth_spread / 2, P, device=device) * torch.pi / 180 # (P,)

    # Side length of the raycasting plane
    side_len = abs(z_far - z_near)

    # get the azimuth and elevation for each pulse
    azimuth = cam_azimuth.reshape(T, 1) + azimuth_offsets.reshape(1, P) # (T,P)
    elevation = torch.tile(cam_elevation.reshape(T, 1), (1, P)) # (T,P)

    # load mesh
    mesh = load_objs_as_meshes([object_filename], device=device)

    # loop over each pulse and render the rays
    scatter_ranges = []
    scatter_energies = []
    for t in range(T):
        for p in range(P):










obj_filename = '/workspace/data/srncars/02958343/7dac31838b627748eb631ba05bd8dfe/models/model_normalized.obj'
device = 'cuda'

mesh = load_objs_as_meshes([obj_filename], device=device)

# Set up the camera
R, T = look_at_view_transform(1.8, 45, 45) # distance, elevation, azimuth
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