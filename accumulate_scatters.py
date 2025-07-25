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
               azimuth_spread=15, n_pulses=30, n_rays_per_side=128,
               alpha_1=0.9, alpha_2=0.1):
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
        alpha_1 (float): scaling factor for the energy return
        alpha_2 (float): offset for the energy return

    outputs:
        range (T,P,R'): the range of all the rays that hit the object
        energy (T,P,R'): the simulated energy of all the rays that hit the object

    '''
    device = target_poses.device
    T = target_poses.shape[0]  # no. of camera views
    P = n_pulses               # no. of pulses per view
    half_side_len = abs(z_far - z_near) / 2

    # Pull out camera positions info
    _, _, _, _, cam_distance, cam_elevation, cam_azimuth = extract_pose_info(target_poses)
    #                 (T,)         (T,)           (T,)

    # Spread the pulses across a small range of azimuth angles
    azimuth_offsets = torch.linspace(-azimuth_spread / 2, azimuth_spread / 2, P, device=device) * torch.pi / 180 # (P,)
    azimuth = cam_azimuth.reshape(T, 1) + azimuth_offsets.reshape(1, P) # (T,P)

    # prepare pytorch3d
    mesh = load_objs_as_meshes([object_filename], device=device)
    raster_settings = RasterizationSettings(
        image_size=n_rays_per_side, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # loop over each pulse and compute the depth map and surface normal
    scatter_ranges = []
    scatter_energies = []
    for t in range(T):
        scatter_ranges.append([])
        scatter_energies.append([])
        for p in range(P):

            # get the depth map and surface normal for each pulse
            R, T = look_at_view_transform(cam_distance[t], cam_elevation[t], azimuth[t, p]) # distance, elevation, azimuth
            cameras = FoVOrthographicCameras(device=device, R=R, T=T, 
                                         min_x = -half_side_len, max_x = half_side_len,
                                         min_y = -half_side_len, max_y = half_side_len,
                                         )
            rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            )
            fragments = rasterizer(mesh)
            depth_map  = fragments.zbuf[0, ..., 0]    # (r, r) # missed rays are -1.0
            normal_map = fragments.normals[0, ..., 0] # (r, r, 3)

            # compute returned energy (cosine similarity between ray direction and surface normal * alpha_1 + alpha_2)
            hit = depth_map >= 0  # (r, r)
            ray_directions = torch.stack([
                torch.cos(azimuth[t, p]) * torch.cos(cam_elevation[t]),
                torch.sin(azimuth[t, p]) * torch.cos(cam_elevation[t]),
                                           torch.sin(cam_elevation[t]),
            ], dim=-1).to(device) # (3,)
            normal_map[hit] = torch.abs(torch.sum(normal_map[hit] * ray_directions, dim=-1)) * alpha_1 + alpha_2  # (r, r)

            # finalize the range and energy
            depth_map[~hit]  = 0.0  # set missed rays to 0
            normal_map[~hit] = 0.0  # set missed rays to 0
            scatter_ranges[t].append(depth_map.reshape(-1))  # (R,)
            scatter_energies[t].append(normal_map.reshape(-1))  # (R)

        # stack the results
        scatter_ranges[t] = torch.stack(scatter_ranges[t], dim=0)  # (P, R)
        scatter_energies[t] = torch.stack(scatter_energies[t], dim=0)  # (P, R)
    scatter_ranges = torch.stack(scatter_ranges, dim=0)  # (T, P, R)
    scatter_energies = torch.stack(scatter_energies, dim=0)  # (T, P, R)

    # tile elevation and distance to match the shape of azimuth
    elevation = torch.tile(cam_elevation.reshape(T, 1), (1, P))  # (T, P)
    distance  = torch.tile( cam_distance.reshape(T, 1), (1, P))  # (T, P)

    return scatter_ranges, scatter_energies, azimuth, elevation, distance
    #           (T, P, R)   (T, P, R)        (T, P)    (T, P)      (T, P)


