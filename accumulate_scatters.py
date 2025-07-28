import imageio
import sys
import torch
from PIL import Image
import numpy as np
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
    cam_distance  = torch.norm(cam_center, dim=-1) # (T,)
    cam_elevation = torch.asin(cam_center[:, 2] / cam_distance) # (T,)
    cam_azimuth   = torch.acos(cam_center[:, 0] / (cam_distance * torch.cos(cam_elevation))) # (T,)
    cam_azimuth   = torch.where(torch.isnan(cam_azimuth), torch.zeros_like(cam_azimuth), cam_azimuth)  # handle NaN values
    cam_azimuth   = torch.where(cam_center[:, 1] < 0, 2 * torch.pi - cam_azimuth, cam_azimuth) # (T,)
    return cam_center, cam_right, cam_up, cam_forward, cam_distance, cam_elevation, cam_azimuth

def accumulate_scatters(target_poses, z_near, z_far, object_filename,
               azimuth_spread=15, n_pulses=30, n_rays_per_side=128,
               alpha_1=1.0, alpha_2=0.0, debug_gif=False):
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
        debug_gif (bool): whether to save a gif of the depth and energy images

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
    cam_elevation = cam_elevation * 180 / np.pi
    cam_azimuth   = cam_azimuth   * 180 / np.pi

    # Spread the pulses across a small range of azimuth angles
    azimuth_offsets = torch.linspace(-azimuth_spread / 2, azimuth_spread / 2, P, device=device) # (P,)
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
    forward_vectors = []
    dm_e_images = []  # to store depth and energy images
    for t in range(T):
        scatter_ranges.append([])
        scatter_energies.append([])
        forward_vectors.append([])
        for p in range(P):

            # perform rasterization to find where the rays hit the mesh
            rotation, translation = look_at_view_transform(cam_distance[t], cam_elevation[t], azimuth[t, p],device=device) # distance, elevation, azimuth
            cameras = FoVOrthographicCameras(device=device, R=rotation, T=translation, 
                                         min_x = -half_side_len, max_x = half_side_len,
                                         min_y = -half_side_len, max_y = half_side_len,
                                         )
            rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            )
            fragments = rasterizer(mesh)

            # get depth map
            depth_map  = fragments.zbuf[0, ..., 0]    # (r, r) # missed rays are -1.0

            # compute surface normals from face indices and mesh vertices/faces
            face_ids = fragments.pix_to_face[0, ..., 0]  # (r, r) face indices
            hit = (depth_map >= 0) # (r, r) valid hits
            
            # get vertices and faces from mesh
            verts = mesh.verts_packed()  # (V, 3)
            faces = mesh.faces_packed()  # (F, 3)
            
            # compute face normals for all faces
            face_verts = verts[faces]  # (F, 3, 3)
            v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
            face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
            face_normals = torch.nn.functional.normalize(face_normals, dim=1)  # (F, 3)
            
            # create normal map by indexing face normals with face IDs
            valid_face_ids = face_ids[hit] # (R',)

            # compute returned energy (cosine similarity between ray direction and surface normal * alpha_1 + alpha_2)
            # ray direction is the same for all rays because we are using orthographic projection, so we can simply grab the forward vector from the rotation matrix
            forward_vector = rotation[0,:,2] # (3,)
            energy_map = torch.zeros(n_rays_per_side, n_rays_per_side, device=device)  # (r, r)
            energy_map[hit] = torch.abs(torch.sum(face_normals[valid_face_ids] * forward_vector, dim=-1)) * alpha_1 + alpha_2 # (r, r)

            # produce a frame of the depth and energy maps
            if debug_gif:
                masked_dm = depth_map[hit]
                masked_dm = masked_dm - masked_dm.min()  # shift to start from 0
                masked_dm = masked_dm / masked_dm.max()  # normalize to [0, 1]
                masked_dm = 1 - masked_dm  # invert the depth map
                dm_im = torch.zeros((n_rays_per_side, n_rays_per_side), device=device)  # (r, r)
                dm_im[hit] = masked_dm  # apply the mask

                masked_e = energy_map[hit]
                masked_e = masked_e - masked_e.min()  # shift to start from 0
                masked_e = masked_e / masked_e.max()  # normalize to [0,1]
                e_im = torch.zeros((n_rays_per_side, n_rays_per_side), device=device)
                e_im[hit] = masked_e  # apply the mask

                e_im = e_im.cpu().numpy()  # convert to numpy for saving
                dm_im = dm_im.cpu().numpy()  # convert to numpy for saving
                dm_e_im = np.concatenate((dm_im, e_im), axis=1)  # concatenate depth and energy maps horizontally
                dm_e_im = (dm_e_im * 255).astype(np.uint8)  # scale to [0, 255] for saving

                dm_e_images.append(dm_e_im)  # store the depth and energy image

            # finalize the range and energy
            depth_map[~hit]  = 0.0  # set missed rays to 0
            scatter_ranges[t].append(depth_map.reshape(-1))  # (R,)
            scatter_energies[t].append(energy_map.reshape(-1))  # (R,)
            forward_vectors[t].append(forward_vector)


        # stack the results
        scatter_ranges[t] = torch.stack(scatter_ranges[t], dim=0)  # (P, R)
        scatter_energies[t] = torch.stack(scatter_energies[t], dim=0)  # (P, R)
        forward_vectors[t] = torch.stack(forward_vectors[t], dim=0)  # (P, 3)
    scatter_ranges = torch.stack(scatter_ranges, dim=0)  # (T, P, R)
    scatter_energies = torch.stack(scatter_energies, dim=0)  # (T, P, R)
    forward_vectors = torch.stack(forward_vectors, dim=0)  # (T, P, 3)

    # tile elevation and distance to match the shape of azimuth
    elevation = torch.tile(cam_elevation.reshape(T, 1), (1, P))  # (T, P)
    distance  = torch.tile( cam_distance.reshape(T, 1), (1, P))  # (T, P)

    # make a gif of the dm_e images lasts 5 seconds
    if debug_gif and len(dm_e_images) > 0:
        dm_e_images = np.stack(dm_e_images, axis=0)  # (N, H, W)
        fps = dm_e_images.shape[0]/4.0
        imageio.mimsave('figures/depth_energy_images.gif', dm_e_images, fps=fps, format='GIF', loop=0)

    return scatter_ranges, scatter_energies, azimuth, elevation, distance, forward_vectors
    #      (T, P, R)       (T, P, R)         (T, P)   (T, P)     (T, P)    (T, P, 3)


