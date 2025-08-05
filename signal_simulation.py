import os
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
import math
from utils import get_next_path, extract_pose_info



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

    # Pull out camera positions info # TODO: this is probably not consistent with the pytorch3d coordinate system
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

        bin_size=0,  # or set to a small value
        max_faces_per_bin=100000  # try increasing from the default (e.g., 10000)
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

                # save current image to a file in a tmp folder in figures
                if not os.path.exists('figures/tmp'):
                    os.makedirs('figures/tmp')
                path = get_next_path(f'figures/tmp/depth_energy.png')
                imageio.imwrite(path, dm_e_im)

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

    return scatter_ranges, scatter_energies, azimuth, elevation, distance, forward_vectors, cam_azimuth, cam_distance
    #      (T, P, R)       (T, P, R)         (T, P)   (T, P)     (T, P)    (T, P, 3)        (T,)         (T,)



def interpolate_signal(scatter_z, scatter_e, z_near, z_far,
        spatial_bw = 20, spatial_fs = 20,
        batch_size = None,
):
    """
    Simulates the received signal for the SAR algorithm given energy-range scatter.
    z is range. Z is the number of samples in the output signal.

    Inputs:
        scatter_z (...,R): the range of each scatter point
        scatter_e (...,R): the energy of each scatter point
        z_near (float): z near to use when rendering
        z_far (float): z far to use when rendering
        batch_size (int): number of signals to process in a batch, None means no batching

    Returns:
        received_signal (tensor): simulated received signal .shape=(..., Z)
    """

    # reshaping
    R = scatter_z.shape[-1]  # number of scatter points
    shape_prefix = scatter_z.shape[:-1]  # shape before the last dimension
    N = np.prod(shape_prefix)
    assert(len(scatter_z.shape) > 1), "scatter_z should have at least 2 dimensions, but got %d" % len(scatter_z.shape)
    assert(scatter_z.shape == scatter_e.shape), "scatter_z and scatter_e should have the same shape, but got %s and %s" % (scatter_z.shape, scatter_e.shape)

    # calculate the center of each spatial sample
    device = scatter_z.device
    first_z = int(math.ceil(z_near*spatial_fs))
    last_z =  int(math.floor(z_far*spatial_fs))
    sample_z = torch.arange(first_z, last_z+1, device=device, dtype=scatter_z.dtype)/spatial_fs # (Z,)
    Z = len(sample_z)

    # calculate the received signal for each pulse
    # received_signal = sum_over_R_scatters( scatter_e * torch.sinc( spatial_bw * (scatter_z - sample_z) ) )
    # When we do the broadcasting we want the shape to be (N,R,Z) before the sum, then sum over the R dimension
    if batch_size is None:
        received_signal = torch.sum(
            scatter_e.reshape(N,R,1) * \
            torch.sinc( spatial_bw * \
                        (scatter_z.reshape(N,R,1) - sample_z.reshape(1,1,Z))
                      ),
            dim=1
        ) # (N, Z)

    # calculate the received signal in batches to save memory
    else:
        received_signal = []

        reshaped_scatter_e = scatter_e.reshape(N,R,1)
        reshaped_scatter_z = scatter_z.reshape(N,R,1)
        reshapes_sample_z  = sample_z.reshape(1,1,Z)

        start = 0
        while start < N:
            end = min(start + batch_size, N)
            received_signal.append(
                torch.sum( reshaped_scatter_e[start:end] * \
                           torch.sinc( spatial_bw * \
                                       (reshaped_scatter_z[start:end] - reshapes_sample_z)
                                     ),
                           dim=2
            ))  # (N', Z)
            start = end

        received_signal = torch.cat(received_signal, dim=1) # (N, Z)

    # return stuff
    received_signal = received_signal.reshape(*shape_prefix, Z)  # (..., Z)
    return received_signal, sample_z



def resample_signal(self, signal, out_samples, dim = -1):
    """
    Resample the signal to the specified number of output samples.
    Uses linear interpolation to resample the signal.
    
    Args:
        signal (tensor): input signal to resample .shape=(..., in_samples, ...)
        out_samples (int): number of output samples to resample to
        dim (int): dimension along which to resample the signal
    
    Returns:
        resampled_signal (tensor): resampled signal .shape=(..., out_samples, ...)
    """
    # get shape information
    in_samples = signal.shape[dim]
    if in_samples == out_samples: # no need to resample when the number of samples is the same
        return signal
    assert(out_samples > 1), "out_samples should be greater than 1, but got %d" % out_samples
    in_sample_z = torch.arange(in_samples, device=signal.device, dtype=signal.dtype)  # (in_samples,)
    out_sample_z = torch.linspace(0, in_samples-1, out_samples, device=signal.device, dtype=signal.dtype)  # (out_samples,)
    shape_prefix = list(signal.shape[:dim])  # shape before the resampling dimension
    shape_suffix = list(signal.shape[dim+1:])  # shape after the resampling dimension
    n_prefix = np.prod(shape_prefix) if shape_prefix else 1  # number of elements before the resampling dimension
    n_suffix = np.prod(shape_suffix) if shape_suffix else 1  # number of elements after the resampling dimension
    # resample the signal using sinc interpolation
    # equation: resampled_signal = sum_over_inputs(torch.sinc(in_sample_z - out_sample_z) * signal)
    resampled_signal = torch.sum(
        #          (1, in_samples)                     (out_samples, 1)
        torch.sinc(in_sample_z.reshape(1,-1) - out_sample_z.reshape(-1, 1)).reshape(1,out_samples, in_samples, 1) * \
        signal.reshape(n_prefix, 1, in_samples, n_suffix),
        dim=2
    ) # (n_prefix, out_samples, n_suffix)
    
    # reshape and return
    resampled_signal = resampled_signal.reshape(shape_prefix + [out_samples] + shape_suffix)  # (shape_prefix, out_samples, shape_suffix)
    return resampled_signal  # (shape_prefix, out_samples, shape_suffix)