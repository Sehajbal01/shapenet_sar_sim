from utils import savefig
import matplotlib.pyplot as plt
import tqdm
import os
import imageio
import sys
import torch
from PIL import Image
import numpy as np
import math
from utils import get_next_path, extract_pose_info
from my_ray_tracer.core.scene import Scene
from my_ray_tracer.camera.orthographic import OrthographicCamera



def accumulate_scatters(target_poses, z_near, z_far, object_filename,
               azimuth_spread=15, n_pulses=30, n_rays_per_side=128,
               alpha_1=1.0, alpha_2=0.0, use_ground=True, debug_gif=False, num_bounces=1):
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
        use_ground (bool): whether to use the ground plane for rendering
        debug_gif (bool): whether to save a gif of the depth and energy images
        num_bounces (int): number of ray bounces to simulate

    outputs:
        range (T,P,R): the range of all the rays
        energy (T,P,R): the simulated energy of all the rays

    '''
    device = target_poses.device
    T = target_poses.shape[0]  # no. of camera views
    P = n_pulses               # no. of pulses per view
    half_side_len = abs(z_far - z_near) / 2

    # Pull out camera positions info
    _, _, _, _, cam_distance, cam_elevation, cam_azimuth = extract_pose_info(target_poses)
    #           (T,)          (T,)           (T,)
    print('Camera azimuth:   ', cam_azimuth)
    print('Camera elevation: ', cam_elevation)
    print('Camera distance:  ', cam_distance)

    # Spread the pulses across a small range of azimuth angles
    azimuth_offsets = torch.linspace(-azimuth_spread / 2, azimuth_spread / 2, P, device=device) # (P,)
    azimuth = cam_azimuth.reshape(T, 1) + azimuth_offsets.reshape(1, P) # (T,P)

    scene = Scene(
        obj_filename=object_filename,
        device=device,
    )  # will automatically build octree for this mesh

    # add a ground if desired to the mesh
    if use_ground:
        scene.add_ground()

    # loop over each pulse and compute the depth map and surface normal
    scatter_ranges = []
    scatter_energies = []
    for t in range(T):  # for each camera
        scatter_ranges.append([])
        scatter_energies.append([])

        # construct P number of cameras due to azimuth spread
        cameras = []
        for p in range(P):  # for each pulse
            elevation = cam_elevation[t] / 180 * torch.pi  # in radians now
            azimuth_ = azimuth[t][p] / 180 * torch.pi  # in radians now
            position_vector = torch.tensor([
                torch.cos(elevation) * torch.sin(azimuth_),
                torch.sin(elevation),
                torch.cos(elevation) * torch.cos(azimuth_)
            ], device=device)
            position_vector = position_vector / torch.norm(position_vector) * cam_distance[t]
            direction_vector = torch.tensor([0, 0, 0], device=device) - position_vector
            direction_vector = direction_vector / torch.norm(direction_vector)
            ortho_cam = OrthographicCamera(
                position_vector.cpu(),  # position
                direction_vector.cpu(),  # direction
                half_side_len * 2,  # sensor width in world space
                half_side_len * 2,  # sensor height in world space
                n_rays_per_side,  # number of rays to shoot in width dimension
                n_rays_per_side,  # number of rays to shoot in height dimension
            )
            cameras.append(ortho_cam)
        
        # # trace rays in parallel
        # with torch.no_grad():
        #     energy_range_values = scene.get_energy_range_values(cameras, num_bounces=num_bounces)

        # trace rays somewhat in parallel
        with torch.no_grad():
            num_cams_at_once = int(30 / num_bounces)
            energy_range_values = []
            for i in range(0, len(cameras), num_cams_at_once):
                energy_range_values.extend(scene.get_energy_range_values(cameras[i:i+num_cams_at_once], num_bounces=num_bounces))

        if debug_gif:
            os.makedirs('figures/tmp', exist_ok=True)
            # depth and diffuse images
            for p in range(P):
                depth, diffuse = scene.get_depth_and_diffuse(cameras[p])
                # concatenate along the width dimension
                dm_e_im = np.concatenate((depth, diffuse), axis=1)  # (h, 2w)
                path = get_next_path(f'figures/tmp/depth_energy.png')
                imageio.imwrite(path, dm_e_im)
            # range energy plots
            e_r_values = energy_range_values[0]  # list[(n, 2)]  # just the first camera for now
            for i in range(len(e_r_values)):  # generate a plot for each bounce
                xy = e_r_values[i].cpu().numpy()
                plt.scatter(xy[:, 0], xy[:, 1], s=1)
                plt.xlabel("Range")
                plt.ylabel("Energy")
                plt.xlim(0, 10)
                plt.ylim(0, 1)
                plt.title(f"Energy vs Range Plot for Bounce {i}")
                plt.savefig(os.path.join("figures", "tmp", f"energy_range_bounce_{i}.png"))
                plt.close()

        # save the energy range values
        energy_range_values = [torch.cat(e_r_values, dim=0) for e_r_values in energy_range_values]  # join plots for different number of bounces. (list[r, 2])
        
        max_R = max([e_r_values.shape[0] for e_r_values in energy_range_values])
        scatter_ranges[t] = torch.zeros((P, max_R), device=device)  # different pulses will have different number of returned hits, pad with zero
        scatter_energies[t] = torch.zeros((P, max_R), device=device)
        # fill in the values
        for p in range(P):
            e_r_values = energy_range_values[p]
            scatter_ranges[t][p, :e_r_values.shape[0]] = e_r_values[:, 0]
            scatter_energies[t][p, :e_r_values.shape[0]] = e_r_values[:, 1]

    scatter_ranges = torch.stack(scatter_ranges, dim=0)  # (T, P, R)
    scatter_energies = torch.stack(scatter_energies, dim=0)  # (T, P, R)

    # tile elevation and distance to match the shape of azimuth
    elevation = torch.tile(cam_elevation.reshape(T, 1), (1, P))  # (T, P)
    distance  = torch.tile(cam_distance.reshape(T, 1), (1, P))  # (T, P)

    return scatter_ranges, scatter_energies, azimuth, elevation, distance, cam_azimuth, cam_distance
    #      (T, P, R)       (T, P, R)         (T, P)   (T, P)     (T, P)    (T,)         (T,)



