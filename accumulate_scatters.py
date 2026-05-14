import torch
import time
import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRasterizer,
)
from utils import cartesian_to_spherical, dot_product, directional_scatter_polynomial_alpha5, get_next_path
from ray_tracer_v2 import ray_trace, build_octree

def accumulate_scatters(target_poses, 
                        mesh, face_normals, material_properties,
                        trajectory,
                        wavelength=None,
                        debug_gif=False,
                        grid_width=1, grid_height=1,
                        n_ray_width=1, n_ray_height=1,
                        num_bounce = 1,
                        second_bounce_batch_size = 2**100,
                    ):
    '''
    returns the energy and range for a bunch of rays for each pulse

    inputs:
        target_poses (T,4,4): the rgb pose for which we want to get a sar image from
        mesh (obj): pytorch3d mesh object of the 3d model
        face_normals (F,3): the normal vector of each face on the mesh
        material_properties (F,5): the r,a,i,d,s of each face of the mesh
        trajectory (T,P,3): the locations of the sensor for each pulse for each target scene
        wavelength (float): the wavelength of the radar signal, if none, there will be no complex value in the energy
        debug_gif (bool): whether to save a gif of the depth and energy images
        grid_width/height (float): the size of the ray grid for the orthonormal camera
        n_ray_width/height (int): the number of rays on the ray grid along the width and height axis.

    outputs:
        range (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only)
        energy (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only)

    '''
    device = target_poses.device
    T = target_poses.shape[0]  # no. of camera views
    P = trajectory.shape[1]    # no. of pulses per view

    def sync_time():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter()

    t_overall_start = sync_time()
    t_bounce1_total = 0.0
    t_multibounce_total = 0.0

    # Build octree once for the whole simulation run (amortised over all pulses).
    # Only needed for multi-bounce; skipped when num_bounce == 1.
    octree = build_octree(mesh) if num_bounce >= 2 else None

    # prepare rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(n_ray_height, n_ray_width), 
        blur_radius=0.0, 
        faces_per_pixel=1, 

        bin_size=0,  # or set to a small value
        max_faces_per_bin=100000  # try increasing from the default (e.g., 10000)
    )

    # loop over each pulse and compute the depth map and surface normal
    scatter_ranges = []
    scatter_energies = []
    dm_e_images = []  # to store depth and energy images
    for t in range(T):
        scatter_ranges.append([])
        scatter_energies.append([])
        for p in range(P):

            t_b1_start = sync_time()

            # perform rasterization to find where the rays hit the mesh
            cam_azimuth_deg, cam_elevation_deg, cam_distance = cartesian_to_spherical(trajectory[t,p])

            rotation, translation = look_at_view_transform(
                cam_distance,
                cam_elevation_deg,
                cam_azimuth_deg+90,
                device=device)
            cameras = FoVOrthographicCameras(
                device = device, R = rotation, T = translation, 
                min_x = -grid_width /2, max_x = grid_width /2,
                min_y = -grid_height/2, max_y = grid_height/2,
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
            
            # create normal map by indexing face normals with face IDs
            valid_face_ids = face_ids[hit] # (R',)

            # ray direction is the same for all rays because we are using orthographic projection, so we can simply grab the forward vector from the rotation matrix
            forward_vector = rotation[0,:,2] # (3,)
            energy_map = torch.zeros(n_ray_height, n_ray_width, device=device)  # (r, r)

            # calculate returned energy
            s = material_properties[valid_face_ids,4] 
            i = material_properties[valid_face_ids,2]
            d = material_properties[valid_face_ids,3]
            alpha = 5
            n = face_normals[valid_face_ids] # 
            u_in = forward_vector # (3,)
            cos_theta_over_2 = torch.abs(dot_product(n, u_in))
            energy_map[hit] = s * (
                (i*cos_theta_over_2**alpha) / \
                (directional_scatter_polynomial_alpha5(cos_theta_over_2)) + \
                d/2/np.pi
            )

            # produce a frame of the depth and energy maps
            if debug_gif:
                masked_dm = depth_map[hit]
                masked_dm = masked_dm - masked_dm.min()  # shift to start from 0
                masked_dm = masked_dm / masked_dm.max()  # normalize to [0, 1]
                masked_dm = 1 - masked_dm  # invert the depth map
                dm_im = torch.zeros((n_ray_height, n_ray_width), device=device)  # (r, r)
                dm_im[hit] = masked_dm  # apply the mask

                masked_e = energy_map[hit]
                masked_e = masked_e - masked_e.min()  # shift to start from 0
                masked_e = masked_e / masked_e.max()  # normalize to [0,1]
                e_im = torch.zeros((n_ray_height, n_ray_width), device=device)
                e_im[hit] = masked_e  # apply the mask

                e_im = e_im.cpu().numpy()  # convert to numpy for saving
                dm_im = dm_im.cpu().numpy()  # convert to numpy for saving
                dm_e_im = np.concatenate((dm_im, e_im), axis=1)  # concatenate depth and energy maps horizontally
                dm_e_im = (dm_e_im * 255).astype(np.uint8)  # scale to [0, 255] for saving

                dm_e_images.append(dm_e_im)  # store the depth and energy image

                # save current image to a file in a tmp folder in figures to be made into a gif later
                if not os.path.exists('figures/tmp'):
                    os.makedirs('figures/tmp')
                path = get_next_path(f'figures/tmp/depth_energy.png')
                imageio.imwrite(path, dm_e_im)

            # finalize the range and energy — keep only hit rays
            scatter_ranges[t].append(2 * depth_map[hit])   # (R',)
            scatter_energies[t].append(energy_map[hit])     # (R',)

            if debug_gif and t == 0:
                tmp1 = scatter_ranges[t][-1].cpu().numpy()
                tmp2 = scatter_energies[t][-1].cpu().numpy()
                plt.scatter(tmp1, tmp2, s=1)
                plt.xlabel("Range")
                plt.ylabel("Energy")
                plt.xlim(0, 6)
                plt.ylim(-0.5, 1.5)
                plt.title(f"Energy vs Range Plot for Bounce 0 for Pulse {p}")
                plt.savefig(os.path.join("figures", "tmp", f"energy_range_bounce_0_pulse_{p}.png"))
                plt.close()

            t_bounce1_total += sync_time() - t_b1_start

            # calculate additional bounces via ray tracing
            if num_bounce >= 2:
                t_mb_start = sync_time()

                # set up geometry to compute first-bounce hit positions
                right_vector = rotation[0, :, 0]  # (3,)
                up_vector    = rotation[0, :, 1]  # (3,)
                x_offsets = torch.linspace(-grid_width/2,  grid_width/2,  n_ray_width,  device=device)  # (W,)
                y_offsets = torch.linspace(-grid_height/2, grid_height/2, n_ray_height, device=device)  # (H,)
                grid_y, grid_x = torch.meshgrid(y_offsets, x_offsets, indexing='ij')  # (H, W)
                first_bounce_origins = (trajectory[t, p].reshape(1, 1, 3)
                                        + grid_x.unsqueeze(-1) * right_vector
                                        + grid_y.unsqueeze(-1) * up_vector)  # (H, W, 3)

                # initial state for the bounce loop
                # prev_origins/directions: the rays leaving the first-bounce surface
                # depth_hit1: depth from sensor to the first-bounce hit (carried along as rays are filtered)
                # cumulative_legs: sum of ray-trace leg lengths accumulated across bounces
                n_hit = face_normals[valid_face_ids]  # (R', 3)
                prev_origins    = first_bounce_origins[hit] + depth_map[hit].unsqueeze(-1) * forward_vector  # (R', 3)
                prev_directions = forward_vector - 2 * dot_product(forward_vector, n_hit, keepdim=True) * n_hit  # (R', 3)
                depth_hit1      = depth_map[hit]  # (R',)
                cumulative_legs = torch.zeros(prev_origins.shape[0], device=device)  # (R',)

                for b in range(2, num_bounce + 1):
                    hit_indecies, distance = ray_trace(prev_origins, prev_directions, mesh, face_normals, octree=octree, batch_size=second_bounce_batch_size)

                    hit_b = distance >= 0  # (H_prev,) boolean
                    if not hit_b.any():
                        break

                    # filter all state to only the rays that hit
                    prev_origins    = prev_origins[hit_b]
                    prev_directions = prev_directions[hit_b]
                    distance        = distance[hit_b]
                    hit_indecies    = hit_indecies[hit_b]
                    depth_hit1      = depth_hit1[hit_b]
                    cumulative_legs = cumulative_legs[hit_b] + distance

                    hit_b_pos = prev_origins + distance.unsqueeze(-1) * prev_directions  # (H, 3)

                    # calculate returned energy
                    s = material_properties[hit_indecies, 4]
                    i = material_properties[hit_indecies, 2]
                    d = material_properties[hit_indecies, 3]
                    n = face_normals[hit_indecies]  # (H, 3)
                    cos_theta_over_2 = torch.abs(dot_product(n, prev_directions))
                    energy_b = s * (
                        (i * cos_theta_over_2**5) /
                        directional_scatter_polynomial_alpha5(cos_theta_over_2) +
                        d / 2 / np.pi
                    )  # (H,)

                    # round-trip range: sensor→hit1 + all ray-trace legs + projection of final hit back onto radar axis
                    distance_to_sensor_plane = dot_product(hit_b_pos - trajectory[t, p], forward_vector)  # (H,)
                    total_range = depth_hit1 + cumulative_legs + distance_to_sensor_plane  # (H,)

                    # append scatter contributions for this bounce (no padding — ragged per pulse)
                    scatter_ranges[t][-1]   = torch.cat((scatter_ranges[t][-1],   total_range))
                    scatter_energies[t][-1] = torch.cat((scatter_energies[t][-1], energy_b))

                    # reflect directions off current surface for the next bounce
                    prev_directions = prev_directions - 2 * dot_product(prev_directions, n, keepdim=True) * n
                    prev_origins    = hit_b_pos

                t_multibounce_total += sync_time() - t_mb_start

        # leave as list — each pulse has a different number of hit rays
        pass

    t_overall = sync_time() - t_overall_start
    print(f"accumulate_scatters: overall={t_overall:.3f}s  bounce1={t_bounce1_total:.3f}s  multibounce={t_multibounce_total:.3f}s")

    # apply complex value to the energy according to wavelength
    if wavelength is not None:
        for t in range(T):
            for p in range(P):
                scatter_energies[t][p] = scatter_energies[t][p] * torch.exp(
                    1j * 2 * np.pi / wavelength * scatter_ranges[t][p]
                )

    return scatter_ranges, scatter_energies
    #      list[T][P] of 1-D tensors (R' hit rays, varies per pulse)
