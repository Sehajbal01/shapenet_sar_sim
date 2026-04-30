from utils import savefig, correct_material_properties, dot_product, directional_scatter_polynomial_alpha5
import matplotlib.pyplot as plt
import tqdm
import os
import imageio
import sys
import torch
from PIL import Image
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings, 
    MeshRasterizer,  
)
import math
from utils import get_next_path, extract_pose_info, spherical_to_cartesian, generate_pose_mat, cartesian_to_spherical, gpu_mem
from ray_tracer.geometry.trimesh import TriMesh
from ray_tracer.accelerator.octree import Octree



def generate_trajectory(pose, trajectory_type='linear', n_pulses=100, azimuth_spread_deg=None, trajectory_noise_var=0, debug=False):
    '''
    Generate a trajectory for multiple poses based on a single pose. The pose is the middle of the trajectory.
    '''
    assert(azimuth_spread_deg >= 0  ), 'the azimuth cannot be negative'

    device = pose.device
    T = pose.shape[0]  # no. of camera views
    P = n_pulses               # no. of pulses per view

    # Pull out camera positions info # TODO: this is probably not consistent with the pytorch3d coordinate system
    cam_center, cam_right, cam_up, cam_forward, cam_distance, cam_elevation_deg, cam_azimuth_deg = extract_pose_info(pose)
    #  (T,3)       (T,3)      (T,3)    (T,3)        (T,)           (T,)              (T,)
   

    if trajectory_type == 'circular':
        # Spread the pulses across a small range of azimuth angles
        azimuth_offsets = torch.linspace(-azimuth_spread_deg / 2, azimuth_spread_deg / 2, P, device=device) # (P,)
        azimuth_deg = cam_azimuth_deg.reshape(T, 1) + azimuth_offsets.reshape(1, P) # (T,P)
        elevation_deg = cam_elevation_deg.reshape(T, 1).repeat(1, P) # (T,P)
        distance = cam_distance.reshape(T, 1).repeat(1, P) # (T,P)
        trajectory = spherical_to_cartesian(azimuth_deg, elevation_deg, distance) # (T,P,3)

    elif trajectory_type == 'linear':

        assert(azimuth_spread_deg < 180), 'with a linear trajectory, the azimuth cannot be greater than 180 degrees'
        mag_ground_pos = cam_center[:,:2].norm(dim=-1)  # (T,) magnitude of the ground position
        dist_from_center = mag_ground_pos * np.tan(azimuth_spread_deg/2 * np.pi/180)  # (T,) distance from the center position for the first and last pulse
        start_pos = cam_center - dist_from_center.reshape(T,1) * cam_right # (T,3) starting position for the first pulse

        trajectory = torch.linspace(0,1, P, device=device).reshape(1, P, 1) * \
                     cam_right.reshape(T, 1, 3) * \
                     (2*dist_from_center).reshape(T, 1, 1) + \
                     start_pos.reshape(T, 1, 3)
        # (T,P,3) trajectory of the camera for each pulse

    else:
        raise ValueError('trajectory_type should be either circular or linear, but got %s' % trajectory_type)

    if debug:
        # plot the trajectory for debugging
        plt.figure(figsize=(6,6))
        min_z = trajectory[:,:,2].min().item()
        max_z = trajectory[:,:,2].max().item()
        plt.scatter(trajectory[0,:,0].cpu().numpy(), trajectory[0,:,1].cpu().numpy())
        plt.title('Camera Trajectory\ntrajectory trajectory_type: %s\nazimuth spread: %d degrees\nz-range: %.2f - %.2f'%(trajectory_type, azimuth_spread_deg, min_z, max_z))
        plt.xlabel('X')
        plt.ylabel('Y')
        path = get_next_path('figures/camera_trajectory.png')
        savefig(path)

    # apply trajectory gaussian noise if desired
    if trajectory_noise_var > 0:
        noise = torch.randn_like(trajectory) * np.sqrt(trajectory_noise_var)
        true_trajectory = trajectory + noise
    else:
        true_trajectory = trajectory
    perceived_trajectory = trajectory
        
    return true_trajectory, perceived_trajectory, cam_azimuth_deg
        








def accumulate_scatters(target_poses,
                        mesh, face_normals, material_properties,
                        trajectory,
                        wavelength=None,
                        debug_gif=False,
                        grid_width=1, grid_height=1,
                        n_ray_width=1, n_ray_height=1,
                        num_bounce = 1,
                        first_bounce_batch_size = 1,
                        second_bounce_batch_size = 1,
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
        range (T,P,R): the range of all the rays
        energy (T,P,R): the simulated energy of all the rays

    '''
    device = target_poses.device
    T = target_poses.shape[0]  # no. of camera views
    P = trajectory.shape[1]    # no. of pulses per view

    mem = lambda: gpu_mem(device)
    # print(f"[accumulate_scatters START] {mem()}")

    # prepare rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(n_ray_height, n_ray_width), 
        blur_radius=0.0, 
        faces_per_pixel=1, 

        bin_size=0,  # or set to a small value
        max_faces_per_bin=100000  # try increasing from the default (e.g., 10000)
    )

    # build octree from PyTorch3D mesh for fast second-bounce intersection
    if num_bounce == 2:
        vp_all = mesh.verts_packed()
        fp_all = mesh.faces_packed()
        _tm = TriMesh()
        _tm.triangles_A      = vp_all[fp_all[:, 0]]
        _tm.triangles_B      = vp_all[fp_all[:, 1]]
        _tm.triangles_C      = vp_all[fp_all[:, 2]]
        _tm.triangles_edge1  = _tm.triangles_B - _tm.triangles_A
        _tm.triangles_edge2  = _tm.triangles_C - _tm.triangles_A
        _tm.triangles_normal = torch.linalg.cross(_tm.triangles_edge1, _tm.triangles_edge2, dim=1)
        _tm.triangles_normal = _tm.triangles_normal / torch.norm(_tm.triangles_normal, dim=1, keepdim=True)
        _tm.triangles_rsa    = torch.zeros((fp_all.shape[0], 3), device=device)
        octree_rt = Octree(max_depth=2, approx_trig_per_bbox=256, mesh=_tm, device=device)
        # print(f"[octree built] {mem()}")

    # loop over each pulse and compute the depth map and surface normal
    scatter_ranges = []
    scatter_energies = []
    scatter_ranges_2nd = []
    scatter_energies_2nd = []
    dm_e_images = []  # to store depth and energy images
    alpha = 5
    num_faces = face_normals.shape[0]  # faces in the original (non-extended) mesh
    t1_grand_total = 0.0
    t2_grand_total = 0.0
    for t in range(T):
        scatter_ranges.append([])
        scatter_energies.append([])
        scatter_ranges_2nd.append([])
        scatter_energies_2nd.append([])
        t1_total = 0.0
        second_bounce_data = [None] * P

        t1_start = torch.cuda.Event(enable_timing=True)
        t1_end   = torch.cuda.Event(enable_timing=True)

        for batch_start in range(0, P, first_bounce_batch_size):
            batch_end = min(batch_start + first_bounce_batch_size, P)
            B = batch_end - batch_start

            t1_start.record()

            # build batched cameras for this group of pulses
            rot_list, trans_list = [], []
            for p in range(batch_start, batch_end):
                az, el, dist = cartesian_to_spherical(trajectory[t, p])
                rot, trans = look_at_view_transform(dist, el, az + 90, device=device)
                rot_list.append(rot)    # (1, 3, 3)
                trans_list.append(trans)  # (1, 3)
            rotations_b    = torch.cat(rot_list,   dim=0)  # (B, 3, 3)
            translations_b = torch.cat(trans_list, dim=0)  # (B, 3)

            cameras = FoVOrthographicCameras(
                device=device, R=rotations_b, T=translations_b,
                min_x=-grid_width/2,  max_x=grid_width/2,
                min_y=-grid_height/2, max_y=grid_height/2,
            )
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(mesh.extend(B))
            print(f"  [1st-bounce t={t} pulses={batch_start}-{batch_end-1}] fragments.zbuf.shape={fragments.zbuf.shape}")

            for bi, p in enumerate(range(batch_start, batch_end)):
                rotation    = rotations_b[bi:bi+1]     # (1, 3, 3)
                translation = translations_b[bi:bi+1]  # (1, 3)

                depth_map = fragments.zbuf[bi, ..., 0]         # (H, W)
                face_ids  = fragments.pix_to_face[bi, ..., 0]  # (H, W)
                hit = (depth_map >= 0)
                valid_face_ids = face_ids[hit] - bi * num_faces  # offset for extended mesh

                forward_vector = rotation[0, :, 2]  # (3,)
                energy_map = torch.zeros(n_ray_height, n_ray_width, device=device)

                s = material_properties[valid_face_ids, 4]
                i = material_properties[valid_face_ids, 2]
                d = material_properties[valid_face_ids, 3]
                n = face_normals[valid_face_ids]
                n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
                u_in = forward_vector / torch.linalg.norm(forward_vector, dim=-1, keepdim=True)
                cos_theta_over_2 = torch.abs(dot_product(n, u_in))
                energy_map[hit] = s * (
                    (i * cos_theta_over_2**alpha) /
                    directional_scatter_polynomial_alpha5(cos_theta_over_2) +
                    d / 2 / np.pi
                )

                if debug_gif:
                    masked_dm = depth_map[hit]
                    masked_dm = masked_dm - masked_dm.min()
                    masked_dm = masked_dm / masked_dm.max()
                    masked_dm = 1 - masked_dm
                    dm_im = torch.zeros((n_ray_height, n_ray_width), device=device)
                    dm_im[hit] = masked_dm

                    masked_e = energy_map[hit]
                    masked_e = masked_e - masked_e.min()
                    masked_e = masked_e / masked_e.max()
                    e_im = torch.zeros((n_ray_height, n_ray_width), device=device)
                    e_im[hit] = masked_e

                    e_im   = e_im.cpu().numpy()
                    dm_im  = dm_im.cpu().numpy()
                    dm_e_im = np.concatenate((dm_im, e_im), axis=1)
                    dm_e_im = (dm_e_im * 255).astype(np.uint8)
                    dm_e_images.append(dm_e_im)

                    if not os.path.exists('figures/tmp'):
                        os.makedirs('figures/tmp')
                    path = get_next_path(f'figures/tmp/depth_energy.png')
                    imageio.imwrite(path, dm_e_im)

                depth_map[~hit] = 0.0
                scatter_ranges[t].append(2 * depth_map.reshape(-1))
                scatter_energies[t].append(energy_map.reshape(-1))

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

                if num_bounce == 2 and hit.any():
                    R0 = rotation[0]
                    T0 = translation[0]
                    fwd_w        = forward_vector / torch.linalg.norm(forward_vector)
                    cam_center_w = -(T0.unsqueeze(0) @ R0.T).squeeze(0)

                    x_cam = torch.linspace(-grid_width/2,  grid_width/2,  n_ray_width,  device=device)
                    y_cam = torch.linspace( grid_height/2, -grid_height/2, n_ray_height, device=device)
                    gy, gx = torch.meshgrid(y_cam, x_cam, indexing='ij')
                    hit_pos_w = (cam_center_w
                                 + gx.unsqueeze(-1) * R0[:, 0]
                                 + gy.unsqueeze(-1) * R0[:, 1]
                                 + depth_map.unsqueeze(-1) * fwd_w)

                    first_hit_flat   = hit.reshape(-1)
                    ro2              = hit_pos_w.reshape(-1, 3)[first_hit_flat]
                    first_hit_depths = depth_map.reshape(-1)[first_hit_flat]

                    n_r = n.clone()
                    n_r[(n_r * fwd_w).sum(-1) > 0] *= -1
                    dot_r = (fwd_w.unsqueeze(0) * n_r).sum(-1, keepdim=True)
                    rd2   = fwd_w.unsqueeze(0) - 2 * dot_r * n_r
                    rd2   = rd2 / torch.linalg.norm(rd2, dim=-1, keepdim=True)
                    ro2   = ro2 + 1e-4 * rd2

                    second_bounce_data[p] = dict(
                        ro2=ro2, rd2=rd2,
                        first_hit_flat=first_hit_flat,
                        first_hit_depths=first_hit_depths,
                        fwd_w=fwd_w, cam_center_w=cam_center_w,
                    )

            t1_end.record(); torch.cuda.synchronize(); t1_total += t1_start.elapsed_time(t1_end)

        # second bounce: second_bounce_batch_size pulses per octree call
        if num_bounce == 2:
            t2_ev_start = torch.cuda.Event(enable_timing=True)
            t2_ev_end   = torch.cuda.Event(enable_timing=True)
            t2_ev_start.record()

            for batch_start in range(0, P, second_bounce_batch_size):
                batch_end = min(batch_start + second_bounce_batch_size, P)
                batch_data = [second_bounce_data[p] for p in range(batch_start, batch_end)]
                valid_mask = [d is not None for d in batch_data]
                valid_data = [d for d in batch_data if d is not None]

                t_p_list = []
                f_p_list = []
                if valid_data:
                    batch_ro2 = torch.cat([d['ro2'] for d in valid_data], dim=0)
                    batch_rd2 = torch.cat([d['rd2'] for d in valid_data], dim=0)
                    print(f"  [2nd-bounce t={t} pulses={batch_start}-{batch_end-1}] batch_ro2.shape={batch_ro2.shape} ({len(valid_data)} valid pulses)")
                    t_p_all, f_p_all = octree_rt.intersect_rays(batch_ro2, batch_rd2)
                    sizes = [d['ro2'].shape[0] for d in valid_data]
                    t_p_list = list(torch.split(t_p_all, sizes))
                    f_p_list = list(torch.split(f_p_all, sizes))

                valid_idx = 0
                for i in range(batch_end - batch_start):
                    e2_full = torch.zeros(n_ray_height * n_ray_width, device=device)
                    r2_full = torch.zeros(n_ray_height * n_ray_width, device=device)

                    if valid_mask[i]:
                        data = batch_data[i]
                        t_p  = t_p_list[valid_idx]
                        f_p  = f_p_list[valid_idx]
                        valid_idx += 1
                        hit2 = t_p >= 0

                        if hit2.any():
                            vf2  = f_p[hit2]
                            d2   = t_p[hit2]
                            s2   = material_properties[vf2, 4]
                            i2   = material_properties[vf2, 2]
                            d2_m = material_properties[vf2, 3]
                            n_b2 = face_normals[vf2]
                            n_b2 = n_b2 / torch.linalg.norm(n_b2, dim=-1, keepdim=True)
                            u_b2 = data['rd2'][hit2]
                            cos2 = torch.abs(dot_product(n_b2, u_b2))
                            energy2 = s2 * (
                                (i2 * cos2**alpha) / directional_scatter_polynomial_alpha5(cos2) +
                                d2_m / 2 / np.pi
                            )
                            pos2      = data['ro2'][hit2] + d2.unsqueeze(-1) * data['rd2'][hit2]
                            dist_back = ((pos2 - data['cam_center_w']) * data['fwd_w']).sum(-1)
                            range2    = 2 * (data['first_hit_depths'][hit2] + d2 + dist_back)

                            idx_fh = data['first_hit_flat'].nonzero(as_tuple=True)[0]
                            idx_sh = idx_fh[hit2]
                            e2_full[idx_sh] = energy2
                            r2_full[idx_sh] = range2

                    scatter_energies_2nd[t].append(e2_full)
                    scatter_ranges_2nd[t].append(r2_full)

            t2_ev_end.record(); torch.cuda.synchronize()
            t2_total = t2_ev_start.elapsed_time(t2_ev_end) / 1000
        else:
            t2_total = 0.0

        t1_grand_total += t1_total / 1000
        t2_grand_total += t2_total
        # stack the results
        scatter_ranges[t] = torch.stack(scatter_ranges[t], dim=0)  # (P, R)
        scatter_energies[t] = torch.stack(scatter_energies[t], dim=0)  # (P, R)
    scatter_ranges = torch.stack(scatter_ranges, dim=0)  # (T, P, R)
    scatter_energies = torch.stack(scatter_energies, dim=0)  # (T, P, R)

    # stack on the second bounce if desired
    if num_bounce == 2:
        sr2 = torch.stack([torch.stack(sr, dim=0) for sr in scatter_ranges_2nd], dim=0)   # (T, P, R)
        se2 = torch.stack([torch.stack(se, dim=0) for se in scatter_energies_2nd], dim=0) # (T, P, R)
        scatter_ranges   = torch.cat([scatter_ranges,   sr2], dim=-1)  # (T, P, 2R)
        scatter_energies = torch.cat([scatter_energies, se2], dim=-1)  # (T, P, 2R)

    # apply complex value to the energy according to wavelength
    if wavelength is not None:
        scatter_energies = scatter_energies * torch.exp(1j * 2 * np.pi / wavelength * scatter_ranges)

    if num_bounce == 2:
        del octree_rt, _tm
        torch.cuda.empty_cache()

    print(f"[accumulate_scatters] 1st-bounce batch={first_bounce_batch_size}: {t1_grand_total:.2f}s  2nd-bounce batch={second_bounce_batch_size}: {t2_grand_total:.2f}s  total: {t1_grand_total+t2_grand_total:.2f}s  ({T}x{P} pulses, {n_ray_width}x{n_ray_height} rays)")
    return scatter_ranges, scatter_energies
    #      (T, P, R)       (T, P, R)         



def interpolate_signal(scatter_z, scatter_e,# range_near, range_far,
        region_radius, sensor_distance,
        spatial_bw = 20, spatial_fs = 20,
        batch_size = None, window_func = 'sinc',
        debug = False,
):
    """
    Simulates the received signal for the SAR algorithm given energy-range scatter.
    z is range. Z is the number of samples in the output signal.

    Inputs:
        scatter_z (...,R): the range of each scatter point
        scatter_e (...,R): the energy of each scatter point
        # range_near (float): z near to use when rendering
        # range_far (float): z far to use when rendering
        region_radius (float): the radius of the region to consider for output signal samples
        sensor_distance (...,): the distance from the sensor to the origin for each pulse
        spatial_bw (float): spatial bandwidth of the radar
        spatial_fs (float): spatial sampling frequency of the radar
        batch_size (int): number of signals to process in a batch, None means no batching
        window_func (str): window function to use ('sinc' or 'gaussian')

    Returns:
        signal (tensor): simulated received signal .shape=(..., Z)
    """

    device = scatter_z.device

    # reshaping
    R = scatter_z.shape[-1]  # number of scatter points
    shape_prefix = scatter_z.shape[:-1]  # shape before the last dimension
    N = np.prod(shape_prefix)
    assert(len(scatter_z.shape) > 1), "scatter_z should have at least 2 dimensions, but got %d" % len(scatter_z.shape)
    assert(scatter_z.shape == scatter_e.shape), "scatter_z and scatter_e should have the same shape, but got %s and %s" % (scatter_z.shape, scatter_e.shape)

    # make the window function
    if window_func == "sinc":
        window = lambda x: torch.sinc(spatial_bw * x)
    elif window_func == "gaussian":
        window = lambda x: torch.exp(-0.5 * (x * spatial_bw) ** 2)
    else:
        raise ValueError("window_func should be 'sinc' or 'gaussian', but got %s" % window_func)


    # calculate the center of each spatial sample
    # first_z = int(math.ceil(range_near*spatial_fs))
    # last_z =  int(math.floor(range_far*spatial_fs))
    # sample_z = torch.arange(first_z, last_z+1, device=device, dtype=scatter_z.dtype)/spatial_fs # (Z,)
    # Z = len(sample_z)
    # the new way
    Z = int(2*region_radius*spatial_fs) + 1
    sample_z = (torch.linspace(0,1, Z, device=device, dtype=scatter_z.dtype) - 0.5 ) * (Z-1)/spatial_fs # (Z,)
    sample_z = sensor_distance.reshape(N,1) + sample_z # (N, 1) + (Z,) -> (N, Z)

    # calculate the received signal for each pulse
    # signal = sum_over_R_scatters( scatter_e * torch.sinc( spatial_bw * (scatter_z - sample_z) ) )
    # When we do the broadcasting we want the shape to be (N,R,Z) before the sum, then sum over the R dimension
    if batch_size is None:
        signal = torch.sum(
            scatter_e.reshape(N,R,1) * \
            window(scatter_z.reshape(N,R,1) - sample_z.reshape(N,1,Z)),
            dim=1
        ) # (N, Z)

    # calculate the received signal in batches to save memory
    else:
        signal = []

        reshaped_scatter_e = scatter_e.reshape(N,R,1)
        reshaped_scatter_z = scatter_z.reshape(N,R,1)
        reshaped_sample_z  = sample_z.reshape(N,1,Z)

        start = 0
        while start < N:
            end = min(start + batch_size, N)
            signal.append(
                torch.sum( reshaped_scatter_e[start:end] * \
                           window(reshaped_scatter_z[start:end] - reshaped_sample_z),
                           dim=2
            ))  # (N', Z)
            start = end

        signal = torch.cat(signal, dim=1) # (N, Z)

    # # debugging by plotting the first signal
    # if debug:
    #     all_energies = scatter_e.reshape(N,R)
    #     all_ranges   = scatter_z.reshape(N,R)
    #     signals      = signal.reshape(N,Z)
    #     if signals.dtype.is_complex:
    #         signals = torch.abs(signals)
    #     if all_energies.dtype.is_complex:
    #         all_energies = torch.abs(all_energies)

    #     # plot the signal and scatters for every pulse
    #     sig_max = signals.max().item()
    #     sig_min = signals.min().item()
    #     energy_max = all_energies.max().item()
    #     energy_min = all_energies.min().item()
    #     p = N//2
    #     plt.figure(figsize=(12, 6))

    #     # plot the scatters
    #     plt.subplot(1, 2, 1)
    #     plt.scatter(all_ranges[p].cpu().numpy(),all_energies[p].cpu().numpy())
    #     plt.title('Scatters')
    #     plt.xlabel('Range')
    #     plt.ylabel('Energy')
    #     plt.xlim(-region_radius, region_radius)
    #     plt.ylim(energy_min, energy_max)

    #     # plot the signal
    #     plt.subplot(1, 2, 2)
    #     plt.plot(sample_z[p].cpu().numpy(), signals[p].cpu().numpy())
    #     plt.title('Signal')
    #     plt.xlabel('Range')
    #     plt.ylabel('Amplitude')
    #     plt.xlim(-region_radius, region_radius)
    #     plt.ylim(sig_min, sig_max)

    #     # # plot the interpolating sinc pulse function along with the scatters
    #     # window_range = torch.linspace(scatter_z.min(), scatter_z.max(), 10000, device=device, dtype=scatter_z.dtype)  # (1000,)
    #     # plt.subplot(1, 3, 3)
    #     # plt.scatter(all_ranges[p].cpu().numpy(),all_energies[p].cpu().numpy())
    #     # for sz in sample_z:
    #     #     window_pulse = window(window_range - sz)
    #     #     plt.plot(window_range.cpu().numpy(), window_pulse.cpu().numpy()*energy_max, color='orange')
    #     # plt.plot(sample_z.cpu().numpy(), (signals[p]/signals[p].max()).cpu().numpy(), color='red')
    #     # plt.title('Window Pulse')
    #     # plt.xlabel('Range')
    #     # plt.ylabel('Energy')
    #     # mid_z = (range_near + range_far) / 2
    #     # plt.xlim(mid_z - 5 / spatial_fs, mid_z + 5 / spatial_fs)
    #     # plt.ylim(window_pulse.min().cpu().numpy(), window_pulse.max().cpu().numpy())

    #     path = get_next_path('figures/scatters_signal_fs%d_bw%d.png'%(int(spatial_fs), int(spatial_bw)))
    #     savefig(path)
    #     print('Figure saved to %s' % path)

    # return stuff
    signal = signal.reshape(*shape_prefix, Z)  # (..., Z)
    sample_z = sample_z.reshape(*shape_prefix, Z)  # (..., Z)
    ray_normalized_signal = signal/R # normalize by the number of rays
    return ray_normalized_signal, sample_z



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




def apply_snr(signal, snr_db, dim=-1):
    """
    Apply a signal-to-noise ratio (SNR) to the input signal.

    Args:
        signal (torch.Tensor): The input signal tensor.
        snr_db (float): The desired SNR in decibels.
        dim (int): The dimension along which to compute the SNR.

    Returns:
        torch.Tensor: The signal with the applied SNR.
    """

    # N = Ni + Nr # real and imaginary parts are iid
    # SNR = Ps/Pn = E[|S|^2]/E[|N|^2]
    # E[|N|^2] = E[|S|^2]/SNR

    # E[|N|^2] = E[Nr^2 + Ni^2] = 2 * E[Nr^2]
    # E[|N|^2]/2 = E[Nr^2] = sigma^2
    # sigma = sqrt( E[|N|^2]/2 ) = sqrt( E[|S|^2]/(2*SNR) )

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Compute the signal power
    signal_power = torch.mean(signal.abs() ** 2, dim=dim, keepdim=True)
    noise_std_dev = torch.sqrt(signal_power / (2 * snr_linear)).cpu().numpy()

    # Generate noise
    noise = torch.tensor(
             np.random.normal(scale=noise_std_dev, size=signal.shape) + \
        1j * np.random.normal(scale=noise_std_dev, size=signal.shape)
    ).to(signal.device, dtype=signal.dtype)

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal


def load_mesh(  file_name,
                obj_raids = (1.0, 1.0, 0.9, 0.1, 1.0),
                make_ground = True,
                ground_below = True,
                ground_raids = (0.1, 0.1, 0.1, 0.9, 1.0),
                device = 'cuda',
                scale = None,
        ):  
    '''
    Load a mesh from an obj file and compute face normals.
    Inputs:
        file_name: str - path to the obj file
        make_ground: bool - whether to add a ground plane
        obj_raids: tuple - roughness, specular, ambient for the object material
        ground_raids: tuple - roughness, specular, ambient for the ground material
        device: str - device to load the mesh onto
    Outputs:
        mesh: Meshes - the loaded mesh with face normals
        face_normals: (F, 3) - the face normals
        raids: (F, 5) - the material properties for each face in Reflectivity, Scatter, Absorption
    '''
    # load verts and faces
    mesh = load_objs_as_meshes([file_name], device=device)
    verts = mesh.verts_packed()  # (V, 3)
    faces = mesh.faces_packed()  # (F, 3)

    # optional scaling
    if scale is not None:
        verts = verts * scale

    # set material properties for each face
    raids = torch.tensor(obj_raids, device=device, dtype=torch.float32).reshape(1, 5).repeat(faces.shape[0], 1)  # (F, 5)

    # add a ground if desired to the mesh
    if make_ground:
        ground_buffer = 0.001
        if ground_below:
            ground_y  = verts[:, 1].min().item() - ground_buffer
        else:
            ground_y = verts[:, 1].max().item() + ground_buffer

        ground_size = 100
        ground_verts,ground_faces = make_big_ground( ground_size, 1, ground_level = ground_y, max_triangle_len = ground_size/100.0, device = device )
        num_verts_before = verts.shape[0]
        verts = torch.cat([verts, ground_verts], dim=0)
        faces = torch.cat([faces, ground_faces + num_verts_before], dim=0)

        # set ground material properties
        ground_properties = torch.tensor(ground_raids, device=device, dtype=torch.float32).reshape(1, 5).repeat(ground_faces.shape[0], 1)  # (F_g, 5)
        raids = torch.cat([raids, ground_properties], dim=0)  # (F, 5)

    # calculate the normals
    face_verts = verts[faces]  # (F, 3, 3)
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
    face_normals = torch.nn.functional.normalize(face_normals, dim=1)  # (F, 3)

    # repack the mesh with the new verts and faces
    mesh = Meshes(verts=[verts], faces=[faces])

    # ensure material properties are valid
    raids = correct_material_properties(raids)

    return mesh, face_normals, raids
    #      obj   (F,3)         (F,5)


def make_big_ground( size, ground_dim, ground_level = 0.0, max_triangle_len = 0.1, device = 'cpu' ):
    """
    Make a big ground plane with the specified size and ground level.
    
    Inputs:
        size (int): size of the ground plane
        ground_dim (int): dimension of the ground plane
        ground_level (float): level of the ground plane
        max_triangle_len (float): maximum length of a side in the triangle mesh
        device (str): device to use for the mesh
    Returns:
        mesh (Mesh): the generated ground mesh
    """
    N = int(math.floor(size/max_triangle_len))+1
    assert(N > 1), "N should be greater than 1, but got %d" % N

    # calculate verticies
    grid = torch.meshgrid(   torch.linspace(-size/2, size/2, N, device=device, dtype=torch.float32),
                             torch.linspace(-size/2, size/2, N, device=device, dtype=torch.float32),
                             indexing='ij')  # (N,N), (N,N)
    verts = torch.stack(grid, dim=-1)  # (N,N,2)
    verts = torch.cat((verts, torch.full((N,N,1), ground_level, device=device, dtype=torch.float32)), dim=-1)  # (V,3)
    verts = verts.reshape(-1, 3)  # (V,3)

    # calculate faces
    h,w = torch.meshgrid( torch.arange(N-1, device=device, dtype=torch.int64),  # (N-1,)
                          torch.arange(N-1, device=device, dtype=torch.int64), # (N-1,)
    indexing='ij') # (N-1,N-1), (N-1,N-1)

    squares = torch.stack((h*N+w, (h+1)*N+w, h*N+w+1, (h+1)*N + w+1), dim=-1).reshape(-1,4)  # (F/2,4)
    upper_triangles = squares[..., :3] # (F/2,3)
    lower_triangles = squares[..., 1:] # (F/2,3)
    faces = torch.cat((upper_triangles, lower_triangles), dim=1)  # (F,3)
    faces = torch.reshape(faces, (-1, 3))  # (F,3)

    # make sure faces are within the range of vertices
    assert( torch.all(faces < verts.shape[0]) ), "faces are out of bounds"  # ensure faces are within the range of vertices

    # set the ground dim, currently it's 2
    verts = torch.roll(verts, shifts=ground_dim+1, dims=1)  # (V,3)

    # return the mesh
    return verts,faces


if __name__ == '__main__':
    # test the trajectory generation
    device = 'cuda'
    T = 1
    center_az = 45
    center_el = 45
    n_pulses = 30
    azimuth_spread_deg = 120
    pose = generate_pose_mat(center_az, center_el, distance=5, device=device).unsqueeze(0).repeat(T,1,1)  # (T,4,4)
    trajectory = generate_trajectory(pose, trajectory_type='linear', n_pulses=n_pulses, azimuth_spread_deg=azimuth_spread_deg,debug=True)  # (T,P,3)
    print('Trajectory:', trajectory)
    print('Trajectory shape:', trajectory.shape)
