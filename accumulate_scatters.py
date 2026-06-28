import torch
import time
import sys
import os
import numpy as np
from utils import cartesian_to_spherical, generate_pose_mat, dot_product, directional_scatter_polynomial_alpha5, plot_rays, get_next_path, plot_image, savefig
from ray_tracer_v2 import ray_trace, build_octree

def accumulate_scatters(mesh, face_normals, material_properties,
                        trajectory,
                        wavelength=None,
                        grid_width=1, grid_height=1,
                        n_ray_width=1, n_ray_height=1,
                        num_bounce = 1,
                        second_bounce_batch_size = 2**100,
                        surface_bias = 1e-3,
                        debug_gif = False,
                    ):
    '''
    returns the energy and range for a bunch of rays for each pulse

    inputs:
        mesh (obj): pytorch3d mesh object of the 3d model
        face_normals (F,3): the normal vector of each face on the mesh
        material_properties (F,5): the r,a,i,d,s of each face of the mesh
        trajectory (T,P,3): the locations of the sensor for each pulse for each target scene
        wavelength (float): the wavelength of the radar signal, if none, there will be no complex value in the energy
        grid_width/height (float): the size of the ray grid for the orthonormal camera
        n_ray_width/height (int): the number of rays on the ray grid along the width and height axis.
        surface_bias (float): distance to push each bounce's outgoing ray origin off the surface
            along the normal, to prevent self-intersection (spurious leg~=0 re-hits). Should be
            small relative to scene features but large relative to float error at the scene scale.

    outputs:
        range (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only)
        energy (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only)

    '''
    device = trajectory.device
    T = trajectory.shape[0]  # no. of camera views
    P = trajectory.shape[1]    # no. of pulses per view

    def sync_time():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter()

    t_overall_start = sync_time()
    t_setup_total   = 0.0
    t_bounce_totals = [0.0] * num_bounce

    octree = build_octree(mesh)

    debugging_maps = {}  # (t, p) -> {'depth': (H,W), 'energy': (H,W)}; only populated when debug_gif=True

    scatter_ranges = []
    scatter_energies = []
    for t in range(T):
        scatter_ranges.append([])
        scatter_energies.append([])
        for p in range(P):

            t_b1_start = sync_time()

            # compute camera axes from pose matrix (columns: right, up, forward)
            cam_azimuth_deg, cam_elevation_deg, cam_distance = cartesian_to_spherical(trajectory[t, p])
            pose           = generate_pose_mat(cam_azimuth_deg, cam_elevation_deg, cam_distance, device=device)
            right_vector   = pose[:3, 0]  # (3,)
            up_vector      = pose[:3, 1]  # (3,)
            forward_vector = pose[:3, 2]  # (3,)
            position_vector = pose[:3, 3]  # (3,)

            # set up ray origins on the sensor plane
            x_offsets = torch.linspace(-grid_width/2,  grid_width/2,  n_ray_width,  device=device)  # (W,) left→right
            y_offsets = torch.linspace( grid_height/2, -grid_height/2, n_ray_height, device=device)  # (H,) top→bottom
            grid_y, grid_x = torch.meshgrid(y_offsets, x_offsets, indexing='ij')  # (H, W)
            first_bounce_origins = (trajectory[t, p].reshape(1, 1, 3)
                                    + grid_x.unsqueeze(-1) * right_vector
                                    + grid_y.unsqueeze(-1) * up_vector)  # (H, W, 3)

            prev_origins    = first_bounce_origins.reshape(-1, 3)                            # (H*W, 3)
            prev_directions = forward_vector.unsqueeze(0).expand(prev_origins.shape[0], -1)  # (H*W, 3)
            cumulative_legs = torch.zeros(prev_origins.shape[0], device=device)              # (H*W,)
            cumulative_reflectivity = torch.ones(prev_origins.shape[0], device=device)       # (H*W,) product of r of prior bounces
            depth_hit1      = None

            scatter_ranges[t].append(torch.empty(0, device=device))
            scatter_energies[t].append(torch.empty(0, device=device))

            t_setup_total += sync_time() - t_b1_start

            # ray-trace all bounces
            for b in range(1, num_bounce + 1):
                t_b_start = sync_time()
                hit_indices, distance = ray_trace(prev_origins, prev_directions, mesh, face_normals, octree=octree, batch_size=second_bounce_batch_size)

                hit_b = distance >= 0
                if not hit_b.any():
                    break

                # filter state to rays that hit
                prev_origins    = prev_origins[hit_b]
                prev_directions = prev_directions[hit_b]
                distance        = distance[hit_b]
                hit_indices     = hit_indices[hit_b]
                cumulative_legs = cumulative_legs[hit_b]
                cumulative_reflectivity = cumulative_reflectivity[hit_b]
                if depth_hit1 is not None:
                    depth_hit1 = depth_hit1[hit_b]

                hit_b_pos = prev_origins + distance.unsqueeze(-1) * prev_directions  # (N, 3)

                # calculate returned energy
                s = material_properties[hit_indices, 4]
                i = material_properties[hit_indices, 2]
                d = material_properties[hit_indices, 3]
                n = face_normals[hit_indices]  # (N, 3)
                cos_theta_over_2 = torch.abs(dot_product(n, prev_directions))
                energy_b = cumulative_reflectivity * s * (
                    (i * cos_theta_over_2**5) /
                    directional_scatter_polynomial_alpha5(cos_theta_over_2) +
                    d / 2 / np.pi
                )  # (N,) attenuated by the reflectivity of all prior bounces

                # store per-pixel depth and energy maps for the first bounce (misses get -1 / 0)
                if b == 1 and debug_gif:
                    HW = n_ray_height * n_ray_width
                    depth_map_flat = torch.full((HW,), -1.0, device=device, dtype=distance.dtype)
                    depth_map_flat[hit_b] = distance
                    energy_map_flat = torch.zeros(HW, device=device, dtype=energy_b.dtype)
                    energy_map_flat[hit_b] = energy_b
                    debugging_maps[(t, p)] = {
                        'depth':  depth_map_flat.reshape(n_ray_height, n_ray_width),
                        'energy': energy_map_flat.reshape(n_ray_height, n_ray_width),
                    }

                # round-trip range: for b=1 this gives 2×distance; for b>1 adds inter-bounce legs
                distance_to_sensor_plane = dot_product(hit_b_pos - trajectory[t, p], forward_vector)  # (N,)
                if b == 1:
                    depth_hit1 = distance_to_sensor_plane  # cumulative_legs stays 0
                else:
                    cumulative_legs = cumulative_legs + distance
                total_range = depth_hit1 + cumulative_legs + distance_to_sensor_plane  # (N,)

                scatter_ranges[t][-1]   = torch.cat((scatter_ranges[t][-1],   total_range))
                scatter_energies[t][-1] = torch.cat((scatter_energies[t][-1], energy_b))

                # attenuate future bounces by this surface's reflectivity
                r = material_properties[hit_indices, 0]
                cumulative_reflectivity = cumulative_reflectivity * r

                # reflect for next bounce
                prev_directions = prev_directions - 2 * dot_product(prev_directions, n, keepdim=True) * n
                # bias the new origin off the surface along the normal so the reflected ray
                # cannot spuriously re-hit the surface it just left. Without this, a reflected
                # ray that grazes the (flat, finely tessellated) ground re-intersects an adjacent
                # coplanar triangle at leg~=0, dumping a duplicate full-energy scatter at the
                # first-bounce range. Those pile up coherently and spike the signal on pulses
                # whose geometry produces many grazing ground reflections.
                offset_sign  = torch.sign(dot_product(prev_directions, n, keepdim=True))
                prev_origins = hit_b_pos + surface_bias * offset_sign * n

                t_bounce_totals[b-1] += sync_time() - t_b_start

        # leave as list — each pulse has a different number of hit rays
        pass

    t_overall = sync_time() - t_overall_start
    bounce_times = '  '.join(f'bounce{b+1}={t:.3f}s' for b, t in enumerate(t_bounce_totals))
    print(f"accumulate_scatters: overall={t_overall:.3f}s  setup={t_setup_total:.3f}s  {bounce_times}")

    # apply complex value to the energy according to wavelength
    if wavelength is not None:
        for t in range(T):
            for p in range(P):
                scatter_energies[t][p] = scatter_energies[t][p] * torch.exp(
                    1j * 2 * np.pi / wavelength * scatter_ranges[t][p]
                )

    return scatter_ranges, scatter_energies, debugging_maps if debug_gif else None
    #      list[T][P] of 1-D tensors (R' hit rays, varies per pulse), dict (t,p)->(H,W) or None
