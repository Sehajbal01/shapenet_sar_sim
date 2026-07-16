import torch
import time
import sys
import os
import numpy as np
from utils import cartesian_to_spherical, generate_pose_mat, dot_product, directional_scatter_polynomial_alpha5, plot_rays, get_next_path, plot_image, savefig
from ray_tracer_v2 import ray_trace, build_octree


def ray_trace_oom_safe(ray_origins, ray_directions, mesh, face_normals,
                       octree=None, batch_size=2**20, min_batch_size=1,
                       show_pbar=False):
    '''
    Call ray_trace, halving batch_size on CUDA OOM until it fits.

    The octree path builds a padded (K, max_Fl, 3) face matrix whose size scales
    with the ray batch B; on dense scenes this can exceed VRAM. Rather than tune
    batch_size per scene, we catch OutOfMemoryError, free the cache, and retry the
    whole call with a smaller batch. ray_trace allocates fresh output tensors each
    call, so a failed attempt leaves nothing to clean up but the allocator cache.
    '''
    R = ray_origins.shape[0]
    # clamp to R so the first halving is actually effective (callers may pass a
    # huge sentinel batch size like 2**100 == "all rays in one batch")
    bs = min(batch_size, R) if R > 0 else batch_size
    while True:
        try:
            return ray_trace(ray_origins, ray_directions, mesh, face_normals,
                             octree=octree, batch_size=bs, show_pbar=show_pbar)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= min_batch_size:
                raise
            bs = max(min_batch_size, bs // 2)
            print(f'ray_trace_oom_safe: CUDA OOM, retrying with batch_size={bs}')


def points_visible_to_sensor(points, sensor_direction, mesh, face_normals,
                             octree=None, surface_bias=1e-3, batch_size=2**20):
    '''
    Determine whether each point has an unobstructed line of sight back to the sensor.

    The sensor is treated as infinitely far away (orthographic / planar-wavefront model),
    so the direction from every point back to the sensor is the same unit vector. A point
    is visible if a ray cast from it toward the sensor escapes the scene without hitting
    any geometry; if it hits a triangle first, the point is occluded.

    Inputs:
        points (N,3): points in space to test
        sensor_direction (3,): unit vector pointing from a point back toward the sensor,
            i.e. normalized trajectory[t,p]
        mesh (obj): pytorch3d mesh of the scene
        face_normals (F,3): face normals (passed through to ray_trace)
        octree: prebuilt Octree for the mesh, or None to build one here
        surface_bias (float): distance to push each ray origin off its surface along the
            sensor direction, to avoid self-intersection (same issue as the bounce origins)
        batch_size (int): max rays per ray_trace batch

    Outputs:
        visible (N,): boolean tensor, True where the point can see the sensor
    '''
    if octree is None:
        octree = build_octree(mesh)

    # every point shoots the same direction toward the (infinitely far) sensor
    directions = sensor_direction.reshape(1, 3).expand(points.shape[0], -1)  # (N, 3)

    # bias origins toward the sensor so the shadow ray doesn't re-hit the surface the
    # point sits on (leg~=0 self-intersection)
    origins = points + surface_bias * directions  # (N, 3)

    _, distance = ray_trace_oom_safe(origins, directions, mesh, face_normals,
                                     octree=octree, batch_size=batch_size)  # (N,)

    # a hit (distance >= 0) means geometry blocks the path to the sensor
    visible = distance < 0  # (N,)
    return visible


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

    # diagnostics for cos(theta/2): sqrt() of a negative silently yields NaN rather than
    # raising, so track how often it happens and how far below zero the argument gets.
    n_nan_cos       = 0
    n_nan_arg       = 0
    n_cos_total     = 0
    min_sqrt_arg    = float('inf')

    # time octree build separately (it can be expensive and was previously
    # omitted from the profiling breakdown)
    t_octree_start = sync_time()
    octree = build_octree(mesh)
    t_octree_build = sync_time() - t_octree_start

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

            # unit direction from any scene point back to the (infinitely far) sensor
            sensor_direction = torch.nn.functional.normalize(trajectory[t, p], dim=-1)  # (3,)

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
                hit_indices, distance = ray_trace_oom_safe(prev_origins, prev_directions, mesh, face_normals, octree=octree, batch_size=second_bounce_batch_size)

                # always account for the time spent in this bounce's ray-trace call
                t_b_elapsed = sync_time() - t_b_start
                t_bounce_totals[b-1] += t_b_elapsed

                hit_b = distance >= 0
                # if no rays hit for this bounce, nothing else to do; report time above
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

                n = face_normals[hit_indices]  # (N, 3)

                # orient each normal to face the incoming ray (outward on the side the ray came
                # from). Mesh triangle winding is not guaranteed consistent, so face_normals may
                # point either way; the reflection below is invariant to n's sign, but poly_input
                # (dot(n, next_directions)) is linear in n and would otherwise flip sign on
                # back-facing normals, collapsing the directional-scatter denominator and spiking
                # the returned energy. prev_directions points into the surface, so an outward
                # normal has dot(prev_directions, n) <= 0; only flip the strictly back-facing ones
                # (a where() rather than -sign(), so an exactly grazing dot==0 leaves n intact
                # instead of being zeroed).
                n = torch.where(dot_product(prev_directions, n, keepdim=True) > 0, -n, n)

                # calculate reflected ray direction
                next_directions = prev_directions - 2 * dot_product(prev_directions, n, keepdim=True) * n

                # calculate returned energy
                s = material_properties[hit_indices, 4]
                i = material_properties[hit_indices, 2]
                d = material_properties[hit_indices, 3]
                poly_input = dot_product(n, next_directions)

                # half-angle identity: cos(theta/2) = sqrt((1 + cos theta)/2), where theta is the
                # angle between the reflected ray and the direction back to the sensor. Both are
                # unit vectors, so the argument is mathematically in [0, 1]; it only dips below 0
                # by float error when the reflected ray points ~directly away from the sensor
                # (cos theta = -1), where the true value is 0. NaN -> 0 is that exact limit.
                sqrt_arg = (1 + dot_product(sensor_direction, next_directions)) / 2
                cos_theta_over_2 = torch.sqrt(sqrt_arg)
                n_nan_cos   += int(torch.isnan(cos_theta_over_2).sum())
                n_cos_total += cos_theta_over_2.numel()
                # an already-NaN argument means NaN arrived from upstream (a bad normal or
                # direction), which is a real bug rather than boundary rounding; count it apart
                # and keep it out of the min so it cannot masquerade as a benign undershoot.
                n_nan_arg    += int(torch.isnan(sqrt_arg).sum())
                min_sqrt_arg  = min(min_sqrt_arg,
                                    float(torch.nan_to_num(sqrt_arg, nan=float('inf')).min()))
                cos_theta_over_2 = torch.nan_to_num(cos_theta_over_2, nan=0.0)
                energy_b = cumulative_reflectivity * s * (
                    (i * cos_theta_over_2**5) /
                    directional_scatter_polynomial_alpha5(poly_input) +
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

                # cull occluded scatters: a hit only returns energy if its path back to the
                # sensor is unobstructed. First-bounce hits are the nearest intersection along
                # the incoming ray, so they are always visible; only later bounces can be
                # occluded (e.g. a ground point hidden behind the object). This filters what we
                # store, NOT the ray that propagates onward to the next bounce.
                if b > 1:
                    visible = points_visible_to_sensor(
                        hit_b_pos, sensor_direction, mesh, face_normals,
                        octree=octree, surface_bias=surface_bias, batch_size=second_bounce_batch_size,
                    )  # (N,)
                else:
                    visible = torch.ones(hit_b_pos.shape[0], dtype=torch.bool, device=device)

                scatter_ranges[t][-1]   = torch.cat((scatter_ranges[t][-1],   total_range[visible]))
                scatter_energies[t][-1] = torch.cat((scatter_energies[t][-1], energy_b[visible]))

                # attenuate future bounces by this surface's reflectivity
                r = material_properties[hit_indices, 0]
                cumulative_reflectivity = cumulative_reflectivity * r

                # reflect for next bounce
                prev_directions = next_directions

                # bias the new origin off the surface along the normal so the reflected ray
                # cannot spuriously re-hit the surface it just left. Without this, a reflected
                # ray that grazes the (flat, finely tessellated) ground re-intersects an adjacent
                # coplanar triangle at leg~=0, dumping a duplicate full-energy scatter at the
                # first-bounce range. Those pile up coherently and spike the signal on pulses
                # whose geometry produces many grazing ground reflections.
                # n faces the incoming ray, so the reflected ray always departs into the +n
                # half-space (reflected.n = -(incoming.n) >= 0); pushing along +n is always the
                # correct side. (Previously this used sign(dot(reflected, n)) to stay robust to
                # arbitrary normal orientation, but the orientation fix above makes that always +1.)
                prev_origins = hit_b_pos + surface_bias * n

                t_bounce_totals[b-1] += sync_time() - t_b_start

        # leave as list — each pulse has a different number of hit rays
        pass

    t_overall = sync_time() - t_overall_start
    bounce_times = '  '.join(f'bounce{b+1}={t:.3f}s' for b, t in enumerate(t_bounce_totals))
    # report octree build time separately and ensure breakdown is clear
    print(f"accumulate_scatters: overall={t_overall:.3f}s  octree={t_octree_build:.3f}s  setup={t_setup_total:.3f}s  {bounce_times}")
    if n_nan_cos:
        print(f"accumulate_scatters: cos(theta/2) NaN -> 0 on {n_nan_cos}/{n_cos_total} scatters "
              f"({100*n_nan_cos/n_cos_total:.3f}%); min sqrt arg={min_sqrt_arg:.3e}")
        if n_nan_arg:
            print(f"accumulate_scatters: WARNING {n_nan_arg} of those had a NaN argument (not "
                  f"boundary rounding) — check face normals / ray directions for NaN")

    # apply complex value to the energy according to wavelength
    if wavelength is not None:
        for t in range(T):
            for p in range(P):
                scatter_energies[t][p] = scatter_energies[t][p] * torch.exp(
                    1j * 2 * np.pi / wavelength * scatter_ranges[t][p]
                )

    return scatter_ranges, scatter_energies, debugging_maps if debug_gif else None
    #      list[T][P] of 1-D tensors (R' hit rays, varies per pulse), dict (t,p)->(H,W) or None
