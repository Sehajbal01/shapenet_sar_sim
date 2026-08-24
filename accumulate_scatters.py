'''
One scatter accumulator for arbitrary rays, behind the three sensor geometries the simulator
uses: accumulate_scatters (planar wavefront), accumulate_scatters_perspective (point sensor)
and accumulate_scatters_side_scan (point sensor, pose supplied). Each was its own near-duplicate
implementation until they were collapsed onto accumulate_scatters_from_rays here.

Run this file to check the wrappers still behave:
    CUDA_VISIBLE_DEVICES=0 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 accumulate_scatters.py
'''

import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the second to
# initialize aborts with "OMP: Error #15". Allow the duplicate, as paper_figures.py does.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import contextlib
import io
import sys
import time

import numpy as np
import torch

from utils import (
    cartesian_to_spherical,
    directional_scatter_polynomial_alpha5,
    dot_product,
    generate_pose_mat,
    plot_image,
    savefig,
    spherical_to_cartesian,
)
from ray_tracer_v2 import build_octree, ray_trace

# dump the side scan fan's azimuth map on the first ping and stop; off so the tests can run
DEBUG_AZIMUTH_MAP = False


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


def centered_linspace(span, n, device, flip=False):
    '''
    n evenly spaced samples covering `span`, centered on 0.

    torch.linspace(-s/2, s/2, 1) returns the *left edge* rather than the center, which would
    aim a single ray at the corner of the fan instead of down boresight; n==1 is special-cased.

    inputs:
        span (float): total extent covered by the samples
        n (int): number of samples
        flip (bool): return the samples in descending order (top->bottom image rows)
    outputs:
        offsets (n,): sample offsets about 0
    '''
    if n == 1:
        return torch.zeros(1, device=device)
    lo, hi = -span / 2, span / 2
    if flip:
        lo, hi = hi, lo
    return torch.linspace(lo, hi, n, device=device)


def points_visible_to_far_sensor(points, to_sensor_directions, mesh, face_normals,
                                 octree=None, surface_bias=1e-3, batch_size=2**20):
    '''
    Line of sight back to an infinitely far sensor, evaluated per point.

    Rays carry their own sensor, so the direction back to it can vary point to point even
    under the planar-wave assumption. A point is visible if a ray cast from it toward the
    sensor escapes the scene; if it hits a triangle first, the point is occluded.

    Inputs:
        points (N,3): points in space to test
        to_sensor_directions (N,3): unit vector from each point back toward its sensor
        mesh (obj): pytorch3d mesh of the scene
        face_normals (F,3): face normals (passed through to ray_trace)
        octree: prebuilt Octree for the mesh, or None to build one here
        surface_bias (float): distance to push each ray origin off its surface along the
            sensor direction, to avoid self-intersection
        batch_size (int): max rays per ray_trace batch

    Outputs:
        visible (N,): boolean tensor, True where the point can see its sensor
    '''
    if octree is None:
        octree = build_octree(mesh)

    # bias origins toward the sensor so the shadow ray doesn't re-hit the surface the
    # point sits on (leg~=0 self-intersection)
    origins = points + surface_bias * to_sensor_directions  # (N, 3)

    _, distance = ray_trace_oom_safe(origins, to_sensor_directions, mesh, face_normals,
                                     octree=octree, batch_size=batch_size)  # (N,)

    # a hit (distance >= 0) means geometry blocks the path to the sensor
    visible = distance < 0  # (N,)
    return visible


def points_visible_to_point_sensor(points, sensor_positions, mesh, face_normals,
                                   octree=None, surface_bias=1e-3, batch_size=2**20):
    '''
    Line of sight to a sensor at a finite location, evaluated per point.

    Only geometry *between* the point and the sensor occludes it -- a hit beyond the sensor
    is behind it and blocks nothing.

    Inputs:
        points (N,3): points in space to test
        sensor_positions (N,3): location of the sensor each point reports back to
        mesh (obj): pytorch3d mesh of the scene
        face_normals (F,3): face normals (passed through to ray_trace)
        octree: prebuilt Octree for the mesh, or None to build one here
        surface_bias (float): distance to push each ray origin off its surface toward the
            sensor, to avoid self-intersection
        batch_size (int): max rays per ray_trace batch

    Outputs:
        visible (N,): boolean tensor, True where the point can see its sensor
    '''
    if octree is None:
        octree = build_octree(mesh)

    to_sensor = sensor_positions - points                       # (N, 3)
    distance_to_sensor = to_sensor.norm(dim=-1)                 # (N,)
    directions = to_sensor / distance_to_sensor.unsqueeze(-1).clamp_min(1e-12)  # (N, 3)

    # bias origins toward the sensor so the shadow ray doesn't re-hit the surface the
    # point sits on (leg~=0 self-intersection)
    origins = points + surface_bias * directions  # (N, 3)

    _, distance = ray_trace_oom_safe(origins, directions, mesh, face_normals,
                                     octree=octree, batch_size=batch_size)  # (N,)

    # a hit (distance >= 0) blocks the path only if it lands short of the sensor. Origins were
    # already pushed surface_bias along the way, so that much less of the path remains.
    visible = (distance < 0) | (distance > distance_to_sensor - surface_bias)  # (N,)
    return visible


def accumulate_scatters_from_rays(
        mesh,
        face_normals,
        material_properties,
        ray_origins,
        ray_directions,
        sensor_positions=None,
        planar_wave=False,
        wavelength=None,
        num_bounce=1,
        spherical_spread=False,
        attenuation=0.0,
        octree=None,
        second_bounce_batch_size=2**100,
        surface_bias=1e-3,
        first_bounce_maps=False,
        stats=None,
    ):
    '''
    returns the energy and range of every scatter produced by an arbitrary set of rays

    One call renders one set of rays -- one pulse, one ping, one whatever. It knows nothing
    about ray grids, poses, or scene/pulse indexing; a caller that wants those loops over
    this, building the octree once and passing it in.

    inputs:
        mesh (obj): pytorch3d mesh object of the 3d model
        face_normals (F,3): the normal vector of each face on the mesh
        material_properties (F,5): the r,a,i,d,s of each face of the mesh
        ray_origins (R,3): where each ray starts
        ray_directions (R,3): unit direction each ray travels
        sensor_positions (R,3) or (3,): where each ray's return is collected. None means the
            sensor is wherever the ray started, which is the case for a fan leaving a point
            sensor; a parallel grid spread across a sensor plane must pass the sensor itself.
        planar_wave (bool): treat the sensor as infinitely far in the direction of its
            position, so the line of sight back to it is -normalize(sensor_position) and is
            fixed per ray. The return leg then becomes a projection onto that line of sight
            rather than a true distance. False treats the sensor as the finite point it is.
        wavelength (float): the wavelength of the radar signal, if none, there will be no complex value in the energy
        num_bounce (int): how many times each ray is allowed to reflect
        spherical_spread (bool): True applies energy /= 4*pi * range**2 over the round-trip
            range, spreading over the surface of the expanding sphere. False disables it.
        attenuation (float): absorption coefficient of the medium, in nepers per unit length
            (0 disables it). Applied over the round trip, so it is already the two-way loss.
            From the dB/m absorption is usually tabulated in: nepers = dB / 8.686.
        octree: prebuilt Octree for the mesh, or None to build one here
        surface_bias (float): distance to push each bounce's outgoing ray origin off the surface
            along the normal, to prevent self-intersection (spurious leg~=0 re-hits). Should be
            small relative to scene features but large relative to float error at the scene scale.
        first_bounce_maps (bool): also return the per-ray first-bounce depth and energy
        stats (dict): accumulated in place with timing and cos(theta/2) NaN diagnostics, for a
            caller that wants one summary line across many calls

    outputs:
        range (R',): round-trip path length of each surviving scatter, so ~2x the distance to
            the scatter on a first bounce. R' varies with how many rays hit and stayed visible
        energy (R',): energy of each surviving scatter, after spreading loss, attenuation and
            the wavelength phasor
        position (R',3): where each surviving scatter sits in world space. Anything measured
            relative to a sensor pose -- azimuth off boresight, say -- is the caller's job
        ray_index (R',): which transmitted ray produced each scatter, indexing into the R
            input rays. The bounce filtering otherwise loses that link, and without it a
            caller cannot attach a per-ray quantity (its fan azimuth, say) to the scatters
        first_bounce: {'depth': (R,), 'energy': (R,)} over the *transmitted* rays, misses
            getting -1 / 0, or None when first_bounce_maps is False
    '''
    device = ray_origins.device
    R = ray_origins.shape[0]

    def sync_time():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter()

    if octree is None:
        octree = build_octree(mesh)

    # every ray carries the sensor it reports back to; by default that is where it started
    if sensor_positions is None:
        sensor_positions = ray_origins
    else:
        sensor_positions = sensor_positions.reshape(-1, 3).expand(R, 3)

    # under the planar-wave assumption the sensor sits infinitely far away in the direction of
    # its position, so this is fixed per ray rather than evaluated per scatter
    if planar_wave:
        to_sensor_directions = torch.nn.functional.normalize(sensor_positions, dim=-1)  # (R,3)

    prev_origins    = ray_origins                                                 # (R,3)
    prev_directions = ray_directions                                              # (R,3)
    cumulative_legs = torch.zeros(R, device=device)                               # (R,)
    cumulative_reflectivity = torch.ones(R, device=device)                        # (R,) product of r of prior bounces
    ray_index       = torch.arange(R, device=device)                              # (R,) survives the bounce filtering
    outbound_range  = None  # sensor -> first hit, shared by every later bounce of that ray

    scatter_ranges      = torch.empty(0, device=device)
    scatter_energies    = torch.empty(0, device=device)
    scatter_positions   = torch.empty(0, 3, device=device)
    scatter_ray_indices = torch.empty(0, dtype=torch.long, device=device)
    first_bounce        = None

    t_bounce_totals = [0.0] * num_bounce

    # diagnostics for cos(theta/2): sqrt() of a negative silently yields NaN rather than
    # raising, so track how often it happens and how far below zero the argument gets.
    n_nan_cos    = 0
    n_nan_arg    = 0
    n_cos_total  = 0
    min_sqrt_arg = float('inf')

    # ray-trace all bounces
    for b in range(1, num_bounce + 1):
        t_b_start = sync_time()
        hit_indices, distance = ray_trace_oom_safe(prev_origins, prev_directions, mesh, face_normals, octree=octree, batch_size=second_bounce_batch_size)

        # bank the ray-trace time now so a bounce that breaks below still reports it
        t_after_trace = sync_time()
        t_bounce_totals[b-1] += t_after_trace - t_b_start

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
        ray_index       = ray_index[hit_b]
        sensor_positions = sensor_positions[hit_b]
        if planar_wave:
            to_sensor_directions = to_sensor_directions[hit_b]
        if outbound_range is not None:
            outbound_range = outbound_range[hit_b]

        hit_b_pos = prev_origins + distance.unsqueeze(-1) * prev_directions  # (N, 3)

        # vector from each scatter back to its sensor. A finite sensor makes the direction and
        # the leg per-scatter; a planar wavefront reuses the ray's fixed line of sight instead.
        to_sensor = sensor_positions - hit_b_pos                                        # (N, 3)
        if planar_wave:
            sensor_direction = to_sensor_directions                                     # (N, 3)
            return_leg = dot_product(to_sensor, sensor_direction)                       # (N,)
        else:
            return_leg = to_sensor.norm(dim=-1)                                         # (N,)
            sensor_direction = to_sensor / return_leg.unsqueeze(-1).clamp_min(1e-12)    # (N, 3)

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

        # store per-ray depth and energy for the first bounce (misses get -1 / 0)
        if b == 1 and first_bounce_maps:
            depth_flat = torch.full((R,), -1.0, device=device, dtype=distance.dtype)
            depth_flat[hit_b] = distance
            energy_flat = torch.zeros(R, device=device, dtype=energy_b.dtype)
            energy_flat[hit_b] = energy_b
            first_bounce = {'depth': depth_flat, 'energy': energy_flat}

        # round-trip range: the outbound leg to the first hit, plus any inter-bounce legs, plus
        # the leg straight back. Rays from a point sensor start at it, so their traced distance
        # *is* the outbound leg; planar rays start on the wavefront plane, so theirs is the
        # same projection as the return.
        if b == 1:
            outbound_range = return_leg if planar_wave else distance  # cumulative_legs stays 0
        else:
            cumulative_legs = cumulative_legs + distance
        total_range = outbound_range + cumulative_legs + return_leg  # (N,)

        # cull occluded scatters: a hit only returns energy if its path back to the
        # sensor is unobstructed. First-bounce hits are the nearest intersection along
        # the incoming ray, so they are always visible; only later bounces can be
        # occluded (e.g. a ground point hidden behind the object). This filters what we
        # store, NOT the ray that propagates onward to the next bounce.
        if b > 1 and planar_wave:
            visible = points_visible_to_far_sensor(
                hit_b_pos, sensor_direction, mesh, face_normals,
                octree=octree, surface_bias=surface_bias, batch_size=second_bounce_batch_size,
            )  # (N,)
        elif b > 1:
            visible = points_visible_to_point_sensor(
                hit_b_pos, sensor_positions, mesh, face_normals,
                octree=octree, surface_bias=surface_bias, batch_size=second_bounce_batch_size,
            )  # (N,)
        else:
            visible = torch.ones(hit_b_pos.shape[0], dtype=torch.bool, device=device)

        scatter_ranges      = torch.cat((scatter_ranges,      total_range[visible]))
        scatter_energies    = torch.cat((scatter_energies,    energy_b[visible]))
        scatter_positions   = torch.cat((scatter_positions,   hit_b_pos[visible]))
        scatter_ray_indices = torch.cat((scatter_ray_indices, ray_index[visible]))

        # attenuate future bounces by this surface's reflectivity
        r = material_properties[hit_indices, 0]
        cumulative_reflectivity = cumulative_reflectivity * r

        # reflect for next bounce
        prev_directions = next_directions

        # bias the new origin off the surface along the normal so the reflected ray
        # cannot spuriously re-hit the surface it just left. Without this, a reflected
        # ray that grazes the (flat, finely tessellated) ground re-intersects an adjacent
        # coplanar triangle at leg~=0, dumping a duplicate full-energy scatter at the
        # first-bounce range. n faces the incoming ray, so the reflected ray always
        # departs into the +n half-space and pushing along +n is always the correct side.
        prev_origins = hit_b_pos + surface_bias * n

        t_bounce_totals[b-1] += sync_time() - t_after_trace

    # spreading loss: the energy of a point source thins over the surface of an expanding
    # sphere. r is the scatter's full round-trip path length, so this attenuates far returns
    # relative to near ones within the same pulse. A planar wavefront does not suffer it.
    if spherical_spread:
        scatter_energies = scatter_energies / (
            4 * np.pi * scatter_ranges.clamp_min(1e-12)**2
        )

    # medium absorption: exponential in path length, which here is already the round trip
    if attenuation:
        scatter_energies = scatter_energies * torch.exp(-attenuation * scatter_ranges)

    # apply complex value to the energy according to wavelength
    if wavelength is not None:
        scatter_energies = scatter_energies * torch.exp(
            1j * 2 * np.pi / wavelength * scatter_ranges
        )

    if stats is not None:
        bounce_totals = stats.setdefault('t_bounce', [])
        while len(bounce_totals) < num_bounce:
            bounce_totals.append(0.0)
        for b, t in enumerate(t_bounce_totals):
            bounce_totals[b] += t
        stats['n_nan_cos']    = stats.get('n_nan_cos', 0) + n_nan_cos
        stats['n_nan_arg']    = stats.get('n_nan_arg', 0) + n_nan_arg
        stats['n_cos_total']  = stats.get('n_cos_total', 0) + n_cos_total
        stats['min_sqrt_arg'] = min(stats.get('min_sqrt_arg', float('inf')), min_sqrt_arg)

    return scatter_ranges, scatter_energies, scatter_positions, scatter_ray_indices, first_bounce
    #      (R',)           (R',)             (R',3)             (R',)                dict of (R,) or None


# ================================================================================
# wrappers reproducing the three older variants
# ================================================================================


def spherical_ray_fan(right_vector, up_vector, forward_vector,
                      fov_width_deg, fov_height_deg, n_ray_width, n_ray_height, device):
    '''
    Unit directions of a ray fan spread over the field of view, (H,W) flattened row-major.

    The fan is parameterized spherically rather than as a pinhole image plane, so a ray's
    azimuth is exactly its azimuth offset and the angular sampling stays uniform -- range
    angle imaging bins on that angle, so an even fan matters more than a flat image plane.

    inputs:
        right/up/forward_vector (3,): the sensor's axes
        fov_width_deg/fov_height_deg (float): angular extent in azimuth (about the sensor's
            up axis) and elevation (about its right axis)
        n_ray_width/height (int): number of rays along the azimuth and elevation axis
    outputs:
        directions (H*W,3): unit direction of every ray in the fan
        azimuths (H*W,): each ray's azimuth off boresight in degrees, positive toward the
            sensor's right vector. This is the angle the fan was built from, not a value
            recovered from the direction, so it is exact.
    '''
    az_offsets = centered_linspace(fov_width_deg  * np.pi/180, n_ray_width,  device)                # (W,) left→right
    el_offsets = centered_linspace(fov_height_deg * np.pi/180, n_ray_height, device, flip=True)     # (H,) top→bottom
    grid_el, grid_az = torch.meshgrid(el_offsets, az_offsets, indexing='ij')  # (H, W)
    directions = (
          (torch.cos(grid_el) * torch.cos(grid_az)).unsqueeze(-1) * forward_vector
        + (torch.cos(grid_el) * torch.sin(grid_az)).unsqueeze(-1) * right_vector
        + torch.sin(grid_el).unsqueeze(-1)                        * up_vector
    )  # (H, W, 3), already unit length
    return directions.reshape(-1, 3), (grid_az * 180 / np.pi).reshape(-1)


def _print_stats(name, stats, t_overall, t_octree_build, t_setup_total):
    '''The per-call timing and NaN summary the three originals print.'''
    bounce_times = '  '.join(f'bounce{b+1}={t:.3f}s'
                             for b, t in enumerate(stats.get('t_bounce', [])))
    print(f"{name}: overall={t_overall:.3f}s  octree={t_octree_build:.3f}s  "
          f"setup={t_setup_total:.3f}s  {bounce_times}")
    n_nan_cos = stats.get('n_nan_cos', 0)
    if n_nan_cos:
        n_cos_total = stats['n_cos_total']
        print(f"{name}: cos(theta/2) NaN -> 0 on {n_nan_cos}/{n_cos_total} scatters "
              f"({100*n_nan_cos/n_cos_total:.3f}%); min sqrt arg={stats['min_sqrt_arg']:.3e}")
        if stats.get('n_nan_arg', 0):
            print(f"{name}: WARNING {stats['n_nan_arg']} of those had a NaN argument (not "
                  f"boundary rounding) — check face normals / ray directions for NaN")


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

    Orthographic wrapper around accumulate_scatters_from_rays: the rays are a parallel grid
    of origins on a sensor plane, and the sensor is treated as infinitely far away, so ranges
    are distances to that plane rather than to a point.

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
        debugging_maps: dict (t,p) -> {'depth','energy'} of (H,W) maps, or None
    '''
    device = trajectory.device
    T = trajectory.shape[0]  # no. of camera views
    P = trajectory.shape[1]  # no. of pulses per view

    def sync_time():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter()

    t_overall_start = sync_time()
    t_setup_total   = 0.0
    stats           = {}

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

            t_setup_start = sync_time()

            # compute camera axes from pose matrix (columns: right, up, forward)
            cam_azimuth_deg, cam_elevation_deg, cam_distance = cartesian_to_spherical(trajectory[t, p])
            pose           = generate_pose_mat(cam_azimuth_deg, cam_elevation_deg, cam_distance, device=device)
            right_vector   = pose[:3, 0]  # (3,)
            up_vector      = pose[:3, 1]  # (3,)
            forward_vector = pose[:3, 2]  # (3,)

            # set up ray origins on the sensor plane
            x_offsets = torch.linspace(-grid_width/2,  grid_width/2,  n_ray_width,  device=device)  # (W,) left→right
            y_offsets = torch.linspace( grid_height/2, -grid_height/2, n_ray_height, device=device)  # (H,) top→bottom
            grid_y, grid_x = torch.meshgrid(y_offsets, x_offsets, indexing='ij')  # (H, W)
            first_bounce_origins = (trajectory[t, p].reshape(1, 1, 3)
                                    + grid_x.unsqueeze(-1) * right_vector
                                    + grid_y.unsqueeze(-1) * up_vector)  # (H, W, 3)

            ray_origins    = first_bounce_origins.reshape(-1, 3)                             # (H*W, 3)
            ray_directions = forward_vector.unsqueeze(0).expand(ray_origins.shape[0], -1)    # (H*W, 3)

            t_setup_total += sync_time() - t_setup_start

            # the rays are spread across the sensor plane, so the sensor is not where they
            # start -- it has to be named explicitly for the planar line of sight to be right
            ranges_p, energies_p, _, _, first_bounce = accumulate_scatters_from_rays(
                mesh, face_normals, material_properties,
                ray_origins, ray_directions,
                sensor_positions = trajectory[t, p],
                planar_wave      = True,
                wavelength       = wavelength,
                num_bounce       = num_bounce,
                octree           = octree,
                second_bounce_batch_size = second_bounce_batch_size,
                surface_bias     = surface_bias,
                first_bounce_maps = debug_gif,
                stats            = stats,
            )

            # normalized by number of rays. The divisor is the constant *transmitted* ray
            # count, not the per-pulse hit count, which would vary with aspect and taper the
            # aperture.
            scatter_ranges[t].append(ranges_p)
            scatter_energies[t].append(energies_p/n_ray_width/n_ray_height)

            if first_bounce is not None:
                debugging_maps[(t, p)] = {
                    'depth':  first_bounce['depth'].reshape(n_ray_height, n_ray_width),
                    'energy': first_bounce['energy'].reshape(n_ray_height, n_ray_width),
                }

    _print_stats('accumulate_scatters', stats,
                 sync_time() - t_overall_start, t_octree_build, t_setup_total)

    return scatter_ranges, scatter_energies, debugging_maps if debug_gif else None
    #      list[T][P] of 1-D tensors (R' hit rays, varies per pulse), dict (t,p)->(H,W) or None


def accumulate_scatters_perspective(mesh, face_normals, material_properties,
                                    trajectory,
                                    wavelength=None,
                                    fov_width_deg=30.0, fov_height_deg=30.0,
                                    n_ray_width=1, n_ray_height=1,
                                    num_bounce = 1,
                                    second_bounce_batch_size = 2**100,
                                    surface_bias = 1e-3,
                                    debug_gif = False,
                                ):
    '''
    Perspective wrapper around accumulate_scatters_from_rays: the sensor is a *point* rather
    than an infinitely distant plane, so the rays fan out from it, ranges are true distances
    to and from it, and the energies take a 1/(4*pi*r^2) spreading loss a planar wavefront does not.

    The sensor pose is rebuilt from each position, which always aims boresight at the scene
    origin; use accumulate_scatters_side_scan when the look direction is given instead.

    inputs:
        mesh (obj): pytorch3d mesh object of the 3d model
        face_normals (F,3): the normal vector of each face on the mesh
        material_properties (F,5): the r,a,i,d,s of each face of the mesh
        trajectory (T,P,3): the locations of the sensor for each pulse for each target scene
        wavelength (float): the wavelength of the radar signal, if none, there will be no complex value in the energy
        fov_width_deg/fov_height_deg (float): angular extent of the ray fan, in azimuth (about
            the sensor's up axis) and elevation (about its right axis)
        n_ray_width/height (int): the number of rays in the fan along the azimuth and elevation axis.
        surface_bias (float): distance to push each bounce's outgoing ray origin off the surface
            along the normal, to prevent self-intersection (spurious leg~=0 re-hits). Should be
            small relative to scene features but large relative to float error at the scene scale.

    outputs:
        range (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only).
            Round-trip path length, so ~2x the distance to the scatter on a first bounce
        energy (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only),
            attenuated by the 1/(4*pi*r^2) spreading loss over that round-trip range
        azimuth (T,)[P,][R']: list of lists of 1-D tensors; the launch azimuth in degrees off
            boresight, positive toward the sensor's right vector, of the ray that produced
            each scatter. On a second bounce that is where the ray left, not where the
            scatter ended up
        debugging_maps: dict (t,p) -> {'depth','energy'} of (H,W) maps, or None. The energy map
            is the raw first-bounce energy, before the spreading loss and ray-count normalization
    '''
    device = trajectory.device
    T = trajectory.shape[0]  # no. of camera views
    P = trajectory.shape[1]  # no. of pulses per view

    def sync_time():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter()

    t_overall_start = sync_time()
    t_setup_total   = 0.0
    stats           = {}

    t_octree_start = sync_time()
    octree = build_octree(mesh)
    t_octree_build = sync_time() - t_octree_start

    debugging_maps = {}  # (t, p) -> {'depth': (H,W), 'energy': (H,W)}; only populated when debug_gif=True

    scatter_ranges = []
    scatter_energies = []
    scatter_azimuths = []
    for t in range(T):
        scatter_ranges.append([])
        scatter_energies.append([])
        scatter_azimuths.append([])
        for p in range(P):

            t_setup_start = sync_time()

            # compute camera axes from pose matrix (columns: right, up, forward)
            cam_azimuth_deg, cam_elevation_deg, cam_distance = cartesian_to_spherical(trajectory[t, p])
            pose           = generate_pose_mat(cam_azimuth_deg, cam_elevation_deg, cam_distance, device=device)
            right_vector   = pose[:3, 0]  # (3,)
            up_vector      = pose[:3, 1]  # (3,)
            forward_vector = pose[:3, 2]  # (3,)

            # set up the ray fan: every ray leaves the sensor point, spread over the field of view
            ray_directions, ray_azimuths = spherical_ray_fan(right_vector, up_vector, forward_vector,
                                               fov_width_deg, fov_height_deg,
                                               n_ray_width, n_ray_height, device)         # (H*W, 3), (H*W,)
            ray_origins    = trajectory[t, p].reshape(1, 3).expand(ray_directions.shape[0], -1)

            t_setup_total += sync_time() - t_setup_start

            # rays leave the sensor itself, so the default sensor_positions is what we want
            ranges_p, energies_p, _, ray_indices_p, first_bounce = accumulate_scatters_from_rays(
                mesh, face_normals, material_properties,
                ray_origins, ray_directions,
                wavelength       = wavelength,
                num_bounce       = num_bounce,
                spherical_spread = True,
                octree           = octree,
                second_bounce_batch_size = second_bounce_batch_size,
                surface_bias     = surface_bias,
                first_bounce_maps = debug_gif,
                stats            = stats,
            )

            # azimuth of each scatter: the launch azimuth of the ray that produced it, taken
            # straight off the fan rather than recovered from the scatter's position.
            azimuths_p = ray_azimuths[ray_indices_p]                                         # (R',)

            # normalized by number of rays. The divisor is the constant *transmitted* ray
            # count, not the per-pulse hit count, which would vary with aspect and taper the
            # aperture.
            scatter_ranges[t].append(ranges_p)
            scatter_energies[t].append(energies_p/n_ray_width/n_ray_height)
            scatter_azimuths[t].append(azimuths_p)

            if first_bounce is not None:
                debugging_maps[(t, p)] = {
                    'depth':  first_bounce['depth'].reshape(n_ray_height, n_ray_width),
                    'energy': first_bounce['energy'].reshape(n_ray_height, n_ray_width),
                }

    _print_stats('accumulate_scatters_perspective', stats,
                 sync_time() - t_overall_start, t_octree_build, t_setup_total)

    return scatter_ranges, scatter_energies, scatter_azimuths, debugging_maps if debug_gif else None
    #      list[T][P] of 1-D tensors (R' hit rays, varies per pulse) x3, dict (t,p)->(H,W) or None


def accumulate_scatters_side_scan(mesh, face_normals, material_properties,
                                  poses,
                                  wavelength=None,
                                  fov_width_deg=30.0, fov_height_deg=30.0,
                                  n_ray_width=1, n_ray_height=1,
                                  num_bounce = 1,
                                  second_bounce_batch_size = 2**100,
                                  surface_bias = 1e-3,
                                  spherical_spread = True,
                                  water_absorption = 0.0,
                                  debug_gif = False,
                              ):
    '''
    Broadside wrapper around accumulate_scatters_from_rays: like the perspective wrapper, but
    the sensor's look direction is given rather than inferred.

    accumulate_scatters_perspective takes sensor *positions* and rebuilds a pose from each one,
    which always aims boresight at the scene origin. A side scan platform does not steer -- it
    flies a straight track staring off to one side -- so this wrapper takes the full pose matrix
    and reads the sensor axes straight out of its columns. Azimuths are therefore measured off
    a fixed boresight, so a scatter's azimuth tells you where it sits along the track rather
    than always being ~0. It also lets the caller switch the spreading loss off.

    inputs:
        mesh (obj): pytorch3d mesh object of the 3d model
        face_normals (F,3): the normal vector of each face on the mesh
        material_properties (F,5): the r,a,i,d,s of each face of the mesh
        poses (T,P,4,4): the sensor pose for each ping for each target scene, columns
            (right, up, forward, center) in the srn_cars convention of generate_pose_mat
        wavelength (float): the wavelength of the radar signal, if none, there will be no complex value in the energy
        fov_width_deg/fov_height_deg (float): angular extent of the ray fan, in azimuth (about
            the sensor's up axis) and elevation (about its right axis)
        n_ray_width/height (int): the number of rays in the fan along the azimuth and elevation axis.
        surface_bias (float): distance to push each bounce's outgoing ray origin off the surface
            along the normal, to prevent self-intersection (spurious leg~=0 re-hits). Should be
            small relative to scene features but large relative to float error at the scene scale.
        spherical_spread (bool): True applies energy /= 4*pi * range**2 over the round trip,
            spreading over the real surface of the expanding sphere; False disables the
            spreading loss entirely
        water_absorption (float): absorption coefficient of the water, in nepers per unit length
            (0 disables it). Applied over the round trip, so it is already the two-way loss.
            From the dB/m absorption is usually tabulated in: nepers = dB / 8.686.

    outputs:
        range (T,)[P,][R']: list of lists of 1-D tensors; R' varies per ping (hit rays only).
            Round-trip path length, so ~2x the distance to the scatter on a first bounce
        energy (T,)[P,][R']: list of lists of 1-D tensors; R' varies per ping (hit rays only),
            attenuated by the 1/(4*pi*r^2) spreading loss over that round-trip range when
            spherical_spread is on, and by water absorption if enabled
        azimuth (T,)[P,][R']: list of lists of 1-D tensors; the launch azimuth in degrees off
            the fixed boresight, positive toward the sensor's right vector (i.e. forward along
            track), of the ray that produced each scatter. On a second bounce that is where the
            ray left, not where the scatter ended up
        debugging_maps: dict (t,p) -> {'depth','energy'} of (H,W) maps, or None. The energy map
            is the raw first-bounce energy, before the spreading loss and ray-count normalization
    '''
    device = poses.device
    T = poses.shape[0]  # no. of scenes
    P = poses.shape[1]  # no. of pings per scene

    def sync_time():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter()

    t_overall_start = sync_time()
    t_setup_total   = 0.0
    stats           = {}

    t_octree_start = sync_time()
    octree = build_octree(mesh)
    t_octree_build = sync_time() - t_octree_start

    debugging_maps = {}  # (t, p) -> {'depth': (H,W), 'energy': (H,W)}; only populated when debug_gif=True

    scatter_ranges = []
    scatter_energies = []
    scatter_azimuths = []
    for t in range(T):
        scatter_ranges.append([])
        scatter_energies.append([])
        scatter_azimuths.append([])
        for p in range(P):

            t_setup_start = sync_time()

            # read the camera axes straight out of the pose (columns: right, up, forward, center)
            right_vector    = poses[t, p, :3, 0]  # (3,)
            up_vector       = poses[t, p, :3, 1]  # (3,)
            forward_vector  = poses[t, p, :3, 2]  # (3,)
            sensor_position = poses[t, p, :3, 3]  # (3,)

            # set up the ray fan: every ray leaves the sensor point, spread over the field of view
            ray_directions, ray_azimuths = spherical_ray_fan(right_vector, up_vector, forward_vector,
                                               fov_width_deg, fov_height_deg,
                                               n_ray_width, n_ray_height, device)         # (H*W,3), (H*W,)
            ray_origins    = sensor_position.reshape(1, 3).expand(ray_directions.shape[0], -1)

            t_setup_total += sync_time() - t_setup_start

            # rays leave the sensor itself, so the default sensor_positions is what we want
            ranges_p, energies_p, _, ray_indices_p, first_bounce = accumulate_scatters_from_rays(
                mesh, face_normals, material_properties,
                ray_origins, ray_directions,
                wavelength       = wavelength,
                num_bounce       = num_bounce,
                spherical_spread = spherical_spread,
                attenuation      = water_absorption,
                octree           = octree,
                second_bounce_batch_size = second_bounce_batch_size,
                surface_bias     = surface_bias,
                first_bounce_maps = debug_gif,
                stats            = stats,
            )

            # azimuth of each scatter: the launch azimuth of the ray that produced it, taken
            # straight off the fan rather than recovered from the scatter's position.
            azimuths_p = ray_azimuths[ray_indices_p]                                         # (R',)

            # DEBUG: azimuths_p is 1-D over hit rays only, so it has no grid to imshow; the
            # fan's azimuths do fill the (H,W) grid the rays were laid out on.
            if DEBUG_AZIMUTH_MAP:
                plot_image(ray_azimuths.reshape(n_ray_height, n_ray_width),
                           title='ray azimuth off boresight (deg)', cmap='viridis')
                savefig('figures/azimuth map.png')
                sys.exit()

            # normalized by number of rays. The divisor is the constant *transmitted* ray
            # count, not the per-ping hit count, which would vary with aspect and taper the track.
            scatter_ranges[t].append(ranges_p)
            scatter_energies[t].append(energies_p/n_ray_width/n_ray_height)
            scatter_azimuths[t].append(azimuths_p)

            if first_bounce is not None:
                debugging_maps[(t, p)] = {
                    'depth':  first_bounce['depth'].reshape(n_ray_height, n_ray_width),
                    'energy': first_bounce['energy'].reshape(n_ray_height, n_ray_width),
                }

    _print_stats('accumulate_scatters_side_scan', stats,
                 sync_time() - t_overall_start, t_octree_build, t_setup_total)

    return scatter_ranges, scatter_energies, scatter_azimuths, debugging_maps if debug_gif else None
    #      list[T][P] of 1-D tensors (R' hit rays, varies per ping) x3, dict (t,p)->(H,W) or None


# ================================================================================
# tests
# ================================================================================
#
# The originals these wrappers replaced are gone, so equality against them can no longer be
# re-checked here; that comparison lives in the history of the commit that deleted them. What
# remains are the two things that still hold without a reference implementation: a cross-check
# tying the two fan wrappers to each other through the shared core, and golden fingerprints
# recorded from the implementation that passed that equality check.

TEST_MESH_PATH = '/workspace/berian/sphere.obj'

# Summary statistics of each wrapper's output, recorded from the implementation that was proven
# equal to the three originals it replaced. Regenerate after an intended change with:
#     CUDA_VISIBLE_DEVICES=0 python3.8 accumulate_scatters.py --record
# The azimuths are the exception: they were re-recorded when the fan wrappers switched from
# deriving a scatter's azimuth from its position to reading it off the ray that produced it,
# which moves second-bounce scatters back inside the transmitted beam. Range, energy and the
# first-bounce maps still carry their original values.
GOLDEN = {
    'planar 2 bounce coherent': {
        'energy': (10443, -9.0569233538e-03, 1.0106582551e-02, 1.1756748427e-04),
        'maps': (12288, 1.9467985562e+04, 0.0000000000e+00, 4.1005740166e+00),
        'range': (10443, 6.8901203158e+04, 0.0000000000e+00, 4.9523815155e+01),
    },
    'planar 1 bounce incoherent': {
        'energy': (6144, 4.2399384540e-01, 0.0000000000e+00, 1.1756747699e-04),
        'maps': (0, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00),
        'range': (6144, 3.8067632387e+04, 0.0000000000e+00, 8.2011499405e+00),
    },
    'perspective 2 bounce': {
        'azimuth': (3228, -1.5193547815e+02, 0.0000000000e+00, 1.4999999046e+01),
        'energy': (3228, 1.1474418648e-05, -6.0732258903e-06, 3.0241915283e-07),
        'maps': (4096, 6.6351604041e+03, 0.0000000000e+00, 5.4100818634e+00),
        'range': (3228, 2.4311271969e+04, 0.0000000000e+00, 1.9580755615e+02),
    },
    'side scan absorption 0.05': {
        'azimuth': (4367, 1.5338709664e+02, 0.0000000000e+00, 1.4999999046e+01),
        'energy': (4367, -2.0285154739e-06, -2.3755772956e-06, 2.2365038888e-07),
        'maps': (6144, 1.0861379572e+04, 0.0000000000e+00, 7.0999932289e+00),
        'range': (4367, 3.4078427491e+04, 0.0000000000e+00, 1.8021507263e+02),
    },
    'side scan absorption 0.00': {
        'azimuth': (4367, 1.5338709664e+02, 0.0000000000e+00, 1.4999999046e+01),
        'energy': (4367, -2.9221323254e-06, -2.7978030329e-06, 2.8793971296e-07),
        'maps': (6144, 1.0861379572e+04, 0.0000000000e+00, 7.0999932289e+00),
        'range': (4367, 3.4078427491e+04, 0.0000000000e+00, 1.8021507263e+02),
    },
}


def _quiet(fn, *args, **kwargs):
    '''Run fn with stdout swallowed -- every wrapper prints a timing line per call.'''
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _test_scene(device):
    '''Sphere over a ground plane: gives real second bounces and real occlusion.'''
    from signal_simulation import load_mesh
    return _quiet(load_mesh, TEST_MESH_PATH, make_ground=True, device=device)


def _flatten(value, out):
    '''Collect every tensor in one of the nested wrapper returns into one flat list.'''
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten(item, out)
    elif isinstance(value, dict):
        for key in sorted(value):
            _flatten(value[key], out)
    elif value is not None:
        out.append(value.reshape(-1))
    return out


def _diff(new, ref):
    '''
    Largest absolute and relative difference between two nested results.

    Walks the list[T][P] / dict structures the wrappers return. A structural or shape
    disagreement is reported as infinite difference, since that already means the two
    runs kept a different set of scatters.

    outputs:
        max_abs (float), max_rel (float), note (str): note is '' when the structures line up
    '''
    if isinstance(ref, (list, tuple)):
        if not isinstance(new, (list, tuple)) or len(new) != len(ref):
            return float('inf'), float('inf'), 'length %s vs %s' % (
                len(new) if hasattr(new, '__len__') else type(new).__name__, len(ref))
        worst = (0.0, 0.0, '')
        for n_i, r_i in zip(new, ref):
            worst = max(worst, _diff(n_i, r_i), key=lambda d: (d[0], d[1]))
        return worst

    if isinstance(ref, dict):
        if not isinstance(new, dict) or set(new) != set(ref):
            return float('inf'), float('inf'), 'dict keys differ'
        worst = (0.0, 0.0, '')
        for key in ref:
            worst = max(worst, _diff(new[key], ref[key]), key=lambda d: (d[0], d[1]))
        return worst

    if ref is None:
        return (0.0, 0.0, '') if new is None else (float('inf'), float('inf'), 'expected None')

    if new.shape != ref.shape:
        return float('inf'), float('inf'), 'shape %s vs %s' % (tuple(new.shape), tuple(ref.shape))
    if ref.numel() == 0:
        return 0.0, 0.0, ''

    abs_err = (new - ref).abs()
    max_abs = float(abs_err.max())
    # relative to the reference magnitude; the floor keeps near-zero entries from dominating
    max_rel = float((abs_err / ref.abs().clamp_min(1e-30)).max())
    return max_abs, max_rel, ''


def _check(name, new, ref, atol=0.0, rtol=0.0):
    '''
    Report whether new matches ref within |new - ref| <= atol + rtol * |ref|.

    atol=rtol=0 demands bit-exact equality.
    '''
    max_abs, max_rel, note = _diff(new, ref)
    ok = np.isfinite(max_abs) and (max_abs <= atol or max_rel <= rtol)
    tol_str = 'exact' if (atol == 0.0 and rtol == 0.0) else 'atol=%.1e rtol=%.1e' % (atol, rtol)
    print('    %-28s %s  max_abs=%.3e  max_rel=%.3e  (%s)%s'
          % (name, 'PASS' if ok else 'FAIL', max_abs, max_rel, tol_str,
             '  ' + note if note else ''))
    return ok


def _fingerprint(**named):
    '''
    Order-insensitive summary of a wrapper's outputs: (count, sum of real parts, sum of
    imaginary parts, largest magnitude) per named output.

    Nothing downstream cares what order the scatters come back in, so sums are the right
    invariant. Real and imaginary parts are kept apart so a phase regression cannot hide
    behind a magnitude that still matches.
    '''
    fp = {}
    for name, value in sorted(named.items()):
        parts = _flatten(value, [])
        flat = torch.cat(parts) if parts else torch.zeros(0)
        wide = flat.to(torch.complex128) if flat.is_complex() else flat.to(torch.float64) + 0j
        fp[name] = (int(flat.numel()), float(wide.real.sum()), float(wide.imag.sum()),
                    float(flat.abs().max()) if flat.numel() else 0.0)
    return fp


def _check_fingerprint(label, fp, rtol=1e-6):
    '''Compare a fingerprint against its golden; counts must match exactly, sums within rtol.'''
    golden = GOLDEN.get(label)
    if golden is None:
        print('    %-28s NO GOLDEN  (run with --record)' % label)
        return False
    ok = True
    for name in sorted(fp):
        got, want = fp[name], golden.get(name)
        if want is None:
            print('    %-28s FAIL  %s missing from golden' % (label, name))
            ok = False
            continue
        if got[0] != want[0]:
            print('    %-28s FAIL  %s count %d vs %d' % (label, name, got[0], want[0]))
            ok = False
            continue
        worst = max(abs(g - w) / max(abs(w), 1e-30) for g, w in zip(got[1:], want[1:]))
        if worst > rtol:
            print('    %-28s FAIL  %s rel=%.3e' % (label, name, worst))
            ok = False
    if ok:
        print('    %-28s PASS  (%d outputs match golden)' % (label, len(fp)))
    return ok


def _wrapper_cases(device):
    '''
    Every wrapper under a config that exercises two bounces, occlusion and the energy tail.

    yields (label, fingerprint) so both the golden check and --record walk the same list.
    '''
    mesh, normals, materials = _test_scene(device)

    az = torch.tensor([[10.0, 12.0, 14.0], [100.0, 102.0, 104.0]], device=device)  # (T,P)
    trajectory = spherical_to_cartesian(az, torch.full_like(az, 30.0),
                                        torch.full_like(az, 4.0))                  # (T,P,3)
    fan_az = torch.tensor([[10.0, 40.0]], device=device)
    fan_trajectory = spherical_to_cartesian(fan_az, torch.full_like(fan_az, 30.0),
                                            torch.full_like(fan_az, 4.0))
    poses = _side_scan_poses(device)

    grid = dict(grid_width=1.2, grid_height=1.2, n_ray_width=32, n_ray_height=32,
                second_bounce_batch_size=2**18)
    fan = dict(fov_width_deg=30.0, fov_height_deg=30.0, n_ray_width=32, n_ray_height=32,
               second_bounce_batch_size=2**18)

    r, e, maps = _quiet(accumulate_scatters, mesh, normals, materials, trajectory,
                        num_bounce=2, wavelength=0.03, debug_gif=True, **grid)
    yield 'planar 2 bounce coherent', _fingerprint(range=r, energy=e, maps=maps)

    r, e, maps = _quiet(accumulate_scatters, mesh, normals, materials, trajectory,
                        num_bounce=1, wavelength=None, debug_gif=False, **grid)
    yield 'planar 1 bounce incoherent', _fingerprint(range=r, energy=e, maps=maps)

    r, e, a, maps = _quiet(accumulate_scatters_perspective, mesh, normals, materials,
                           fan_trajectory, num_bounce=2, wavelength=0.03, debug_gif=True, **fan)
    yield 'perspective 2 bounce', _fingerprint(range=r, energy=e, azimuth=a, maps=maps)

    for absorption in (0.05, 0.0):
        r, e, a, maps = _quiet(accumulate_scatters_side_scan, mesh, normals, materials, poses,
                               num_bounce=2, wavelength=0.03, debug_gif=True,
                               water_absorption=absorption, **fan)
        yield ('side scan absorption %.2f' % absorption,
               _fingerprint(range=r, energy=e, azimuth=a, maps=maps))


def test_golden_fingerprints(device='cuda'):
    '''Each wrapper still produces what it produced when it was checked against the originals.'''
    print('  golden fingerprints')
    ok = True
    for label, fp in _wrapper_cases(device):
        ok &= _check_fingerprint(label, fp)
    return ok


def _side_scan_poses(device, num_pings=3, track_length=1.5):
    '''
    Straight-track broadside poses, built the way side_scan_sonar_image builds them:
    fixed look direction, sensor sliding along the track.
    '''
    mean_sensor_position = spherical_to_cartesian(
        torch.tensor(20.0, device=device), torch.tensor(35.0, device=device),
        torch.tensor(4.0, device=device))                                          # (3,)
    line_of_sight   = torch.nn.functional.normalize(-mean_sensor_position, dim=-1)
    world_up        = torch.tensor([0.0, 0.0, 1.0], device=device)
    track_direction = torch.nn.functional.normalize(
        torch.linalg.cross(line_of_sight, world_up), dim=-1)
    up_vector       = torch.linalg.cross(track_direction, line_of_sight)

    ping_offsets = centered_linspace(track_length, num_pings, device)               # (P,)
    trajectory   = (mean_sensor_position.reshape(1, 3)
                    + ping_offsets.reshape(num_pings, 1) * track_direction.reshape(1, 3))

    poses = torch.zeros(num_pings, 4, 4, device=device)
    poses[:, :3, 0] = track_direction
    poses[:, :3, 1] = up_vector
    poses[:, :3, 2] = line_of_sight
    poses[:, :3, 3] = trajectory
    poses[:,  3, 3] = 1.0
    return poses.unsqueeze(0)  # (1,P,4,4), one scene


def test_perspective_matches_side_scan(device='cuda'):
    '''
    The two fan wrappers are the same physics with the pose supplied rather than inferred, so
    handing side scan exactly the pose perspective builds for itself must give the same
    scatters. Both spread over 4*pi*r**2 now that spherical_spread is a plain on/off, so the
    energies must match outright rather than up to a constant.

    This is the cross-check that survives deleting the originals: it pins the two wrappers to
    each other through the shared core without needing any reference implementation.
    '''
    print('  perspective == side scan on the same pose')
    mesh, normals, materials = _test_scene(device)

    az = torch.tensor([[15.0, 45.0]], device=device)                               # (T,P)
    trajectory = spherical_to_cartesian(az, torch.full_like(az, 30.0),
                                        torch.full_like(az, 4.0))                  # (T,P,3)

    # the pose perspective builds for itself, with the sensor position substituted straight back
    # in so both wrappers start from bit-identical geometry rather than a round-tripped position
    T, P = trajectory.shape[:2]
    poses = torch.zeros(T, P, 4, 4, device=device)
    for t in range(T):
        for p in range(P):
            a, e, d = cartesian_to_spherical(trajectory[t, p])
            poses[t, p] = generate_pose_mat(a, e, d, device=device)
            poses[t, p, :3, 3] = trajectory[t, p]

    kwargs = dict(wavelength=0.03, fov_width_deg=30.0, fov_height_deg=30.0,
                  n_ray_width=32, n_ray_height=32, num_bounce=2,
                  second_bounce_batch_size=2**18)
    per = _quiet(accumulate_scatters_perspective, mesh, normals, materials, trajectory, **kwargs)
    sss = _quiet(accumulate_scatters_side_scan, mesh, normals, materials, poses, **kwargs)

    ok = True
    ok &= _check('ranges', sss[0], per[0])
    ok &= _check('azimuths', sss[2], per[2])
    ok &= _check('energies', sss[1], per[1])
    return ok


def record_goldens(device=None):
    '''Print a paste-ready GOLDEN dict for the current implementation.'''
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('GOLDEN = {')
    for label, fp in _wrapper_cases(device):
        print("    '%s': {" % label)
        for name in sorted(fp):
            n, re_, im, absmax = fp[name]
            print("        '%s': (%d, %.10e, %.10e, %.10e)," % (name, n, re_, im, absmax))
        print('    },')
    print('}')


def run_all_tests(device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: %s' % device)
    print('scene:  %s + ground\n' % TEST_MESH_PATH)

    results = {}
    for test in (test_perspective_matches_side_scan, test_golden_fingerprints):
        results[test.__name__] = test(device)
        print()

    print('All tests passed!' if all(results.values()) else 'SOME TESTS FAILED!')
    for name, ok in results.items():
        print('  %-42s %s' % (name, 'pass' if ok else 'FAIL'))
    return all(results.values())


if __name__ == '__main__':
    if '--record' in sys.argv:
        record_goldens()
    else:
        sys.exit(0 if run_all_tests() else 1)
