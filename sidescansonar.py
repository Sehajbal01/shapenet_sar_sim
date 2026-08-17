import os
import time

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the second to
# initialize aborts with "OMP: Error #15". Allow the duplicate, as paper_figures.py does.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import PIL
from PIL import ImageDraw
import torch

from utils import dot_product, directional_scatter_polynomial_alpha5, extract_pose_info
from ray_tracer_v2 import build_octree
from accumulate_scatters import (
    centered_linspace,
    ray_trace_oom_safe,
    points_visible_to_finite_sensor,
)
from range_angle_images import beam_spread_weights
from signal_simulation import interpolate_signal, load_mesh


def accumulate_scatters_side_scan(mesh, face_normals, material_properties,
                                  poses,
                                  wavelength=None,
                                  fov_width_deg=30.0, fov_height_deg=30.0,
                                  n_ray_width=1, n_ray_height=1,
                                  num_bounce = 1,
                                  second_bounce_batch_size = 2**100,
                                  surface_bias = 1e-3,
                                  water_absorption = 0.0,
                                  debug_gif = False,
                              ):
    '''
    Broadside clone of accumulate_scatters_perspective: returns the energy, range, and azimuth
    angle of every scatter, for a sensor whose look direction is given rather than inferred.

    One thing differs from accumulate_scatters_perspective. That function takes sensor
    *positions* and rebuilds a pose from each one via cartesian_to_spherical/generate_pose_mat,
    which always aims boresight at the scene origin. A side scan platform does not steer -- it
    flies a straight track staring off to one side -- so this clone takes the full pose matrix
    and reads the sensor axes straight out of its columns. Everything downstream (the ray fan,
    the ranges, the per-scatter azimuths, the spreading loss, the occlusion culling) is
    unchanged, but the azimuths are now measured off a fixed boresight, so a scatter's azimuth
    tells you where it sits along the track rather than always being ~0.

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
        water_absorption (float): absorption coefficient of the water, in nepers per unit length
            (0 disables it). Applied over the round trip, so it is already the two-way loss.
            From the dB/m absorption is usually tabulated in: nepers = dB / 8.686.

    outputs:
        range (T,)[P,][R']: list of lists of 1-D tensors; R' varies per ping (hit rays only).
            Round-trip path length, so ~2x the distance to the scatter on a first bounce
        energy (T,)[P,][R']: list of lists of 1-D tensors; R' varies per ping (hit rays only),
            attenuated by the 1/r^2 spreading loss over that round-trip range, and by water
            absorption if enabled
        azimuth (T,)[P,][R']: list of lists of 1-D tensors; scatter azimuth in degrees off the
            fixed boresight, positive toward the sensor's right vector (i.e. forward along track)
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
    t_bounce_totals = [0.0] * num_bounce

    # diagnostics for cos(theta/2): sqrt() of a negative silently yields NaN rather than
    # raising, so track how often it happens and how far below zero the argument gets.
    n_nan_cos       = 0
    n_nan_arg       = 0
    n_cos_total     = 0
    min_sqrt_arg    = float('inf')

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

            t_b1_start = sync_time()

            # read the camera axes straight out of the pose (columns: right, up, forward, center)
            right_vector    = poses[t, p, :3, 0]  # (3,)
            up_vector       = poses[t, p, :3, 1]  # (3,)
            forward_vector  = poses[t, p, :3, 2]  # (3,)
            sensor_position = poses[t, p, :3, 3]  # (3,)

            # set up the ray fan: every ray leaves the sensor point, spread over the field of view
            az_offsets = centered_linspace(fov_width_deg  * np.pi/180, n_ray_width,  device)                # (W,) left→right
            el_offsets = centered_linspace(fov_height_deg * np.pi/180, n_ray_height, device, flip=True)     # (H,) top→bottom
            grid_el, grid_az = torch.meshgrid(el_offsets, az_offsets, indexing='ij')  # (H, W)
            first_bounce_directions = (
                  (torch.cos(grid_el) * torch.cos(grid_az)).unsqueeze(-1) * forward_vector
                + (torch.cos(grid_el) * torch.sin(grid_az)).unsqueeze(-1) * right_vector
                + torch.sin(grid_el).unsqueeze(-1)                        * up_vector
            )  # (H, W, 3), already unit length

            prev_directions = first_bounce_directions.reshape(-1, 3)                                    # (H*W, 3)
            prev_origins    = sensor_position.reshape(1, 3).expand(prev_directions.shape[0], -1)        # (H*W, 3)
            cumulative_legs = torch.zeros(prev_origins.shape[0], device=device)                         # (H*W,)
            cumulative_reflectivity = torch.ones(prev_origins.shape[0], device=device)                  # (H*W,) product of r of prior bounces
            outbound_range  = None  # sensor -> first hit, shared by every later bounce of that ray

            scatter_ranges[t].append(torch.empty(0, device=device))
            scatter_energies[t].append(torch.empty(0, device=device))
            scatter_azimuths[t].append(torch.empty(0, device=device))

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
                if outbound_range is not None:
                    outbound_range = outbound_range[hit_b]

                hit_b_pos = prev_origins + distance.unsqueeze(-1) * prev_directions  # (N, 3)

                # vector from each scatter back to the sensor. Unlike the planar-wavefront
                # model this is a per-scatter quantity, since the sensor is a finite point.
                to_sensor = sensor_position.reshape(1, 3) - hit_b_pos                           # (N, 3)
                return_leg = to_sensor.norm(dim=-1)                                             # (N,)
                sensor_direction = to_sensor / return_leg.unsqueeze(-1).clamp_min(1e-12)        # (N, 3)

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

                # round-trip range: the outbound leg from the sensor to the first hit, plus any
                # inter-bounce legs, plus the leg straight back to the sensor. First-bounce rays
                # start at the sensor itself, so their traced distance *is* the outbound range,
                # and b=1 gives 2x it.
                if b == 1:
                    outbound_range = distance  # cumulative_legs stays 0
                else:
                    cumulative_legs = cumulative_legs + distance
                total_range = outbound_range + cumulative_legs + return_leg  # (N,)

                # azimuth of each scatter as seen from the sensor: the angle off boresight in the
                # forward/right plane, positive toward the sensor's right vector. For a first
                # bounce this recovers the launch azimuth of the ray.
                azimuth_b = torch.atan2(dot_product(-to_sensor, right_vector),
                                        dot_product(-to_sensor, forward_vector)) * 180 / np.pi  # (N,)

                # cull occluded scatters: a hit only returns energy if its path back to the
                # sensor is unobstructed. First-bounce hits are the nearest intersection along
                # the incoming ray, so they are always visible; only later bounces can be
                # occluded (e.g. a ground point hidden behind the object). This filters what we
                # store, NOT the ray that propagates onward to the next bounce.
                if b > 1:
                    visible = points_visible_to_finite_sensor(
                        hit_b_pos, sensor_position, mesh, face_normals,
                        octree=octree, surface_bias=surface_bias, batch_size=second_bounce_batch_size,
                    )  # (N,)
                else:
                    visible = torch.ones(hit_b_pos.shape[0], dtype=torch.bool, device=device)

                scatter_ranges[t][-1]   = torch.cat((scatter_ranges[t][-1],   total_range[visible]))
                scatter_energies[t][-1] = torch.cat((scatter_energies[t][-1], energy_b[visible]))
                scatter_azimuths[t][-1] = torch.cat((scatter_azimuths[t][-1], azimuth_b[visible]))

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

                t_bounce_totals[b-1] += sync_time() - t_b_start

    t_overall = sync_time() - t_overall_start
    bounce_times = '  '.join(f'bounce{b+1}={t:.3f}s' for b, t in enumerate(t_bounce_totals))
    print(f"accumulate_scatters_side_scan: overall={t_overall:.3f}s  octree={t_octree_build:.3f}s  setup={t_setup_total:.3f}s  {bounce_times}")
    if n_nan_cos:
        print(f"accumulate_scatters_side_scan: cos(theta/2) NaN -> 0 on {n_nan_cos}/{n_cos_total} scatters "
              f"({100*n_nan_cos/n_cos_total:.3f}%); min sqrt arg={min_sqrt_arg:.3e}")
        if n_nan_arg:
            print(f"accumulate_scatters_side_scan: WARNING {n_nan_arg} of those had a NaN argument (not "
                  f"boundary rounding) — check face normals / ray directions for NaN")

    # spherical spreading: energy thins over the 4*pi*r^2 surface of an expanding sphere
    for t in range(T):
        for p in range(P):
            scatter_energies[t][p] = scatter_energies[t][p] / (
                4 * np.pi * scatter_ranges[t][p].clamp_min(1e-12)**2
            )

    # water absorption: exponential in path length, which here is already the round trip
    if water_absorption:
        for t in range(T):
            for p in range(P):
                scatter_energies[t][p] = scatter_energies[t][p] * torch.exp(
                    -water_absorption * scatter_ranges[t][p]
                )

    # apply complex value to the energy according to wavelength
    if wavelength is not None:
        for t in range(T):
            for p in range(P):
                scatter_energies[t][p] = scatter_energies[t][p] * torch.exp(
                    1j * 2 * np.pi / wavelength * scatter_ranges[t][p]
                )

    # normalized by number of rays. scatter_energies is list[T][P] of 1-D tensors, so the
    # division has to be applied per ping (dividing the list itself raises TypeError).
    # The divisor is the constant *transmitted* ray count, not the per-ping hit count,
    # which would vary with aspect and taper the track.
    for t in range(T):
        for p in range(P):
            scatter_energies[t][p] = scatter_energies[t][p]/n_ray_width/n_ray_height

    return scatter_ranges, scatter_energies, scatter_azimuths, debugging_maps if debug_gif else None
    #      list[T][P] of 1-D tensors (R' hit rays, varies per ping) x3, dict (t,p)->(H,W) or None


def target_vertices(object_mesh, material_properties):
    '''
    Vertices of the target alone, with load_mesh's 200-unit ground plane left out.

    load_mesh appends the ground's faces after the object's and gives them their own material
    row, so "same material as face 0" separates the two. Falls back to every vertex when no
    ground was added or both materials match.

    inputs:
        object_mesh (obj): pytorch3d mesh of the scene, object faces first
        material_properties (F,5): the r,a,i,d,s of each face
    outputs:
        verts (V',3): the object's vertices
    '''
    faces = object_mesh.faces_packed()
    is_object = (material_properties == material_properties[0:1]).all(dim=-1)  # (F,)
    if bool(is_object.all()):
        return object_mesh.verts_packed()
    return object_mesh.verts_packed()[faces[is_object].reshape(-1).unique()]


def side_scan_sonar_image(
    mean_sensor_position,
    track_length,
    num_pings,
    elevation_fov_deg,
    azimuth_beam_width_deg,
    object_mesh,
    face_normals,
    material_properties,
    num_ray_width, # (azimuth direction)
    num_ray_height, # elevation direction
    region_radius,

    wavelength = None,
    num_bounce = 1,
    second_bounce_batch_size = 2**9,
    spatial_bw = 64,
    spatial_fs = 64,
    window_func = 'sinc',
    use_sig_magnitude = True,

        ):

    # figure out each sensor position. The track is a straight line through
    # mean_sensor_position, running along the sensor's right vector, so the object stays off
    # to one side of the platform the whole way -- that sidelong look is what makes this a
    # side scan.
    device = mean_sensor_position.device
    line_of_sight   = torch.nn.functional.normalize(-mean_sensor_position, dim=-1)  # (3,) sensor -> origin
    world_up        = torch.tensor([0.0, 0.0, 1.0], device=device)                  # +z
    track_direction = torch.nn.functional.normalize(
        torch.linalg.cross(line_of_sight, world_up), dim=-1)                        # (3,) sensor's right

    # a ray at depression theta meets the flat seafloor at one-way range altitude/sin(theta)
    altitude             = mean_sensor_position[2]
    boresight_depression = torch.asin(-line_of_sight[2].clamp(-1.0, 1.0))
    half_fan             = np.pi / 180 * elevation_fov_deg / 2
    # clamp the shallow edge below horizontal, since a fan edge above it never meets the ground
    min_depression = (boresight_depression - half_fan).clamp_min(np.pi / 180)
    max_depression = (boresight_depression + half_fan).clamp_max(np.pi / 2)
    swath_near     = altitude / torch.sin(max_depression)                           # () one-way
    swath_far      = altitude / torch.sin(min_depression)                           # () one-way

    # an unset radius images the whole swath and nothing else
    if region_radius is None:
        region_radius = float((swath_far - swath_near) / 2)

    # fly the object's own along-track extent plus a beam footprint of run-in and run-out
    if track_length is None:
        along_track    = target_vertices(object_mesh, material_properties) @ track_direction  # (V',)
        beam_footprint = 2 * float(torch.linalg.norm(mean_sensor_position)) * np.tan(
            np.pi / 180 * azimuth_beam_width_deg / 2)
        track_length   = float(along_track.max() - along_track.min()) + 2 * beam_footprint

    # ping offsets along the track, centered on mean_sensor_position. centered_linspace puts
    # a lone ping at the track center rather than at its left edge, which is where
    # linspace(start, stop, 1) would leave it.
    ping_offsets = centered_linspace(track_length, num_pings, device)               # (P,)
    trajectory   = (mean_sensor_position.reshape(1, 3)
                    + ping_offsets.reshape(num_pings, 1) * track_direction.reshape(1, 3))  # (P,3)

    # figure out the camera matrix for each sensor position. Broadside: every ping shares the
    # single orientation built at the track center, so all the boresights are parallel and only
    # the translation column changes down the track. That is what separates this from the
    # spotlight geometry of range_angle_images, where each pose steers onto the origin and the
    # object therefore never leaves azimuth 0; here the object drifts across the beam as the
    # platform passes, and the azimuth weighting below has something to bite on.
    # Columns are (right, up, forward, center), the srn_cars layout of generate_pose_mat.
    up_vector = torch.linalg.cross(track_direction, line_of_sight)                  # (3,)
    poses = torch.zeros(num_pings, 4, 4, device=device)                             # (P,4,4)
    poses[:, :3, 0] = track_direction
    poses[:, :3, 1] = up_vector
    poses[:, :3, 2] = line_of_sight
    poses[:, :3, 3] = trajectory
    poses[:,  3, 3] = 1.0

    # the ray fan only needs to cover the angles where the beam pattern still has weight. The
    # Gaussian is down to ~0.002 at 1.5 beam widths off boresight, so 3 beam widths total spends
    # rays where the energy actually is instead of on the tails.
    fan_azimuth_fov_deg   = azimuth_beam_width_deg * 3
    fan_elevation_fov_deg = elevation_fov_deg

    # use accumulate_scatters to ray trace on the object for each camera matrix
    scatter_ranges, scatter_energies, scatter_azimuths, _ = accumulate_scatters_side_scan(
        object_mesh, face_normals, material_properties,
        poses.unsqueeze(0),                      # (1,P,4,4), one scene
        wavelength     = wavelength,
        fov_width_deg  = fan_azimuth_fov_deg,
        fov_height_deg = fan_elevation_fov_deg,
        n_ray_width    = num_ray_width,
        n_ray_height   = num_ray_height,
        num_bounce     = num_bounce,
        second_bounce_batch_size = second_bounce_batch_size,
    )  # list[T][P] of (R',) each

    # weigh received scatters according to azimuth beam width with gaussian
    scatter_energies = [
        [energy * beam_spread_weights(azimuth, azimuth_beam_width_deg)
         for energy, azimuth in zip(energies_t, azimuths_t)]
        for energies_t, azimuths_t in zip(scatter_energies, scatter_azimuths)
    ]  # list[T][P] of (R',)

    # interpolate signal, on one range window shared by every ping so the columns line up.
    # Anchored on the swath near edge, so the first row is the first return that can exist.
    window_center = (swath_near + region_radius).reshape(1)  # (1,)
    signals = []
    sample_z = []
    for ranges_t, energies_t in zip(scatter_ranges, scatter_energies):
        signals_t = []
        sample_z_t = []
        for scatter_range, scatter_energy in zip(ranges_t, energies_t):
            signal_p, sample_z_p = interpolate_signal(
                scatter_range.unsqueeze(0) / 2,   # (1,R') round trip -> one-way range
                scatter_energy.unsqueeze(0),      # (1,R')
                region_radius,
                window_center,
                spatial_bw = spatial_bw, spatial_fs = spatial_fs,
                window_func = window_func,
            )
            signals_t.append(signal_p.squeeze(0))      # (Z,)
            sample_z_t.append(sample_z_p.squeeze(0))   # (Z,)
        signals.append(torch.stack(signals_t))         # (P,Z)
        sample_z.append(torch.stack(sample_z_t))       # (P,Z)
    signals  = torch.stack(signals)   # (T,P,Z)
    sample_z = torch.stack(sample_z)  # (T,P,Z)

    # line up each signal as columns on an image. There is no aperture to focus here: a side
    # scan waterfall *is* the raw pings side by side, one column per ping, and the along-track
    # resolution is whatever the azimuth beam width already gave us.
    images = signals.transpose(-1, -2)  # (T,Z,P), rows = range, columns = ping
    if use_sig_magnitude:
        images = images.abs()  # a complex image is not displayable

    # near range at the bottom row, matching sar_render_range_angle_image
    images     = images.flip(-2)                # (T,Z,P)
    range_bins = sample_z[:, 0, :].flip(-1)     # (T,Z) every ping shares the one range window

    return images, range_bins, ping_offsets
    #      (T,Z,P), (T,Z) one-way range of each row, (P,) along-track offset of each column


def render_side_scan_image(
        obj_id = None,
        pose_num = None,
        suffix = None,
        device = 'cuda',

        override_obj_path = None,

        # track geometry
        track_length = None,
        num_pings = 128,
        elevation_fov_deg = 60.0,
        azimuth_beam_width_deg = 1.0,
        num_ray_width = 64,
        num_ray_height = 512,
        region_radius = None,

        # signal / physics
        wavelength = None,
        num_bounce = 1,
        second_bounce_batch_size = 2**9,
        spatial_bw = 64,
        spatial_fs = 64,
        window_func = 'sinc',
        use_sig_magnitude = True,

        # display
        db = True,
        db_floor = -60.0,

        # mesh
        mesh_scale = None,
        make_ground = True,
        level_with_ground = True,
        object_x_flip = False,
        object_rotate_xyz = (90.0, 0.0, 0.0),

        # material properties
        obj_raids =    (1.0, 1.0, 100.0, 0.1, 0.9),
        ground_raids = (1.0, 1.0,   1.0, 0.9, 0.1),
    ):
    '''
    Render a side scan sonar waterfall of one srn_cars object and save it beside the RGB view
    that shares its pose, the same way render_images.render_random_image saves SAR next to RGB.

    The pose file only supplies the sensor *position*: side_scan_sonar_image flies a straight
    track through it and stares off to one side, so the RGB camera and the sonar platform share
    a vantage point but not a look direction. Columns of the sonar image are pings along that
    track, rows are one-way range with near range at the bottom.

    inputs:
        obj_id (str): srn_cars object id; a random one is drawn when None
        pose_num (str): pose/rgb file stem for that object; a random one is drawn when None
        suffix (str): name for the saved files, defaults to '<pose_num>_<obj_id>'
        override_obj_path (str): render this .obj instead of the selected object's mesh
        track_length (float): along-track extent the platform flies, centered on the pose
            position; when None, the object's own along-track extent plus a beam footprint of
            run-in and run-out on each end
        num_pings (int): pings along the track, i.e. columns of the image
        elevation_fov_deg (float): vertical extent of the ray fan, about the boresight
        azimuth_beam_width_deg (float): FWHM of the along-track beam, which sets the along-track
            resolution; the ray fan spans 3x this
        region_radius (float): half the range extent imaged, starting at the near edge of the
            swath; when None, exactly the whole swath the elevation fan illuminates
        num_ray_width/num_ray_height (int): rays per ping across the fan
        db (bool): display the panel in dB relative to its brightest pixel, floored at db_floor,
            as plot_range_angle_image does. A linear stretch is all seafloor and specular glint,
            since the returns span ~100 dB
        db_floor (float): black point of the dB display
        remaining arguments: as in side_scan_sonar_image / render_images.render_random_image

    outputs:
        images (T,Z,P): the side scan image(s), near range at the bottom row
        range_bins (T,Z): one-way range of each row
        ping_offsets (P,): along-track offset of each column
    '''

    # cluster dirs, same as render_images.render_random_image
    dataset_dir = '/workspace/data/srncars/cars_train/'
    models_dir = '/workspace/data/srncars/02958343'

    if obj_id is None:
        obj_id = np.random.choice(os.listdir(dataset_dir), 1)[0]
    print('Selected object ID: ', obj_id)

    if pose_num is None:
        all_pose_nums = os.listdir(os.path.join(dataset_dir, obj_id, 'pose'))
        pose_num = np.random.choice(all_pose_nums, 1)[0].split('.')[0]
    print('Selected pose number: ', pose_num)

    if suffix is None:
        suffix = '%s_%s' % (pose_num, obj_id)

    # load image, pose, and mesh
    rgb_path  = os.path.join(dataset_dir, obj_id, 'rgb', '%s.png' % pose_num)
    pose_path = os.path.join(dataset_dir, obj_id, 'pose', '%s.txt' % pose_num)
    mesh_path = os.path.join(models_dir, obj_id, 'models', 'model_normalized.obj')
    if override_obj_path is not None:
        print('Overriding object path to %s.' % override_obj_path)
        mesh_path = override_obj_path
    rgb  = np.array(PIL.Image.open(rgb_path))[..., :3]  # (H,W,3)
    pose = np.loadtxt(pose_path).reshape(1, 4, 4).astype(np.float32)
    poses = torch.tensor(pose, device=device)  # (1,4,4)

    pose_info = extract_pose_info(poses)
    az, el = pose_info[6].item(), pose_info[5].item()
    print('Center azimuth (deg):   ', az)
    print('Center elevation (deg): ', el)

    # the track runs along cross(line of sight, +z), which collapses for a nadir look, and the
    # swath geometry needs the platform above the seafloor rather than below it
    assert el < 85.0, 'pose elevation %.1f deg is too close to vertical for a side scan track' % el
    assert el > 0.0,  'pose elevation %.1f deg puts the sensor below the seafloor' % el

    mean_sensor_position = pose_info[0].reshape(3)  # (3,) camera center of the rgb view

    mesh, normals, material_properties = load_mesh( mesh_path,
                                                    device=device,
                                                    make_ground=make_ground,
                                                    scale=mesh_scale,
                                                    obj_raids = obj_raids,
                                                    ground_raids = ground_raids,
                                                    level_with_ground = level_with_ground,
                                                    x_flip = object_x_flip,
                                                    rotate_xyz = object_rotate_xyz,
                                                )

    torch.cuda.empty_cache()
    images, range_bins, ping_offsets = side_scan_sonar_image(
        mean_sensor_position,
        track_length,
        num_pings,
        elevation_fov_deg,
        azimuth_beam_width_deg,
        mesh, normals, material_properties,
        num_ray_width,
        num_ray_height,
        region_radius,
        wavelength = wavelength,
        num_bounce = num_bounce,
        second_bounce_batch_size = second_bounce_batch_size,
        spatial_bw = spatial_bw,
        spatial_fs = spatial_fs,
        window_func = window_func,
        use_sig_magnitude = use_sig_magnitude,
    )  # (T,Z,P), (T,Z), (P,)

    # one figure per track, so a multi-track run writes one file each instead of dropping all
    # but the first
    T = images.shape[0]
    for t in range(T):
        track_suffix = '' if T == 1 else '_track%02d' % t

        # save raw amplitude so downstream stitching can use a shared color scale
        sonar_amp = images[t].detach().cpu().numpy()  # (Z,P)

        # normalize to a displayable 8-bit gray, then widen to 3 channels to sit next to the rgb
        peak = sonar_amp.max()
        if db and peak > 0:
            sonar = 20 * np.log10(np.clip(sonar_amp / peak, 10 ** (db_floor / 20), None))
            sonar = (sonar - db_floor) / -db_floor * 255.0
        else:
            span = max(float(peak - sonar_amp.min()), 1e-12)  # an all-dark image would divide by 0
            sonar = (sonar_amp - sonar_amp.min()) / span * 255.0
        sonar = np.tile(sonar[..., None], (1, 1, 3)).astype(np.uint8)  # (Z,P,3)
        sonar = cv2.resize(sonar, (rgb.shape[1], rgb.shape[0]))  # (H,W,3)
        image = np.concatenate((rgb, sonar), axis=1)

        # write azimuth and elevation at the top left of the image
        image = PIL.Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), 'Az: %.1f, El: %.1f' % (az, el), fill=(0, 0, 0))

        path = 'figures/side_scan_rgb_image_%s%s.png' % (suffix, track_suffix)
        npy_path = 'figures/side_scan_amp_%s%s.npy' % (suffix, track_suffix)
        image.save(path)
        np.save(npy_path, sonar_amp)
        print('Saved side scan and RGB image to: ', path)

    return images, range_bins, ping_offsets


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    render_side_scan_image()

