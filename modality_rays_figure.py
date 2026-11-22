"""Ray geometry of three pulses per imaging modality, each ray colored by its one-way beam weight.

Rows are spotlight SAR, side-scan sonar and forward-looking sonar at their config.json baseline
geometry; columns are the first, middle and last pulse. Each ray is drawn from the sensor to its
first hit on the car or seafloor. Only the ray counts and the sonar beam width are illustrative.

    CUDA_VISIBLE_DEVICES=0 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 modality_rays_figure.py
"""
import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the second to
# initialize aborts with "OMP: Error #15". Allow the duplicate. Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from accumulate_scatters import centered_linspace, ray_trace_oom_safe, spherical_ray_fan
from config import CONFIG, SHAPENET_CARS_DIR, srn_split_dir
from ray_tracer_v2 import build_octree
from sidescansonar import beam_spread_weights
from signal_simulation import generate_trajectory, load_mesh
from utils import cartesian_to_spherical, extract_pose_info, generate_pose_mat


SAR = CONFIG['sar_baseline']
SSS = CONFIG['side_scan_sonar_baseline']
FLS = CONFIG['forward_looking_sonar_baseline']

# illustrative only: few enough rays to tell apart, and azimuth angles wide enough to see. The
# baselines' 0.1 deg beam and 10 deg fls fan are slivers at this scale
BEAM_WIDTH_DEG = 8.0
FLS_AZIMUTH_SCALE = 4  # widens the fls fan and its steering span together
SAR_RAYS = (7, 7)    # (height, width)
SSS_RAYS = (5, 15)
FLS_RAYS = (5, 25)

CMAP = 'viridis'
CAR_POINTS = 20000
ROW_TITLES = ['Spotlight SAR', 'Side-Scan Sonar', 'Forward-Looking Sonar']
PULSE_NAMES = ['Pulse', 'Ping', 'Ping']
# per row: camera elevation, and its azimuth relative to the sensor's; the sonar fans dip 29-52 deg, so look down on them
VIEWS = [dict(elev=28, azim_offset=-50), dict(elev=50, azim_offset=-25), dict(elev=50, azim_offset=-25)]


def pulse_indices(num_pulses):
    '''First, middle and last pulse.'''
    return [0, (num_pulses - 1) // 2, num_pulses - 1]


def surface_points(mesh, n, seed=0):
    '''n points spread evenly over the mesh surface, so the car reads as a shape rather than vertex clumps.'''
    tris = mesh.verts_packed()[mesh.faces_packed()].cpu().numpy()  # (F,3,3)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=-1)  # (F,)
    rng = np.random.default_rng(seed)
    tris = tris[rng.choice(len(tris), n, p=area / area.sum())]  # (n,3,3)
    u, v = rng.random((2, n, 1))
    flip = u + v > 1
    u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
    return tris[:, 0] + u * (tris[:, 1] - tris[:, 0]) + v * (tris[:, 2] - tris[:, 0])  # (n,3)


def baseline_pose(baseline, device):
    '''(4,4) pose file of the baseline's object and pose.'''
    path = os.path.join(srn_split_dir('cars_train'), baseline['obj_id'], 'pose', '%s.txt' % baseline['pose_num'])
    return torch.tensor(np.loadtxt(path).reshape(4, 4), dtype=torch.float32, device=device)


def sonar_position(baseline, device):
    '''The pose's camera center rescaled to the baseline's sensor_distance, as the sonar renderers do.'''
    center = extract_pose_info(baseline_pose(baseline, device))[0]  # (3,)
    return torch.nn.functional.normalize(center, dim=-1) * baseline['sensor_distance']


def sar_pulses(device):
    '''Parallel ray grid of three pulses on the linear trajectory, as accumulate_scatters builds them.'''
    trajectory = generate_trajectory(baseline_pose(SAR, device).reshape(1, 4, 4), SAR['trajectory_type'],
                                     SAR['num_pulse'], SAR['azimuth_spread'])[0][0]  # (P,3)
    n_h, n_w = SAR_RAYS
    x_offsets = torch.linspace(-SAR['grid_width'] / 2, SAR['grid_width'] / 2, n_w, device=device)    # (W,)
    y_offsets = torch.linspace(SAR['grid_height'] / 2, -SAR['grid_height'] / 2, n_h, device=device)  # (H,)
    grid_y, grid_x = torch.meshgrid(y_offsets, x_offsets, indexing='ij')  # (H,W)
    pulses = []
    for p in pulse_indices(SAR['num_pulse']):
        pose = generate_pose_mat(*cartesian_to_spherical(trajectory[p]), device=device)
        right, up, forward = pose[:3, 0], pose[:3, 1], pose[:3, 2]
        origins = (trajectory[p] + grid_x.reshape(-1, 1) * right + grid_y.reshape(-1, 1) * up)  # (H*W,3)
        corners = torch.stack([trajectory[p] + sx * SAR['grid_width'] / 2 * right + sy * SAR['grid_height'] / 2 * up
                               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])  # (4,3) the sensor plane
        pulses.append(dict(origins=origins, directions=forward.expand_as(origins),
                           weights=torch.ones(len(origins), device=device),  # no beam pattern in SAR
                           sensor=corners, label='%s %d' % (PULSE_NAMES[0], p + 1)))
    return pulses, trajectory


def sss_pulses(device):
    '''Three pings' fans along the straight track, every boresight parallel, as side_scan_sonar_image builds them.'''
    mean_position = sonar_position(SSS, device)
    line_of_sight = torch.nn.functional.normalize(-mean_position, dim=-1)
    track = torch.nn.functional.normalize(torch.linalg.cross(line_of_sight, torch.tensor([0.0, 0.0, 1.0], device=device)), dim=-1)
    up = torch.linalg.cross(track, line_of_sight)
    trajectory = mean_position + centered_linspace(SSS['track_length'], SSS['num_pings'], device).reshape(-1, 1) * track  # (P,3)
    directions, azimuths = spherical_ray_fan(track, up, line_of_sight, 3 * BEAM_WIDTH_DEG, SSS['elevation_fov_deg'],
                                             SSS_RAYS[1], SSS_RAYS[0], device)  # (H*W,3), (H*W,)
    pulses = []
    for p in pulse_indices(SSS['num_pings']):
        pulses.append(dict(origins=trajectory[p].expand_as(directions), directions=directions,
                           weights=beam_spread_weights(azimuths, BEAM_WIDTH_DEG),  # fixed beam
                           sensor=trajectory[p:p + 1], label='%s %d' % (PULSE_NAMES[1], p + 1)))
    return pulses, trajectory


def fls_pulses(device):
    '''One fan from one position, with the beam steered to three pings' azimuths.'''
    position = sonar_position(FLS, device)
    pose = generate_pose_mat(*cartesian_to_spherical(position), device=device)
    directions, azimuths = spherical_ray_fan(pose[:3, 0], pose[:3, 1], pose[:3, 2], FLS_AZIMUTH_SCALE * FLS['ray_fov_az'],
                                             FLS['ray_fov_el'], FLS_RAYS[1], FLS_RAYS[0], device)  # (H*W,3), (H*W,)
    ping_azimuths = centered_linspace(FLS_AZIMUTH_SCALE * FLS['image_azimuth_range'], FLS['num_pings'], device)  # (P,)
    pulses = []
    for p in pulse_indices(FLS['num_pings']):
        pulses.append(dict(origins=position.expand_as(directions), directions=directions,
                           weights=beam_spread_weights(azimuths - ping_azimuths[p], BEAM_WIDTH_DEG),  # steered beam
                           sensor=position.reshape(1, 3), label='%s %d' % (PULSE_NAMES[2], p + 1)))
    return pulses, position.reshape(1, 3)


def trace_ends(pulses, mesh, normals, octree):
    '''Each ray's first hit on the scene; a miss is drawn out to twice the longest hit.'''
    for pulse in pulses:
        _, distances = ray_trace_oom_safe(pulse['origins'], pulse['directions'], mesh, normals, octree=octree)  # (R,)
        distances = torch.where(distances > 0, distances, 2 * distances.max())
        pulse['ends'] = pulse['origins'] + distances.reshape(-1, 1) * pulse['directions']  # (R,3)


def draw_panel(ax, pulse, trajectory, car_points, limits, view, cmap, norm):
    ax.computed_zorder = False  # draw in the order below: ground, rays, car, sensor

    # seafloor under the ray hits only, so it doesn't spill out of the panel behind the sensor
    (x0, y0), (x1, y1) = pulse['ends'][:, :2].min(axis=0) - 0.5, pulse['ends'][:, :2].max(axis=0) + 0.5
    ax.add_collection3d(Poly3DCollection([[(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]],
                                         facecolor='0.92', edgecolor='none', zorder=0))

    # dim rays first, so the bright ones under the beam stay on top
    order = np.argsort(pulse['weights'])
    origins, ends, weights = pulse['origins'][order], pulse['ends'][order], pulse['weights'][order]
    ax.add_collection3d(Line3DCollection(np.stack((origins, ends), axis=1), colors=cmap(norm(weights)),
                                         linewidths=0.6, zorder=1))
    ax.scatter(*car_points.T, s=0.05, c='0.3', linewidths=0, depthshade=False, zorder=2)

    if len(trajectory) > 1:
        ax.plot(*trajectory.T, '--', color='0.3', linewidth=0.8, zorder=3)
    sensor = pulse['sensor']
    if len(sensor) > 1:  # the SAR sensor plane
        ax.add_collection3d(Poly3DCollection([sensor], facecolor=(0, 0, 0, 0), edgecolor='k', linewidth=0.8, zorder=5))
    else:
        ax.scatter(*sensor.T, s=12, c='k', zorder=5)

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect([hi - lo for lo, hi in limits], zoom=1.35)
    ax.view_init(**view)
    ax.set_axis_off()
    ax.set_title(pulse['label'], fontsize=14, y=0.9)


def row_limits(pulses, trajectory, margin=0.04):
    '''One equal-scale bounding box for a row, so its three panels share a view.'''
    points = np.concatenate([trajectory] + [np.concatenate((p['origins'], p['ends'], p['sensor'])) for p in pulses])
    lo, hi = points.min(axis=0), points.max(axis=0)
    lo[2] = 0.0
    pad = margin * (hi - lo).max()
    return [(l - pad, h + pad) for l, h in zip(lo, hi)]


def modality_rays_figure(path='figures/modality_rays.png', device='cuda' if torch.cuda.is_available() else 'cpu'):
    mesh_path = os.path.join(SHAPENET_CARS_DIR, SAR['obj_id'], 'models', 'model_normalized.obj')
    mesh_kwargs = dict(device=device, obj_raids=SAR['obj_raids'], ground_raids=SAR['ground_raids'],
                       x_flip=SAR['object_x_flip'], rotate_xyz=SAR['object_rotate_xyz'])
    mesh, normals, _ = load_mesh(mesh_path, make_ground=True, **mesh_kwargs)
    car_points = surface_points(load_mesh(mesh_path, make_ground=False, **mesh_kwargs)[0], CAR_POINTS)  # (N,3)
    octree = build_octree(mesh)

    rows = [sar_pulses(device), sss_pulses(device), fls_pulses(device)]
    sensor_azimuth = float(cartesian_to_spherical(sonar_position(SSS, device))[0])

    cmap, norm = plt.get_cmap(CMAP), Normalize(0.0, 1.0)
    fig = plt.figure(figsize=(9.6, 9.0))
    grid = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.0, hspace=0.02,
                            left=0.05, right=0.93, top=0.96, bottom=0.02)
    for r, (pulses, trajectory) in enumerate(rows):
        trace_ends(pulses, mesh, normals, octree)
        pulses = [{k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in p.items()} for p in pulses]
        trajectory = trajectory.detach().cpu().numpy()
        limits = row_limits(pulses, trajectory)
        for c, pulse in enumerate(pulses):
            ax = fig.add_subplot(grid[r, c], projection='3d')
            view = dict(elev=VIEWS[r]['elev'], azim=sensor_azimuth + VIEWS[r]['azim_offset'])
            draw_panel(ax, pulse, trajectory, car_points, limits, view, cmap, norm)
        fig.text(0.02, 0.96 - (r + 0.5) * 0.94 / 3, ROW_TITLES[r], rotation=90, ha='center', va='center', fontsize=15)

    cax = fig.add_subplot(grid[:, 3])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label('Beam weight', fontsize=14)
    cax.tick_params(labelsize=12)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.25 * pos.height, pos.width, 0.5 * pos.height])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    print('Saved ray figure to: ', path)


if __name__ == '__main__':
    modality_rays_figure()
