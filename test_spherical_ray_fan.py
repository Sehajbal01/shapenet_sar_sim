'''
Visual check of accumulate_scatters.spherical_ray_fan.

Each figure draws one randomly parameterized fan two ways: the rays themselves in 3-D next
to the sensor's right/up/forward triad, and the same rays folded back into the sensor's own
azimuth/elevation frame. The second panel is the real test -- if the fan is built correctly,
recovering az = atan2(d.right, d.forward) and el = asin(d.up) has to land exactly on the
requested angle grid, no matter how the frame is oriented in the world.

run: /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 test_spherical_ray_fan.py [seed]
'''

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, registers the 3d projection
import numpy as np
import torch

from accumulate_scatters import centered_linspace, spherical_ray_fan
from utils import generate_pose_mat


N_FIGURES     = 30
N_RAY_WIDTH   = 4
N_RAY_HEIGHT  = 4
FIGURE_DIR    = 'figures'
DEVICE        = torch.device('cpu')

COLOR_RIGHT   = '#d62728'
COLOR_UP      = '#2ca02c'
COLOR_FORWARD = '#1f77b4'


def vec_str(v):
    '''Compact fixed-width printout of a 3-vector, for figure titles.'''
    return '[' + ', '.join('%+.3f' % x for x in np.asarray(v).reshape(-1)) + ']'


def random_pose_frame(rng):
    '''
    A sensor frame the way accumulate_scatters_spherical makes one: from generate_pose_mat.

    forward points from the sensor back at the world origin here, so the dashed 'to world
    origin' line in the 3-D panel should sit exactly on top of the forward arrow.
    '''
    azimuth   = rng.uniform(-180.0, 180.0)
    elevation = rng.uniform(-80.0, 80.0)
    distance  = rng.uniform(0.8, 3.0)
    pose      = generate_pose_mat(azimuth, elevation, distance, device=DEVICE)  # (4,4)
    label     = 'generate_pose_mat(az=%+.1f, el=%+.1f, d=%.2f)' % (azimuth, elevation, distance)
    return pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3], label


def random_free_frame(rng):
    '''
    A sensor frame the way sidescansonar makes one: hand-built from crossed world vectors.

    These are not tied to the world origin, so they exercise orientations generate_pose_mat
    never produces (rolled frames, forward not aimed at the scene center).
    '''
    forward = torch.tensor(rng.normal(size=3), dtype=torch.float32, device=DEVICE)
    helper  = torch.tensor(rng.normal(size=3), dtype=torch.float32, device=DEVICE)
    forward = torch.nn.functional.normalize(forward, dim=0)
    right   = torch.nn.functional.normalize(torch.linalg.cross(helper, forward), dim=0)
    up      = torch.linalg.cross(forward, right)  # already unit, right and forward are orthonormal
    origin  = torch.tensor(rng.uniform(-2.0, 2.0, size=3), dtype=torch.float32, device=DEVICE)
    return right, up, forward, origin, 'free frame (crossed random vectors)'


def fan_errors(directions, azimuths, right, up, forward, fov_width_deg, fov_height_deg):
    '''
    Fold the rays back into the sensor frame and compare against the angles that were asked for.

    outputs:
        az_recovered (H*W,), el_recovered (H*W,): degrees, measured off the returned directions
        errors (dict): max abs discrepancy of the unit length, azimuth, and elevation
    '''
    d_right   = directions @ right                                                   # (H*W,)
    d_up      = directions @ up                                                      # (H*W,)
    d_forward = directions @ forward                                                 # (H*W,)

    az_recovered = torch.atan2(d_right, d_forward) * 180 / np.pi                     # (H*W,)
    el_recovered = torch.asin(d_up.clamp(-1.0, 1.0))    * 180 / np.pi                # (H*W,)

    # the angle grid spherical_ray_fan claims to sample, rebuilt independently here
    az_wanted = centered_linspace(fov_width_deg,  N_RAY_WIDTH,  DEVICE)              # (W,)
    el_wanted = centered_linspace(fov_height_deg, N_RAY_HEIGHT, DEVICE, flip=True)   # (H,)
    grid_el, grid_az = torch.meshgrid(el_wanted, az_wanted, indexing='ij')           # (H,W)

    errors = {
        'unit': (directions.norm(dim=-1) - 1).abs().max().item(),
        'az':   (az_recovered - grid_az.reshape(-1)).abs().max().item(),
        'el':   (el_recovered - grid_el.reshape(-1)).abs().max().item(),
        'az_returned': (azimuths - grid_az.reshape(-1)).abs().max().item(),
    }
    return az_recovered, el_recovered, grid_az, grid_el, errors


def plot_rays_3d(ax, origin, right, up, forward, directions, ray_length):
    '''The fan in world coordinates, with the sensor triad drawn on the same scale.'''
    origin_np     = origin.numpy()
    directions_np = directions.numpy().reshape(N_RAY_HEIGHT, N_RAY_WIDTH, 3)
    endpoints     = origin_np + ray_length * directions_np                           # (H,W,3)

    # each ray as a segment out of the sensor, shaded by its row so the fan's top is readable
    row_colors = plt.cm.viridis(np.linspace(0.15, 0.9, N_RAY_HEIGHT))
    for i in range(N_RAY_HEIGHT):
        for j in range(N_RAY_WIDTH):
            ax.plot(*zip(origin_np, endpoints[i, j]), color=row_colors[i], lw=1.0, alpha=0.85)
    ax.scatter(endpoints[..., 0], endpoints[..., 1], endpoints[..., 2],
               c=np.repeat(row_colors, N_RAY_WIDTH, axis=0), s=18, depthshade=False)

    # lattice through the endpoints: an evenly sampled fan shows an even, unbuckled mesh
    for i in range(N_RAY_HEIGHT):
        ax.plot(endpoints[i, :, 0], endpoints[i, :, 1], endpoints[i, :, 2], color='0.55', lw=0.7)
    for j in range(N_RAY_WIDTH):
        ax.plot(endpoints[:, j, 0], endpoints[:, j, 1], endpoints[:, j, 2], color='0.55', lw=0.7)

    # ray 0 of the flattened output: should be the top-left corner, most positive elevation
    ax.scatter(*endpoints[0, 0], s=90, facecolors='none', edgecolors='k', lw=1.4)
    ax.text(*endpoints[0, 0], '  ray 0', fontsize=7)

    axis_length = 0.55 * ray_length
    for vector, color, name in ((right,   COLOR_RIGHT,   'right'),
                                (up,      COLOR_UP,      'up'),
                                (forward, COLOR_FORWARD, 'forward')):
        tip = origin_np + axis_length * vector.numpy()
        ax.quiver(*origin_np, *(axis_length * vector.numpy()), color=color, lw=2.2, arrow_length_ratio=0.15)
        ax.text(*tip, ' ' + name, color=color, fontsize=9, fontweight='bold')

    # dashed sightline to the world origin; for generate_pose_mat frames it lies on forward
    to_world_origin = -origin_np / max(np.linalg.norm(origin_np), 1e-9)
    sight_tip = origin_np + 0.75 * axis_length * to_world_origin  # stops short of the forward label
    ax.plot(*zip(origin_np, sight_tip), color='k', ls='--', lw=1.0, alpha=0.6)
    ax.text(*sight_tip, '  to world origin', fontsize=7, alpha=0.7)

    ax.scatter(*origin_np, color='k', s=40)

    # equal aspect about the sensor, so fan angles are not distorted by axis scaling
    span = 1.15 * ray_length
    ax.set_xlim(origin_np[0] - span, origin_np[0] + span)
    ax.set_ylim(origin_np[1] - span, origin_np[1] + span)
    ax.set_zlim(origin_np[2] - span, origin_np[2] + span)
    ax.set_box_aspect([1, 1, 1], zoom=1.35)  # 3d axes leave a lot of dead margin without this
    ax.set_xlabel('world x', fontsize=8)
    ax.set_ylabel('world y', fontsize=8)
    ax.set_zlabel('world z', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_title('rays in world coordinates', fontsize=10)


def plot_angles_2d(ax, az_recovered, el_recovered, grid_az, grid_el,
                   fov_width_deg, fov_height_deg, errors):
    '''The same rays back in the sensor's angle frame, against the grid they were asked for.'''
    for az in grid_az[0].numpy():
        ax.axvline(az, color='0.8', lw=0.8, zorder=0)
    for el in grid_el[:, 0].numpy():
        ax.axhline(el, color='0.8', lw=0.8, zorder=0)

    row_colors = plt.cm.viridis(np.linspace(0.15, 0.9, N_RAY_HEIGHT))
    ax.scatter(az_recovered.numpy(), el_recovered.numpy(),
               c=np.repeat(row_colors, N_RAY_WIDTH, axis=0), s=45, zorder=3)
    for k, (az, el) in enumerate(zip(az_recovered.numpy(), el_recovered.numpy())):
        ax.annotate(str(k), (az, el), textcoords='offset points', xytext=(5, 4), fontsize=7)

    ax.axvline(0.0, color='k', lw=0.8, ls=':')
    ax.axhline(0.0, color='k', lw=0.8, ls=':')
    ax.set_xlim(-0.62 * fov_width_deg,  0.62 * fov_width_deg)
    ax.set_ylim(-0.62 * fov_height_deg, 0.62 * fov_height_deg)
    ax.set_xlabel('recovered azimuth atan2(d.right, d.forward)  [deg]', fontsize=8)
    ax.set_ylabel('recovered elevation asin(d.up)  [deg]', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title('rays in sensor angle coordinates\nmax err: az %.2e deg, el %.2e deg, |d|-1 %.2e'
                 % (errors['az'], errors['el'], errors['unit']), fontsize=10)
    ax.grid(False)


def make_figure(index, rng, out_dir):
    '''Build one random fan, draw both views of it, and return its numeric errors.'''
    frame_fn = random_pose_frame if index % 2 == 0 else random_free_frame
    right, up, forward, origin, frame_label = frame_fn(rng)

    fov_width_deg  = rng.uniform(10.0, 120.0)
    fov_height_deg = rng.uniform(10.0, 120.0)

    directions, azimuths = spherical_ray_fan(right, up, forward,
                                             fov_width_deg, fov_height_deg,
                                             N_RAY_WIDTH, N_RAY_HEIGHT, DEVICE)  # (H*W,3), (H*W,)

    az_recovered, el_recovered, grid_az, grid_el, errors = fan_errors(
        directions, azimuths, right, up, forward, fov_width_deg, fov_height_deg)

    # how far the frame itself is from orthonormal, so a bad frame is not blamed on the fan
    frame  = torch.stack([right, up, forward])                                   # (3,3)
    errors['frame'] = (frame @ frame.T - torch.eye(3)).abs().max().item()

    fig  = plt.figure(figsize=(15.0, 7.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax3d = fig.add_subplot(grid[0, 0], projection='3d')
    ax2d = fig.add_subplot(grid[0, 1])
    plot_rays_3d(ax3d, origin, right, up, forward, directions, ray_length=1.0)
    plot_angles_2d(ax2d, az_recovered, el_recovered, grid_az, grid_el,
                   fov_width_deg, fov_height_deg, errors)

    fig.suptitle(
        'spherical_ray_fan #%02d   %d x %d rays   fov = %.1f deg wide x %.1f deg high   %s\n'
        'origin  = %s\n'
        'right   = %s        up = %s        forward = %s'
        % (index, N_RAY_WIDTH, N_RAY_HEIGHT, fov_width_deg, fov_height_deg, frame_label,
           vec_str(origin), vec_str(right), vec_str(up), vec_str(forward)),
        fontsize=10, family='monospace', y=0.98, va='top')

    fig.subplots_adjust(left=0.02, right=0.96, top=0.84, bottom=0.08, wspace=0.12)
    path = os.path.join(out_dir, 'spherical_ray_fan_%02d.png' % index)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, errors


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng  = np.random.RandomState(seed)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print('seed = %d, writing %d figures to %s/' % (seed, N_FIGURES, FIGURE_DIR))
    print('%-32s %10s %10s %10s %10s' % ('figure', 'az err', 'el err', '|d|-1', 'frame err'))

    worst = 0.0
    for index in range(N_FIGURES):
        path, errors = make_figure(index, rng, FIGURE_DIR)
        worst = max(worst, errors['az'], errors['el'], errors['unit'], errors['az_returned'])
        print('%-32s %10.2e %10.2e %10.2e %10.2e'
              % (os.path.basename(path), errors['az'], errors['el'], errors['unit'], errors['frame']))

    print()
    print('worst discrepancy between the fan and the angle grid it claims to sample: %.3e' % worst)
    print('PASS' if worst < 1e-3 else 'FAIL')


if __name__ == '__main__':
    main()
