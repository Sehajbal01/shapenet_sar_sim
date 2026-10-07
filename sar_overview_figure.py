"""
Figure 1 of the paper: one srn_cars pose rendered in spotlight-mode SAR at the paper baseline.

  (a) the pose's rgb image           (b) the SAR image, in range and cross-range
  (c) first-hit range of each ray    (d) first-bounce return weight E_k of each ray
  (e) energy-range scatter           (f) interposummed signal magnitude

(c)-(f) are one pulse, traced with two bounces, and (e)-(f) show the sensor distance +/- half the
region radius. The paper panels are saved one png each for latex's subfloats; -gif
instead draws the same six panels on one frame per pulse, on limits shared by all the pulses.

    python sar_overview_figure.py
    python sar_overview_figure.py -obj_id <id> -pose_num 000046 -gif
"""
import argparse
import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the
# second to initialize aborts with "OMP: Error #15". Allow the duplicate.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import PIL.Image
import torch
import tqdm
from matplotlib import pyplot as plt

from config import SHAPENET_CARS_DIR
from generate_dataset import SAR_KEYS
from paper_figure_layout import panel_display
from render_images import _find_split_dir, sar_render_image
from sar_paper_figures import SAR_PAPER_BASELINE
from signal_visualization import figure_to_frame, save_boomerang


DEFAULT_OBJ_ID   = 'e6846796e15020e02bc9f17412005422'
DEFAULT_POSE_NUM = '000046'
PANEL_SIZE = (3.0, 2.6)  # inches, one paper panel
FONT_SIZE  = 13          # readable at 0.3 of the paper's text width


def render(obj_id, pose_num, baseline=SAR_PAPER_BASELINE, seed=0, num_bounce=2):
    '''Render the pose once and return everything the panels draw, as numpy.'''
    baseline = {**baseline, 'num_bounce': num_bounce}
    torch.manual_seed(seed)
    np.random.seed(seed)
    object_dir = os.path.join(_find_split_dir(obj_id), obj_id)
    pose = np.loadtxt(os.path.join(object_dir, 'pose', f'{pose_num}.txt')).reshape(1, 4, 4)
    mesh_path = os.path.join(SHAPENET_CARS_DIR, obj_id, 'models', 'model_normalized.obj')

    sar, pulses = sar_render_image(
        mesh_path, baseline['num_pulse'], torch.tensor(pose, dtype=torch.float32, device='cuda'),
        baseline['azimuth_spread'], imaging_algorithm='cbp',
        trajectory_type=baseline['trajectory_type'], return_pulses=True,
        **{k: baseline[k] for k in SAR_KEYS})

    n_ray = baseline['n_ray_width'] * baseline['n_ray_height']
    P = pulses['trajectory'].shape[1]
    maps = [pulses['debugging_maps'][(0, p)] for p in range(P)]
    depth = np.stack([m['depth'].cpu().numpy() for m in maps])                  # (P,H,W)
    returns = sum(r.numel() for r in pulses['ranges'][0])
    first_hits = int(sum((m['depth'] >= 0).sum() for m in maps))
    print('%d bounces: %d returns from %d first hits over %d pulses, %.0f%% from later bounces'
          % (num_bounce, returns, first_hits, P, 100 * (1 - first_hits / returns)))
    return dict(
        rgb    = np.array(PIL.Image.open(os.path.join(object_dir, 'rgb', f'{pose_num}.png')))[..., :3],
        sar    = sar[0].detach().cpu().numpy(),                                 # (H,W)
        depth  = np.where(depth < 0, np.nan, depth),                            # misses are -1
        energy = np.stack([m['energy'].cpu().numpy() for m in maps]) / n_ray,   # (P,H,W), as E_k
        scatter_range  = [r.cpu().numpy() / 2 for r in pulses['ranges'][0]],    # half round trip
        scatter_energy = [e.abs().cpu().numpy() for e in pulses['energies'][0]],
        signal   = pulses['signals'][0].abs().cpu().numpy(),                    # (P,Z)
        sample_z = pulses['sample_z'][0].cpu().numpy(),                         # (P,Z)
        sensor_distance = pulses['trajectory'][0].norm(dim=-1).cpu().numpy(),   # (P,)
        half_window = baseline['region_radius'] / 2,
        grid   = (baseline['grid_width'], baseline['grid_height']),
        plane  = (baseline['image_plane_width'], baseline['image_plane_height']),
        display = {k: baseline[k] for k in ('compression', 'db_floor', 'asinh_k_ratio')},
    )


def limits(data, pulses):
    '''Color and axis limits covering the given pulses, so the gif's frames share one scale.'''
    energy = data['energy'][pulses]
    scatter_max = max(data['scatter_energy'][p][_in_window(data, p, data['scatter_range'][p])].max()
                      for p in pulses)
    signal_max = max(data['signal'][p][_in_window(data, p, data['sample_z'][p])].max()
                     for p in pulses)
    return dict(
        depth  = (np.nanmin(data['depth'][pulses]), np.nanmax(data['depth'][pulses])),
        energy = np.percentile(energy, 99.9),
        energy_power = _power(energy.max()),
        scatter_power = _power(scatter_max),
        scatter_max = scatter_max,
        signal_power = _power(signal_max),
        signal_max = signal_max,
    )


def _window(data, p):
    '''Range shown in (e) and (f): the sensor distance +/- half the region radius.'''
    d = data['sensor_distance'][p]
    return d - data['half_window'], d + data['half_window']


def _in_window(data, p, z):
    lo, hi = _window(data, p)
    return (z >= lo) & (z <= hi)


def _power(peak):
    return int(np.floor(np.log10(peak)))


def _scale_label(label, power):
    return r'%s ($\times 10^{%d}$)' % (label, power)


# ---------------------------------------------------------------------------- panels

def draw_rgb(fig, ax, data, p, lim):
    ax.imshow(data['rgb'])
    ax.axis('off')


def draw_sar(fig, ax, data, p, lim):
    # the sensor is below the image, so up is range and across is cross-range, both about the
    # scene center
    panel, vmin, vmax, label, fmt = panel_display(data['sar'], **data['display'])
    w, h = data['plane']
    im = ax.imshow(panel, cmap='gray', vmin=vmin, vmax=vmax, extent=(-w / 2, w / 2, -h / 2, h / 2))
    _ticks(ax)
    ax.set_xlabel(r'cross-range ($\ell$)')
    ax.set_ylabel(r'range ($\ell$)')
    fig.colorbar(im, ax=ax, label=label, format=fmt)


def draw_depth(fig, ax, data, p, lim):
    im = ax.imshow(data['depth'][p], cmap='gray', extent=_grid_extent(data),
                   vmin=lim['depth'][0], vmax=lim['depth'][1])
    _grid_axes(ax)
    fig.colorbar(im, ax=ax, label=r'first-hit range ($\ell$)')


def draw_energy(fig, ax, data, p, lim):
    scale = 10.0 ** lim['energy_power']
    im = ax.imshow(data['energy'][p] / scale, cmap='gray', extent=_grid_extent(data),
                   vmin=0, vmax=lim['energy'] / scale)
    _grid_axes(ax)
    fig.colorbar(im, ax=ax, extend='max',
                 label=_scale_label(r'first-bounce $E_k$', lim['energy_power']))


def draw_scatter(fig, ax, data, p, lim):
    scale = 10.0 ** lim['scatter_power']
    ax.scatter(data['scatter_range'][p], data['scatter_energy'][p] / scale, s=1, linewidths=0,
               rasterized=True)
    ax.set_xlim(*_window(data, p))
    ax.set_ylim(-0.05 * lim['scatter_max'] / scale, 1.05 * lim['scatter_max'] / scale)
    ax.set_xlabel(r'range ($\ell$)')
    ax.set_ylabel(_scale_label(r'$|E_k|$', lim['scatter_power']))


def draw_signal(fig, ax, data, p, lim):
    scale = 10.0 ** lim['signal_power']
    ax.plot(data['sample_z'][p], data['signal'][p] / scale)
    ax.set_xlim(*_window(data, p))
    ax.set_ylim(-0.05 * lim['signal_max'] / scale, 1.05 * lim['signal_max'] / scale)
    ax.set_xlabel(r'range ($\ell$)')
    ax.set_ylabel(_scale_label(r'$|s(z)|$', lim['signal_power']))


def _grid_extent(data):
    # the ray grid's own coordinates, with +y up, so the maps read like the rgb image
    w, h = data['grid']
    return (-w / 2, w / 2, -h / 2, h / 2)


def _grid_axes(ax):
    _ticks(ax)
    ax.set_xlabel(r'$x$ ($\ell$)')
    ax.set_ylabel(r'$y$ ($\ell$)')


def _ticks(ax):
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])


# (letter, paper png, drawer) in the paper's reading order. (a) is the paper's own rgb.png, which
# carries the pose's az/el, so only (b)-(f) are written for it
PANELS = (('a', None,                         draw_rgb),
          ('b', 'sar_overview.png',           draw_sar),
          ('c', 'sar_overview_depth.png',     draw_depth),
          ('d', 'sar_overview_energy.png',    draw_energy),
          ('e', 'sar_overview_scatters.png',  draw_scatter),
          ('f', 'sar_overview_signal.png',    draw_signal))


def save_paper_panels(data, pulse, output_dir):
    lim = limits(data, [pulse])
    with plt.rc_context({'font.size': FONT_SIZE}):
        for _, name, draw in PANELS:
            if name is None:
                continue
            fig, ax = plt.subplots(figsize=PANEL_SIZE)
            draw(fig, ax, data, pulse, lim)
            fig.savefig(os.path.join(output_dir, name), dpi=200, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved paper panels of pulse {pulse} to: {output_dir}')


def save_gif(data, path):
    '''Every pulse on the paper's 2x3 layout, with the paper's panel letters underneath.'''
    P = data['signal'].shape[0]
    lim = limits(data, list(range(P)))
    frames = []
    with plt.rc_context({'font.size': FONT_SIZE}):
        for p in tqdm.tqdm(range(P), desc='Creating GIF'):
            fig, axes = plt.subplots(2, 3, figsize=(3 * PANEL_SIZE[0] * 1.25, 2 * PANEL_SIZE[1] * 1.3),
                                     layout='constrained')
            for ax, (letter, _, draw) in zip(axes.flat, PANELS):
                draw(fig, ax, data, p, lim)
                ax.set_title('(%s)' % letter, y=-0.45 if letter != 'a' else -0.12, fontweight='bold')
            fig.suptitle('pulse %d of %d' % (p + 1, P))
            frames.append(figure_to_frame(fig, dpi=80))
    save_boomerang(frames, path, fps=P / 4.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-obj_id', default=DEFAULT_OBJ_ID, help='srn_cars object id, any split')
    parser.add_argument('-pose_num', default=DEFAULT_POSE_NUM, help='pose/rgb file stem')
    parser.add_argument('-pulse', type=int, default=None,
                        help='pulse shown in (c)-(f) of the paper panels; the middle one if omitted')
    parser.add_argument('-gif', action='store_true', help='also save a gif of every pulse')
    parser.add_argument('-output_dir', default='figures/sar_overview')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = render(args.obj_id, args.pose_num)
    P = data['signal'].shape[0]
    save_paper_panels(data, P // 2 if args.pulse is None else args.pulse, args.output_dir)
    if args.gif:
        save_gif(data, os.path.join(args.output_dir,
                                    'sar_overview_%s_%s.gif' % (args.obj_id, args.pose_num)))


if __name__ == '__main__':
    main()
