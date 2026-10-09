"""
Bandwidth and ray density figures of the paper: the SAR signal, SAR image and forward looking
sonar image as the spatial bandwidth falls, and as the ray grid densifies, plus a close-up of
the SAR energy-range scatter under the signal it is interposummed into.

  fsbw_and_raydensity_fsbw.png      B_s falling over BW_VALS, F_s = 2 B_s, baseline rays
  fsbw_and_raydensity_rays.png      rays per side x RAY_SCALES, at B_s = RAY_BW
  fsbw_and_raydensity_scatters.png  (a) the middle pulse's energy-range scatter, (b) its zoom
                                    under the signal at CLOSEUP_BWS, (c) the sparsest grid's
                                    scatters under the signal of every grid at RAY_BW

Both modalities run at their config.json baselines, which share one object and pose, except
SAR's snr_db, which is off so the noise floor does not hide the sampling artifacts. Each render's raw arrays are saved to
figures/fsbw_and_raydensity/, and -replot redraws the figures from them without rendering.

    python fsbw_and_raydensity_exp.py
    python fsbw_and_raydensity_exp.py -replot
"""
import argparse
import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the
# second to initialize aborts with "OMP: Error #15". Allow the duplicate.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from matplotlib import pyplot as plt

from config import SHAPENET_CARS_DIR
from fls_paper_figures import FLS_PAPER_BASELINE
from forwardlookingsonar import render_forward_looking_sonar_image
from generate_dataset import SAR_KEYS
from paper_figure_layout import panel_display
from render_images import _find_split_dir, sar_render_image
from sar_paper_figures import SAR_PAPER_BASELINE


BW_VALS     = (512, 256, 128, 64, 32)  # falling, so the panels clear up left to right
RAY_BW      = 64                       # the ray sweep's B_s; the track ends' sparse rows still ring here
RAY_SCALES  = (0.25, 0.5, 1, 2, 4)     # rays per side, relative to each baseline
CLOSEUP_BWS = (512, 128, 32)           # signals drawn over the scatter zoom
ZOOM        = (-0.66, -0.625)          # seafloor short of the car, as range - sensor distance
RAY_ZOOM    = (-0.70, -0.47)           # wider, to hold several of the sparsest grid's clusters
SAR_OVERRIDES = dict(snr_db=None)
OUTPUT_DIR  = 'figures/fsbw_and_raydensity'
FONT_SIZE   = 9

# the SAR and sonar columns image one car from one pose, at their own distances
assert all(SAR_PAPER_BASELINE[k] == FLS_PAPER_BASELINE[k] for k in ('obj_id', 'pose_num')), \
    'config.json sar_baseline and forward_looking_sonar_baseline must share obj_id and pose_num'


def _columns():
    '''(name, title, SAR overrides, FLS overrides) of each panel column, per sweep.'''
    fsbw = [('bw%d' % bw, r'$B_s = %d$, $F_s = %d$' % (bw, 2 * bw),
             dict(spatial_bw=bw, spatial_fs=2 * bw), dict(spatial_bw=bw, spatial_fs=2 * bw))
            for bw in BW_VALS]
    rays = []
    for scale in RAY_SCALES:
        n_sar = round(scale * SAR_PAPER_BASELINE['n_ray_width'])
        n_az  = round(scale * FLS_PAPER_BASELINE['num_ray_width'])
        n_el  = round(scale * FLS_PAPER_BASELINE['num_ray_height'])
        bw = dict(spatial_bw=RAY_BW, spatial_fs=2 * RAY_BW)
        rays.append(('rays%g' % scale, r'$%g\times$ rays' % scale,
                     dict(bw, n_ray_width=n_sar, n_ray_height=n_sar),
                     dict(bw, num_ray_width=n_az, num_ray_height=n_el)))
    return dict(fsbw=fsbw, rays=rays)


# ---------------------------------------------------------------------------- rendering

def render_sar(overrides, seed=0):
    '''One SAR render: the image, and the middle pulse's signal and scatters, as numpy.'''
    b = {**SAR_PAPER_BASELINE, **SAR_OVERRIDES, **overrides}
    torch.manual_seed(seed)
    np.random.seed(seed)
    split_dir = _find_split_dir(b['obj_id'])
    pose = np.loadtxt(os.path.join(split_dir, b['obj_id'], 'pose', '%s.txt' % b['pose_num'])).reshape(1, 4, 4)
    mesh_path = os.path.join(SHAPENET_CARS_DIR, b['obj_id'], 'models', 'model_normalized.obj')

    sar, pulses = sar_render_image(
        mesh_path, b['num_pulse'], torch.tensor(pose, dtype=torch.float32, device='cuda'),
        b['azimuth_spread'], imaging_algorithm='cbp', trajectory_type=b['trajectory_type'],
        return_pulses=True, **{k: b[k] for k in SAR_KEYS})

    p = pulses['signals'].shape[1] // 2
    return dict(
        image          = sar[0].detach().cpu().numpy(),                    # (H,W)
        signal         = pulses['signals'][0, p].abs().cpu().numpy(),      # (Z,)
        sample_z       = pulses['sample_z'][0, p].cpu().numpy(),           # (Z,)
        scatter_range  = pulses['ranges'][0][p].cpu().numpy() / 2,         # (R',) half round trip
        scatter_energy = pulses['energies'][0][p].abs().cpu().numpy(),     # (R',)
        sensor_distance = pulses['trajectory'][0, p].norm().item(),
        half_window    = b['region_radius'] / 2,
        plane          = (b['image_plane_width'], b['image_plane_height']),
        n_ray_width    = b['n_ray_width'],
    )


def render_fls(overrides, name):
    b = {**FLS_PAPER_BASELINE, **overrides, 'suffix': 'fsbw_and_raydensity_%s' % name}
    images, row_ranges, ping_azimuths = render_forward_looking_sonar_image(**b)
    return dict(image=images[0].detach().cpu().numpy(),
                row_ranges=row_ranges.cpu().numpy(),
                ping_azimuths=ping_azimuths.cpu().numpy())


def _path(sweep, name, modality):
    return os.path.join(OUTPUT_DIR, '%s_%s_%s.npz' % (sweep, name, modality))


def render_all(columns):
    for sweep, cols in columns.items():
        for name, title, sar_kw, fls_kw in cols:
            print('=== %s: %s ===' % (sweep, title))
            np.savez(_path(sweep, name, 'sar'), **render_sar(sar_kw))
            np.savez(_path(sweep, name, 'fls'), **render_fls(fls_kw, name))


def load(sweep, name, modality):
    with np.load(_path(sweep, name, modality)) as f:
        return {k: f[k] for k in f.files}


# ---------------------------------------------------------------------------- figures

def _display(baseline):
    '''A modality's image display, as its config.json baseline sets it.'''
    return {k: baseline[k] for k in ('compression', 'db_floor', 'asinh_k_ratio')}


def _draw_signal(ax, sar):
    d, h = float(sar['sensor_distance']), float(sar['half_window'])
    ax.plot(sar['sample_z'] - d, sar['signal'] / sar['signal'].max(), lw=0.4)
    ax.set_xlim(-h, h)
    ax.set_ylim(0, 1.05)
    ax.text(0.97, 0.95, 'peak %.2g' % sar['signal'].max(), transform=ax.transAxes,
            ha='right', va='top', fontsize=FONT_SIZE - 2)


def _draw_image(ax, amplitude, extent, origin, display):
    panel, vmin, vmax, _, _ = panel_display(np.abs(amplitude), **display)
    return ax.imshow(panel, cmap='gray', vmin=vmin, vmax=vmax, extent=extent, origin=origin,
                     aspect='auto')


def _colorbar_label(display):
    return 'dB re peak' if display['compression'] == 'db' else 'amplitude / peak'


def save_sweep_figure(sweep, cols, path):
    '''Rows: SAR signal, SAR image, FLS image. One column per sweep value.'''
    n = len(cols)
    sar_display, fls_display = _display(SAR_PAPER_BASELINE), _display(FLS_PAPER_BASELINE)
    with plt.rc_context({'font.size': FONT_SIZE}):
        fig, axes = plt.subplots(3, n, figsize=(1.75 * n + 0.9, 5.6), squeeze=False,
                                 layout='constrained')
        for j, (name, title, _, _) in enumerate(cols):
            sar, fls = load(sweep, name, 'sar'), load(sweep, name, 'fls')
            w, h = sar['plane']
            _draw_signal(axes[0, j], sar)
            sar_im = _draw_image(axes[1, j], sar['image'], (-w / 2, w / 2, -h / 2, h / 2), 'upper',
                                 sar_display)
            a, r = fls['ping_azimuths'], fls['row_ranges']
            fls_im = _draw_image(axes[2, j], fls['image'], (a[0], a[-1], r[-1], r[0]), 'upper',
                                 fls_display)
            axes[0, j].set_title(title, fontsize=FONT_SIZE)
            axes[0, j].set_xlabel(r'range $-$ $d$ ($\ell$)')
            axes[1, j].set_xlabel(r'cross-range ($\ell$)')
            axes[2, j].set_xlabel('azimuth (deg)')
            if j > 0:
                for ax in axes[:, j]:
                    ax.set_yticklabels([])
        axes[0, 0].set_ylabel('SAR\n' + r'$|s(z)|$ / peak')
        axes[1, 0].set_ylabel('SAR\n' + r'range ($\ell$)')
        axes[2, 0].set_ylabel('FLS\n' + r'range ($\ell$)')
        for row, im, display in ((1, sar_im, sar_display), (2, fls_im, fls_display)):
            fig.colorbar(im, ax=axes[row, :], label=_colorbar_label(display), shrink=0.9, pad=0.01)
        fig.savefig(path, dpi=300)
        plt.close(fig)
    print('Saved: %s' % path)


def _draw_zoom(ax, sar, signals, labels, zoom, scatter_label):
    '''Each scatter as a stem, under the signals normalized to their peaks in the zoom.'''
    d = float(sar['sensor_distance'])
    z, e = sar['scatter_range'] - d, sar['scatter_energy']
    keep = (z >= zoom[0]) & (z <= zoom[1])
    ax.vlines(z[keep], 0, e[keep] / e[keep].max(), color='0.4', lw=1.5, label=scatter_label)
    for sig, label in zip(signals, labels):
        sz = sig['sample_z'] - d
        in_zoom = (sz >= zoom[0]) & (sz <= zoom[1])
        ax.plot(sz[in_zoom], sig['signal'][in_zoom] / sig['signal'][in_zoom].max(), lw=1, label=label)
    ax.set_xlim(*zoom)
    ax.set_ylim(0, 1.5)  # headroom for the legend
    ax.set_xlabel(r'range $-$ $d$ ($\ell$)')
    ax.set_ylabel('normalized')
    ax.legend(fontsize=FONT_SIZE - 2, loc='upper center', ncols=3, columnspacing=0.8,
              handlelength=1.2, frameon=False)


def save_scatter_figure(cols, path):
    fsbw, rays = cols['fsbw'], cols['rays']
    base = load('fsbw', fsbw[0][0], 'sar')
    d, h = float(base['sensor_distance']), float(base['half_window'])
    with plt.rc_context({'font.size': FONT_SIZE}):
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), layout='constrained')

        # (a) the whole window, with the zoom shaded
        ax = axes[0]
        e = base['scatter_energy']
        ax.scatter(base['scatter_range'] - d, e / e.max(), s=1, linewidths=0, rasterized=True)
        # (c)'s span holds (b)'s, so its label sits over the wider part of it outside (b)
        left, right = (RAY_ZOOM[0], ZOOM[0]), (ZOOM[1], RAY_ZOOM[1])
        c_x = sum(max(left, right, key=lambda seg: seg[1] - seg[0])) / 2
        for zoom, x, letter in ((ZOOM, sum(ZOOM) / 2, 'b'), (RAY_ZOOM, c_x, 'c')):
            ax.axvspan(*zoom, color='0.5', alpha=0.25, lw=0)
            ax.text(x, 1.02, '(%s)' % letter, ha='center', va='bottom', fontsize=FONT_SIZE - 2)
        ax.set_ylim(-0.03, 1.12)
        ax.set_xlim(-h, h)
        ax.set_xlabel(r'range $-$ $d$ ($\ell$)')
        ax.set_ylabel(r'$|E_k|$ / peak')

        # (b) the baseline grid under the bandwidth sweep's signals
        assert set(CLOSEUP_BWS) <= set(BW_VALS), 'CLOSEUP_BWS must be in BW_VALS'
        _draw_zoom(axes[1], base, [load('fsbw', 'bw%d' % bw, 'sar') for bw in CLOSEUP_BWS],
                   [r'$B_s = %d$' % bw for bw in CLOSEUP_BWS], ZOOM,
                   r'$R_w = %d$' % base['n_ray_width'])

        # (c) the sparsest grid under the ray sweep's signals, all at RAY_BW
        sparse = load('rays', rays[0][0], 'sar')
        _draw_zoom(axes[2], sparse, [load('rays', name, 'sar') for name, _, _, _ in rays],
                   [title for _, title, _, _ in rays], RAY_ZOOM,
                   r'$R_w = %d$' % sparse['n_ray_width'])

        for ax, letter in zip(axes, 'abc'):
            ax.set_title('(%s)' % letter, fontsize=FONT_SIZE)
        fig.savefig(path, dpi=300)
        plt.close(fig)
    print('Saved: %s' % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-replot', action='store_true',
                        help='redraw the figures from the saved renders, without rendering')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    columns = _columns()
    if not args.replot:
        render_all(columns)
    for sweep, cols in columns.items():
        save_sweep_figure(sweep, cols, os.path.join(OUTPUT_DIR, 'fsbw_and_raydensity_%s.png' % sweep))
    save_scatter_figure(columns, os.path.join(OUTPUT_DIR, 'fsbw_and_raydensity_scatters.png'))


if __name__ == '__main__':
    main()
