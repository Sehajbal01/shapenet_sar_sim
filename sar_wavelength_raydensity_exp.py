"""
Short-wavelength ray density figure of the paper: the signal of one SAR pulse at a short carrier
wavelength, as the ray grid densifies.

  sar_wavelength_raydensity.png     |s(z)| of the one pulse, rays per side x RAY_SCALES, at WAVELENGTH

The pulse is config.json's sar_baseline with WAVELENGTH and num_pulse 1, which puts it at the
start of the linear aperture, its most grazing look, and with snr_db off so the noise floor does
not hide the interference.
Each render's raw arrays are saved to figures/sar_wavelength_raydensity/, and -replot redraws
the figure from them without rendering.

    python sar_wavelength_raydensity_exp.py
    python sar_wavelength_raydensity_exp.py -replot
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
from generate_dataset import SAR_KEYS
from render_images import _find_split_dir, sar_render_image
from sar_paper_figures import SAR_PAPER_BASELINE


WAVELENGTH  = 0.05
RAY_SCALES  = (0.5, 1, 2, 5, 10)       # rays per side, relative to the baseline
SAR_OVERRIDES = dict(wavelength=WAVELENGTH, num_pulse=1, snr_db=None)  # the one pulse is the aperture's start
OUTPUT_DIR  = 'figures/sar_wavelength_raydensity'
FONT_SIZE   = 9


def _columns():
    '''(name, title, SAR overrides) of each panel column.'''
    cols = []
    for scale in RAY_SCALES:
        n = round(scale * SAR_PAPER_BASELINE['n_ray_width'])
        cols.append(('rays%g' % scale, r'$%d \times %d$ rays' % (n, n),
                     dict(n_ray_width=n, n_ray_height=n)))
    return cols


# ---------------------------------------------------------------------------- rendering

def render_sar(overrides, seed=0):
    '''One SAR render: the one pulse's signal, as numpy.'''
    b = {**SAR_PAPER_BASELINE, **SAR_OVERRIDES, **overrides}
    torch.manual_seed(seed)
    np.random.seed(seed)
    split_dir = _find_split_dir(b['obj_id'])
    pose = np.loadtxt(os.path.join(split_dir, b['obj_id'], 'pose', '%s.txt' % b['pose_num'])).reshape(1, 4, 4)
    mesh_path = os.path.join(SHAPENET_CARS_DIR, b['obj_id'], 'models', 'model_normalized.obj')

    _, pulses = sar_render_image(
        mesh_path, b['num_pulse'], torch.tensor(pose, dtype=torch.float32, device='cuda'),
        b['azimuth_spread'], imaging_algorithm='cbp', trajectory_type=b['trajectory_type'],
        return_pulses=True, **{k: b[k] for k in SAR_KEYS})

    return dict(
        signal          = pulses['signals'][0, 0].abs().cpu().numpy(),    # (Z,)
        sample_z        = pulses['sample_z'][0, 0].cpu().numpy(),         # (Z,)
        sensor_distance = pulses['trajectory'][0, 0].norm().item(),
        half_window     = b['region_radius'] / 2,
    )


def _path(name):
    return os.path.join(OUTPUT_DIR, '%s_sar.npz' % name)


def render_all(cols):
    for name, title, sar_kw in cols:
        print('=== %s ===' % title)
        np.savez(_path(name), **render_sar(sar_kw))


def load(name):
    with np.load(_path(name)) as f:
        return {k: f[k] for k in f.files}


# ---------------------------------------------------------------------------- figure

def save_figure(cols, path):
    '''One column per ray grid: the pulse's signal magnitude, normalized to its peak.'''
    n = len(cols)
    with plt.rc_context({'font.size': FONT_SIZE}):
        fig, axes = plt.subplots(1, n, figsize=(1.75 * n + 0.9, 2.1), squeeze=False,
                                 layout='constrained')
        for j, (name, title, _) in enumerate(cols):
            sar, ax = load(name), axes[0, j]
            d, h = float(sar['sensor_distance']), float(sar['half_window'])
            ax.plot(sar['sample_z'] - d, sar['signal'] / sar['signal'].max(), lw=0.4)
            ax.set_xlim(-h, h)
            ax.set_ylim(0, 1.05)
            ax.text(0.97, 0.95, 'peak %.2g' % sar['signal'].max(), transform=ax.transAxes,
                    ha='right', va='top', fontsize=FONT_SIZE - 2)
            ax.set_title(title, fontsize=FONT_SIZE)
            ax.set_xlabel(r'range $-$ $d$ ($\ell$)')
            if j > 0:
                ax.set_yticklabels([])
        axes[0, 0].set_ylabel(r'$|s(z)|$ / peak')
        fig.savefig(path, dpi=300)
        plt.close(fig)
    print('Saved: %s' % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-replot', action='store_true',
                        help='redraw the figure from the saved renders, without rendering')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cols = _columns()
    if not args.replot:
        render_all(cols)
    save_figure(cols, os.path.join(OUTPUT_DIR, 'sar_wavelength_raydensity.png'))


if __name__ == '__main__':
    main()
