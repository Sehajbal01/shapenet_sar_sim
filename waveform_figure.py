"""Transmit waveform figure: SAR and side scan sonar rows, the same waveform per column."""
import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the
# second to initialize aborts with "OMP: Error #15". Allow the duplicate.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from paper_figure_layout import panel_display, stitch_panels
from render_images import multi_param_sar_experiment
from sar_paper_figures import SAR_PAPER_BASELINE
from sss_paper_figures import SSS_PAPER_BASELINE, multi_param_sss_experiment


EXPERIMENT_NAME = 'waveform'
TITLES = ['Sinc Interpolation', 'Gaussian Pulse', 'LFM Chirp', 'Barker 13', 'LFM Chirp, 2x BW']


def _waveform_vary(baseline):
    '''The SAR suite's waveform panels: four waveforms at the baseline bandwidth, then the chirp at 2x.'''
    bw = baseline['spatial_bw']
    bw_vals = [bw] * 4 + [2 * bw]
    return {'waveform': ['sinc', 'gaussian', 'lfm', 'barker13', 'lfm'],
            'spatial_bw': bw_vals,
            'spatial_fs': [2 * b for b in bw_vals]}


def _load_panels(prefix, baseline):
    '''This sweep's saved amplitudes in panel order, in the baseline's own display.'''
    files = sorted(f for f in os.listdir('figures')
                   if f.startswith('%s_%s_' % (prefix, EXPERIMENT_NAME)) and f.endswith('.npy'))
    return [panel_display(np.load(os.path.join('figures', f)),
                          compression=baseline['compression'], db_floor=baseline['db_floor'],
                          asinh_k_ratio=baseline['asinh_k_ratio'])
            for f in files]


def waveform_figure(path='figures/waveform_stitched_sar_sss.png'):
    multi_param_sar_experiment(_waveform_vary(SAR_PAPER_BASELINE), SAR_PAPER_BASELINE,
                               EXPERIMENT_NAME, custom_title_strings=TITLES)
    multi_param_sss_experiment(_waveform_vary(SSS_PAPER_BASELINE), SSS_PAPER_BASELINE,
                               EXPERIMENT_NAME, custom_title_strings=TITLES)

    # row 1 SAR, row 2 side scan, each in its own baseline's display
    sar = _load_panels('sar_amp', SAR_PAPER_BASELINE)
    sss = _load_panels('side_scan_amp', SSS_PAPER_BASELINE)
    panels, vmins, vmaxs, labels, fmts = map(list, zip(*(sar + sss)))
    titles = ['SAR: ' + t for t in TITLES] + ['Side Scan: ' + t for t in TITLES]
    return stitch_panels(panels, titles, path, cmap='gray', vmin=vmins, vmax=vmaxs,
                         cbar_label=labels, cbar_tick_fmt=fmts, ncols=len(TITLES))


if __name__ == '__main__':
    waveform_figure()
