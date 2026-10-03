"""Side scan sonar paper figure experiments. `main` runs the full suite via one call."""
import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the
# second to initialize aborts with "OMP: Error #15". Allow the duplicate.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from config import CONFIG
from sidescansonar import render_side_scan_image
from paper_figure_layout import panel_display, stitch_panels


# config.json's side_scan_sonar_baseline. Notes on its keys:
#   obj_id/pose_num pin the object and the pose. render_side_scan_image draws both at random, and a
#     sweep only reads as a sweep when the geometry is the one thing that does not change between
#     panels. 000000 is 40.6 deg elevation, a grazing angle that throws a visible shadow
#   sensor_distance None keeps the pose file's own range; set to override it
#   num_ray_width 1 is one boresight ray: no azimuth beam spreading outside the beam_width sweep
#   waveform 'gaussian', since the sinc pulse shows heavy side lobes. it may be a bug
#   compression/db_floor/asinh_k_ratio -- the one place these are decided; both the paper sweeps'
#     stitched figures and debug_side_scan.py's render_side_scan_image call read these off the
#     baseline. compression is 'linear' | 'db' | 'asinh'; k = asinh_k_ratio * ref, where ref is
#     each image's own 99.9th-percentile amplitude
SSS_PAPER_BASELINE = dict(CONFIG['side_scan_sonar_baseline'])


def _sss_experiments():

    # Azimuth beam width sweep -- the beam is what resolves along track, so this is the knob that
    # takes the target from a smear to a shape. Logarithmic, since a degree is a big step at 0.1
    # and a small one at 10.
    beam_width_vals = np.logspace(-1, 1, 5).tolist()
    beam_width = dict(
        name='beam_width',
        vary={'azimuth_beam_width_deg': beam_width_vals},
        overrides={'num_ray_width': 250},  # the baseline's one ray leaves the beam nothing to weight
        custom_title_strings=['Beam Width: %.2f deg' % b for b in beam_width_vals],
    )

    # Azimuth ray count sweep at a fixed 10 deg beam, to test whether beam width is ray-starved
    ray_width_beam_deg = 10.0
    ray_width_vals = [3 ** p for p in range(1, 7)]  # 3 .. 729, odd so one ray stays on boresight
    num_ray_width = dict(
        name='num_ray_width',
        vary={'num_ray_width': ray_width_vals},
        overrides={'azimuth_beam_width_deg': ray_width_beam_deg},
        custom_title_strings=['%d Az Rays, %.0f deg Beam' % (n, ray_width_beam_deg)
                              for n in ray_width_vals],
    )

    # Time varying gain sweep -- how hard the receiver ramp lifts far range against the
    # seafloor's fall with range. 0 is the raw echo, 4 is the two-way spreading loss undone.
    tvg_vals = np.linspace(-20, 30, 5).tolist()
    tvg = dict(
        name='tvg',
        vary={'tvg_exponent': tvg_vals},
        custom_title_strings=['TVG Exponent: %.1f' % t for t in tvg_vals],
    )

    # asinh softening scale sweep -- k = asinh_k_ratio * ref sets where the display rolls from
    # linear to logarithmic. Small k pushes almost everything above the seafloor into the log
    # regime (dB-like, texture-heavy); k near 1 keeps most of the image linear, close to the
    # 'linear' panel. Logarithmic spacing for the same reason as beam width. The raw amplitude
    # this sweep renders is identical panel to panel -- only display changes -- but it still goes
    # through render_side_scan_image so this sweep reuses multi_param_sss_experiment like the
    # others instead of a one-off display-only path.
    asinh_k_vals = np.logspace(-3, 0, 5).tolist()
    asinh_k = dict(
        name='asinh_k',
        vary={'asinh_k_ratio': asinh_k_vals},
        overrides={'compression': 'asinh'},
        custom_title_strings=['asinh k/ref: %.2g' % k for k in asinh_k_vals],
    )

    # Display compression comparison -- one amplitude four ways: linear, dB, the baseline's asinh,
    # and a larger asinh k at the near-linear end of the family.
    compression_k = SSS_PAPER_BASELINE['asinh_k_ratio']
    compression = dict(
        name='compression',
        vary={'compression': ['linear', 'db', 'asinh', 'asinh'],
              'asinh_k_ratio': [compression_k, compression_k, compression_k, 0.1]},
        custom_title_strings=['Linear Amplitude',
                              'dB, %.0f dB Floor' % SSS_PAPER_BASELINE['db_floor'],
                              'asinh, k/ref %.3g' % compression_k,
                              'asinh, k/ref 0.1'],
    )

    # Elevation FOV sweep -- how much of the seafloor the fan lights up around the target, from a
    # narrow beam on the object alone to a fan that fills the range window.
    elevation_fov_vals = np.linspace(5, 50, 5).tolist()
    elevation_fov = dict(
        name='elevation_fov',
        vary={'elevation_fov_deg': elevation_fov_vals},
        custom_title_strings=['Elevation FOV: %.1f deg' % e for e in elevation_fov_vals],
    )

    # Spatial bandwidth sweep -- range resolution goes as 1/bw, so this is what turns the target
    # from one bright range cell into a resolved hull and shadow. Fs = 2*bw keeps the sampling
    # ahead of the band instead of aliasing it away.
    spatial_bw_vals = [2 ** p for p in range(2, 10)]  # 4 .. 512
    spatial_bw = dict(
        name='spatial_bw',
        vary={'spatial_bw': spatial_bw_vals,
              'spatial_fs': [2 * bw for bw in spatial_bw_vals]},
        custom_title_strings=['BW: %d, Fs: %d' % (bw, 2 * bw) for bw in spatial_bw_vals],
    )

    # Transmit waveform comparison -- every waveform interpolate_signal implements, at the
    # baseline bandwidth so only the pulse changes. All five have a ~1/bw mainlobe, so what
    # separates the panels is range side lobe level, which is what the baseline's waveform
    # was chosen on.
    waveform_vals = ['sinc', 'hamming', 'gaussian', 'lfm', 'barker13']
    waveform = dict(
        name='waveform',
        vary={'waveform': waveform_vals},
        custom_title_strings=['Sinc Interpolation', 'Hamming Window', 'Gaussian Pulse',
                              'LFM Chirp', 'Barker 13'],
    )

    # The baseline at fls_paper_figures' 5 random in-band poses of the same car, so the two figures
    # show the same views: (azimuth, elevation) in deg
    random_poses = {'000025': (325.2, 23.4), '000037': (256.6, 33.8), '000028': (46.2, 37.7),
                    '000029': (286.2, 37.8), '000023': (8.5, 42.1)}
    poses = dict(
        name='random_poses',
        vary={'pose_num': list(random_poses)},
        custom_title_strings=['Az %.1f, El %.1f deg' % random_poses[pose] for pose in random_poses],
    )

    # Elevation sweep from the seafloor to overhead, at the baseline pose's azimuth, in a 4x4 grid
    elevation_vals = np.linspace(0, 90, 16).tolist()
    elevation_angle = dict(
        name='elevation_angle',
        vary={'elevation_angle_deg': elevation_vals},
        ncols=4,
        custom_title_strings=['Elevation: %.0f deg' % e for e in elevation_vals],
    )

    return [
        # beam_width,
        # num_ray_width,
        # tvg,
        # compression,
        # asinh_k,
        # elevation_fov,
        # spatial_bw,
        # waveform,
        # poses,
        elevation_angle,
    ]


SSS_PAPER_EXPERIMENTS = _sss_experiments()


def multi_param_sss_experiment(param_dict, default_kwargs, experiment_name='experiment',
                               custom_title_strings=None, ncols=None):
    '''
    Run one side scan sweep and stitch its panels into a single figure.

    The side scan analogue of render_images.multi_param_sar_experiment: render_side_scan_image
    writes the raw amplitude of each run to figures/side_scan_amp_<suffix>.npy, and those are read
    back here so the panels share one display treatment instead of each run's own saved png.

    inputs:
        param_dict (dict): parameter name -> list of values, one entry per panel. Every list must
            be the same length
        default_kwargs (dict): the baseline passed to render_side_scan_image. Its
            'compression'/'db_floor'/'asinh_k_ratio' entries also set the stitched figure's
            display, so SSS_PAPER_BASELINE is the one place that decides all three -- unless
            param_dict itself varies one of those three (e.g. an asinh_k_ratio sweep), in which
            case each panel is displayed with its own swept value instead of the baseline's
        experiment_name (str): names the saved files, and picks out this sweep's .npy files
        custom_title_strings (list[str]): panel titles, built from the varied values when None
        ncols (int): panels per row of the stitched figure, all in one row when None
    outputs:
        path (str): the stitched figure written
    '''
    lengths = [len(vals) for vals in param_dict.values()]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError("All parameter arrays must have the same length")
    n_experiments = lengths[0]

    os.makedirs('figures', exist_ok=True)

    # clear this sweep's earlier output, so a stale panel cannot survive into the new figure.
    # Matching on this suite's own prefixes as well as the name keeps a name it shares with
    # another suite (beam_width) from deleting that suite's figures out of the same directory.
    for f in os.listdir('figures'):
        if (f.startswith('side_scan_') or f.startswith('signal_columns_')) \
                and experiment_name in f and (f.endswith('.png') or f.endswith('.npy')):
            os.remove(os.path.join('figures', f))

    # create strings to title each experiment
    if custom_title_strings is None:
        experiment_strings = []
        for i in range(n_experiments):
            param_str_parts = []
            for param_name, param_vals in param_dict.items():
                try:
                    val = float(param_vals[i])
                    if val < 0.1:
                        param_str_parts.append("%s%.2e" % (param_name, val))
                    else:
                        param_str_parts.append("%s%.2f" % (param_name, val))
                except (TypeError, ValueError):
                    param_str_parts.append("%s%s" % (param_name, param_vals[i]))
            experiment_strings.append('_'.join(param_str_parts))
    else:
        experiment_strings = custom_title_strings

    # generate the image for each parameter value, keeping each panel's own display kwargs -- a
    # sweep can vary compression/db_floor/asinh_k_ratio themselves, not just physics params, so
    # the stitched figure must not assume every panel shares the baseline's display settings
    panel_display_kwargs = []
    for i in range(n_experiments):
        kwargs = default_kwargs.copy()
        for param_name, param_vals in param_dict.items():
            kwargs[param_name] = param_vals[i]
        panel_display_kwargs.append(dict(
            compression=kwargs.get('compression', 'linear'),
            db_floor=kwargs.get('db_floor', -60.0),
            asinh_k_ratio=kwargs.get('asinh_k_ratio', 0.1),
        ))

        # a numeric id keeps the panels in sweep order once they are read back off disk, and the
        # title is stripped down to filename-safe characters before it joins the suffix
        safe_title = ''.join(c if c.isalnum() or c in '-._' else '_' for c in experiment_strings[i])
        kwargs['suffix'] = '%s_%03d_%s' % (experiment_name, i, safe_title)

        print('=== %s [%d/%d] %s ===' % (experiment_name, i + 1, n_experiments, experiment_strings[i]))
        render_side_scan_image(**kwargs)

    # find all raw side scan amplitude arrays saved for this experiment, in sweep order. the
    # numeric id embedded in the suffix above is the panel's index i, so it also indexes
    # panel_display_kwargs
    npy_files = [f for f in os.listdir('figures')
                 if 'side_scan_amp_%s' % experiment_name in f and f.endswith('.npy')]
    npy_ids = [int(f.split(experiment_name + '_')[1][:3]) for f in npy_files]
    sorted_ids, sorted_npy = zip(*sorted(zip(npy_ids, npy_files)))

    # One colorbar per panel, in that panel's own display units. panel_display normalizes every
    # panel to its own peak, so what the bar carries that a shared one could not is the per-panel
    # setting the normalization hides -- that panel's raw peak, and its asinh k.
    raw_amplitudes = [np.load(os.path.join('figures', f)) for f in sorted_npy]
    panels, vmins, vmaxs, cbar_labels, tick_fmts = [], [], [], [], []
    for idx, amplitude in zip(sorted_ids, raw_amplitudes):
        panel, vmin, vmax, cbar_label, tick_fmt = panel_display(amplitude, **panel_display_kwargs[idx])
        panels.append(panel)
        vmins.append(vmin)
        vmaxs.append(vmax)
        cbar_labels.append(cbar_label)
        tick_fmts.append(tick_fmt)

    path = 'figures/side_scan_stitched_%s.png' % experiment_name
    return stitch_panels(
        panels,
        experiment_strings,
        path,
        cmap='gray',
        vmin=vmins,
        vmax=vmaxs,
        cbar_label=cbar_labels,
        cbar_tick_fmt=tick_fmts,
        ncols=ncols,
    )


def run_sss_paper_experiments(experiments=SSS_PAPER_EXPERIMENTS, baseline=SSS_PAPER_BASELINE):
    paths = []
    for exp in experiments:
        kwargs = {**baseline, **exp.get('overrides', {})}
        paths.append(multi_param_sss_experiment(
            exp['vary'],
            kwargs,
            exp['name'],
            custom_title_strings=exp.get('custom_title_strings'),
            ncols=exp.get('ncols'),
        ))
    return paths


if __name__ == '__main__':
    run_sss_paper_experiments()
