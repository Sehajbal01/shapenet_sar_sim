"""Forward looking sonar paper figure experiments. `main` runs the full suite via one call."""
import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the
# second to initialize aborts with "OMP: Error #15". Allow the duplicate.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from config import CONFIG
from forwardlookingsonar import render_forward_looking_sonar_image
from paper_figure_layout import panel_display, stitch_panels


# config.json's forward_looking_sonar_baseline
FLS_PAPER_BASELINE = dict(CONFIG['forward_looking_sonar_baseline'])


def _fls_experiments():

    # Azimuth beam width sweep -- the two-way beam each ping steers is what resolves azimuth, so
    # this takes the target from sharp to smeared. Logarithmic, as in the side scan suite.
    beam_width_vals = np.logspace(-1, 1, 5).tolist()
    beam_width = dict(
        name='beam_width',
        vary={'azimuth_beam_width_deg': beam_width_vals},
        custom_title_strings=['Beam Width: %.2f deg' % b for b in beam_width_vals],
    )

    # Spatial bandwidth sweep, the side scan suite's -- range resolution goes as 1/BW. Fs = 2*BW
    # throughout, so the panels differ by bandwidth alone.
    spatial_bw_vals = [2 ** p for p in range(2, 10)]  # 4 .. 512
    spatial_bw = dict(
        name='spatial_bw',
        vary={'spatial_bw': spatial_bw_vals,
              'spatial_fs': [2 * bw for bw in spatial_bw_vals]},
        custom_title_strings=['BW: %d, Fs: %d' % (bw, 2 * bw) for bw in spatial_bw_vals],
    )

    # dB floor sweep, display only -- -60 was set for the near seafloor's dominance at range 1.3
    db_floor_vals = [-30.0, -40.0, -50.0, -60.0, -70.0]
    db_floor = dict(
        name='db_floor',
        vary={'db_floor': db_floor_vals},
        custom_title_strings=['dB Floor: %.0f' % f for f in db_floor_vals],
    )

    # Ray count sweep, as many elevation rays as azimuth rays, in factors of 2 about the baseline's 300
    num_rays_vals = [75, 150, 300, 600, 1200]
    num_rays = dict(
        name='num_rays',
        vary={'num_ray_width': num_rays_vals,
              'num_ray_height': num_rays_vals},
        custom_title_strings=['%d Az x %d El Rays' % (n, n) for n in num_rays_vals],
    )

    # Elevation ray count sweep at 20 deg, the band's most grazing look and so its sparsest seafloor rays
    num_ray_height_vals = list(range(100, 1100, 100))  # 100 .. 1000
    num_ray_height = dict(
        name='num_ray_height',
        vary={'num_ray_height': num_ray_height_vals},
        overrides={'elevation_angle_deg': 20.0, 'ray_fov_el': 20.0,
                   'sensor_distance': 7.5},  # the side scan's distance
        ncols=5,
        custom_title_strings=['%d El Rays' % n for n in num_ray_height_vals],
    )

    # Sensor distance x pose grid, a row per distance, the image spanning 1 across the origin in azimuth
    # 5 of the baseline car's 20 in-band poses, drawn at random: (azimuth, elevation) in deg
    random_poses = {'000025': (325.2, 23.4), '000037': (256.6, 33.8), '000028': (46.2, 37.7),
                    '000029': (286.2, 37.8), '000023': (8.5, 42.1)}
    beam_per_span = FLS_PAPER_BASELINE['azimuth_beam_width_deg'] / FLS_PAPER_BASELINE['image_azimuth_range']
    grid = [(d, pose) for d in [2.5, 5.0, 7.5, 10.0] for pose in random_poses]
    grid_spans = [2 * np.degrees(np.arctan(0.5 / d)) for d, _ in grid]
    grid_beams = [beam_per_span * span for span in grid_spans]  # the baseline's ~1.75 ping spacings
    sensor_distance = dict(
        name='sensor_distance',
        vary={'sensor_distance': [d for d, _ in grid],
              'pose_num': [pose for _, pose in grid],
              'image_azimuth_range': grid_spans,
              'azimuth_beam_width_deg': grid_beams,
              'ray_fov_az': [span + 3 * beam for span, beam in zip(grid_spans, grid_beams)],  # paper's margin
              'ray_fov_el': [2 * np.degrees(np.arctan(1.0 / d)) for d, _ in grid]},  # past the +/-0.9 window
        ncols=len(random_poses),
        custom_title_strings=['Dist %.1f, Az %.0f, El %.1f deg' % ((d,) + random_poses[pose])
                              for d, pose in grid],
    )

    # The baseline at each of the 5 random poses above
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
        # spatial_bw,
        # db_floor,
        # num_rays,
        num_ray_height,
        # sensor_distance,
        # poses,
        # elevation_angle,
    ]


FLS_PAPER_EXPERIMENTS = _fls_experiments()


def multi_param_fls_experiment(param_dict, default_kwargs, experiment_name='experiment',
                               custom_title_strings=None, ncols=None):
    '''
    Run one forward looking sonar sweep and stitch its panels into a single figure.

    The clone of sss_paper_figures.multi_param_sss_experiment: render_forward_looking_sonar_image
    writes the raw amplitude and axes of each run to figures/fls_amp_<suffix>.npz, and those are
    read back here so the panels share one display treatment and keep their azimuth and range axes.

    inputs:
        param_dict (dict): parameter name -> list of values, one entry per panel. Every list must
            be the same length
        default_kwargs (dict): the baseline passed to render_forward_looking_sonar_image. Its
            'compression'/'db_floor'/'asinh_k_ratio' entries also set the stitched figure's
            display, unless param_dict varies one of them, in which case each panel is displayed
            with its own swept value
        experiment_name (str): names the saved files, and picks out this sweep's .npz files
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

    # clear this sweep's earlier output; the fls_ prefix keeps a name shared with another suite
    # (beam_width) from deleting that suite's figures
    for f in os.listdir('figures'):
        if f.startswith('fls_') and experiment_name in f and (f.endswith('.png') or f.endswith('.npz')):
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

    # generate the image for each parameter value, keeping each panel's own display kwargs, since a
    # sweep can vary compression/db_floor/asinh_k_ratio themselves
    panel_display_kwargs = []
    for i in range(n_experiments):
        kwargs = default_kwargs.copy()
        for param_name, param_vals in param_dict.items():
            kwargs[param_name] = param_vals[i]
        panel_display_kwargs.append(dict(
            compression=kwargs.get('compression', 'db'),
            db_floor=kwargs.get('db_floor', -40.0),
            asinh_k_ratio=kwargs.get('asinh_k_ratio', 0.1),
        ))

        # a numeric id keeps the panels in sweep order once they are read back off disk, and the
        # title is stripped down to filename-safe characters before it joins the suffix
        safe_title = ''.join(c if c.isalnum() or c in '-._' else '_' for c in experiment_strings[i])
        kwargs['suffix'] = '%s_%03d_%s' % (experiment_name, i, safe_title)

        print('=== %s [%d/%d] %s ===' % (experiment_name, i + 1, n_experiments, experiment_strings[i]))
        render_forward_looking_sonar_image(**kwargs)

    # find all raw amplitude arrays saved for this experiment, in sweep order. the numeric id in
    # the suffix is the panel's index i, so it also indexes panel_display_kwargs
    npz_files = [f for f in os.listdir('figures')
                 if 'fls_amp_%s' % experiment_name in f and f.endswith('.npz')]
    npz_ids = [int(f.split(experiment_name + '_')[1][:3]) for f in npz_files]
    sorted_ids, sorted_npz = zip(*sorted(zip(npz_ids, npz_files)))

    # one colorbar per panel in its own display units, and each panel keeps its own axes
    loaded = [np.load(os.path.join('figures', f)) for f in sorted_npz]
    panels, vmins, vmaxs, cbar_labels, tick_fmts, extents = [], [], [], [], [], []
    for idx, data in zip(sorted_ids, loaded):
        panel, vmin, vmax, cbar_label, tick_fmt = panel_display(np.abs(data['image']),
                                                                **panel_display_kwargs[idx])
        panels.append(panel)
        vmins.append(vmin)
        vmaxs.append(vmax)
        cbar_labels.append(cbar_label)
        tick_fmts.append(tick_fmt)
        # near range at the bottom, matching the row order of the image
        extents.append([data['ping_azimuths'][0], data['ping_azimuths'][-1],
                        data['row_ranges'][-1], data['row_ranges'][0]])

    path = 'figures/fls_stitched_%s.png' % experiment_name
    return stitch_panels(
        panels,
        experiment_strings,
        path,
        cmap='gray',
        vmin=vmins,
        vmax=vmaxs,
        cbar_label=cbar_labels,
        cbar_tick_fmt=tick_fmts,
        extents=extents,
        xlabel='Azimuth (deg)',
        ylabel='Range',
        show_axes=True,
        ncols=ncols,
    )


def run_fls_paper_experiments(experiments=FLS_PAPER_EXPERIMENTS, baseline=FLS_PAPER_BASELINE):
    paths = []
    for exp in experiments:
        kwargs = {**baseline, **exp.get('overrides', {})}
        paths.append(multi_param_fls_experiment(
            exp['vary'],
            kwargs,
            exp['name'],
            custom_title_strings=exp.get('custom_title_strings'),
            ncols=exp.get('ncols'),
        ))
    return paths


if __name__ == '__main__':
    run_fls_paper_experiments()
