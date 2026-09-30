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

    # Ray azimuth fov sweep -- from half the image's azimuth span to 1.5x it. A fan narrower than the
    # image leaves the edge pings with no rays under their beam. num_ray_width follows the fov, so
    # the rays per degree stay the baseline's and only the coverage changes.
    base_image_az = FLS_PAPER_BASELINE['image_azimuth_range']
    base_fov_az = FLS_PAPER_BASELINE['ray_fov_az']
    base_n_ray = FLS_PAPER_BASELINE['num_ray_width']
    ray_fov_az_vals = [f * base_image_az for f in (0.5, 0.75, 1.0, 1.2, 1.5)]
    ray_fov_az = dict(
        name='ray_fov_az',
        vary={'ray_fov_az': ray_fov_az_vals,
              'num_ray_width': [int(round(base_n_ray * fov / base_fov_az)) for fov in ray_fov_az_vals]},
        custom_title_strings=['Ray Az FOV: %.1f deg' % fov for fov in ray_fov_az_vals],
    )

    # Spatial bandwidth sweep -- range resolution goes as 1/BW. Fs = 2*BW throughout, as in
    # paper_figures' fsbw, so the panels differ by bandwidth alone.
    bwfs_vals = [4, 16, 64, 128, 512]
    fsbw = dict(
        name='fsbw',
        vary={'spatial_bw': bwfs_vals, 'spatial_fs': [2 * bw for bw in bwfs_vals]},
        custom_title_strings=['BW: %d, Fs: %d' % (bw, 2 * bw) for bw in bwfs_vals],
    )

    return [
        beam_width,
        ray_fov_az,
        fsbw,
    ]


FLS_PAPER_EXPERIMENTS = _fls_experiments()


def multi_param_fls_experiment(param_dict, default_kwargs, experiment_name='experiment',
                               custom_title_strings=None):
    '''
    Run one forward looking sonar sweep and stitch its panels into a single figure.

    The clone of sonar_paper_figures.multi_param_sonar_experiment: render_forward_looking_sonar_image
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
        ))
    return paths


if __name__ == '__main__':
    run_fls_paper_experiments()
