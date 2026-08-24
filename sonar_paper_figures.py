"""Side scan sonar paper figure experiments. `main` runs the full suite via one call."""
import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the
# second to initialize aborts with "OMP: Error #15". Allow the duplicate.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from matplotlib import pyplot as plt

from sidescansonar import render_side_scan_image


SONAR_PAPER_BASELINE = dict(
    # pin the object and the pose. render_side_scan_image draws both at random, and a sweep only
    # reads as a sweep when the geometry is the one thing that does not change between panels.
    obj_id = '100715345ee54d7ae38b52b4ee9d36a3',
    pose_num = '000000',  # 40.6 deg elevation, a grazing angle that throws a visible shadow
    sensor_distance = None,  # None keeps the pose file's own range; set to override it

    # track geometry
    track_length = 2.0,
    num_pings = 64,
    elevation_fov_deg = 45.0,
    azimuth_beam_width_deg = 0.1,
    num_ray_width = 3,
    num_ray_height = 256,
    region_radius = 1.0,

    # ground plane image geometry
    image_width  = 128,
    image_height = 128,
    image_plane_width  = 1,
    image_plane_height = 1,

    # signal / physics
    wavelength = None,
    num_bounce = 1,
    spherical_spread = False,
    water_absorption = 0.0,
    tvg_exponent = 0.0,
    spatial_bw = 64,
    spatial_fs = 32,
    window_func = 'sinc',
    use_sig_magnitude = True,

    # display -- the one place db/db_floor are decided; both the paper sweeps' stitched figures
    # and debug_side_scan.py's render_side_scan_image call read these off the baseline
    db = False,
    db_floor = -60.0,

    # mesh
    make_ground = True,
    level_with_ground = True,
    object_x_flip = False,
    object_rotate_xyz = (90.0, 0.0, 0.0),

    # material properties
    obj_raids    = (1.0, 1.0, 100.0, 0.1, 0.9),
    ground_raids = (1.0, 1.0,   1.0,   5, 0.1),
)


def _sonar_experiments():

    # Azimuth beam width sweep -- the beam is what resolves along track, so this is the knob that
    # takes the target from a smear to a shape. Logarithmic, since a degree is a big step at 0.1
    # and a small one at 10.
    beam_width_vals = np.logspace(-1, 1, 5).tolist()
    beam_width = dict(
        name='beam_width',
        vary={'azimuth_beam_width_deg': beam_width_vals},
        custom_title_strings=['Beam Width: %.2f deg' % b for b in beam_width_vals],
    )

    # Time varying gain sweep -- how hard the receiver ramp lifts far range against the
    # seafloor's fall with range. 0 is the raw echo, 4 is the two-way spreading loss undone.
    tvg_vals = np.linspace(0, 8, 5).tolist()
    tvg = dict(
        name='tvg',
        vary={'tvg_exponent': tvg_vals},
        custom_title_strings=['TVG Exponent: %.1f' % t for t in tvg_vals],
    )

    return [
        beam_width,
        tvg,
    ]


SONAR_PAPER_EXPERIMENTS = _sonar_experiments()


def _panel(amplitude, db=False, db_floor=-60.0):
    '''
    One panel normalized to its own peak, as linear amplitude or in dB.

    Each panel is normalized to itself rather than to a scale shared across the figure, because
    both sweeps move the absolute level by orders of magnitude and neither one means anything.
    The beam width sweep narrows the ray fan with the beam while the energy divisor stays at the
    transmitted ray count, and the gain sweep multiplies the whole image by R^n. Against a shared
    scale most panels would come out black, so the figures compare shape and not level.

    inputs:
        amplitude (H,W): raw side scan amplitude, as saved by render_side_scan_image
        db (bool): display in dB below the panel's peak instead of linear amplitude. The returns
            span ~100 dB, so linear is all specular glint and near range, and dB is what shows
            the seafloor and the shadow
        db_floor (float): black point of the dB display, ignored when db is False
    outputs:
        panel (H,W): amplitude in [0,1], or dB below the panel's own peak clipped to [db_floor,0]
    '''
    amplitude = np.asarray(amplitude, dtype=np.float32)
    peak = float(amplitude.max())
    if peak <= 0.0:  # an all-dark panel has no peak to normalize against
        return np.full_like(amplitude, db_floor if db else 0.0)
    if not db:
        return amplitude / peak
    return np.clip(20.0 * np.log10(np.clip(amplitude / peak, 1e-12, None)), db_floor, 0.0)


def multi_param_sonar_experiment(param_dict, default_kwargs, experiment_name='experiment',
                                 custom_title_strings=None):
    '''
    Run one side scan sweep and stitch its panels into a single figure.

    The side scan analogue of render_images.multi_param_experiment: render_side_scan_image writes
    the raw amplitude of each run to figures/side_scan_amp_<suffix>.npy, and those are read back
    here so the panels share one display treatment instead of each run's own saved png.

    inputs:
        param_dict (dict): parameter name -> list of values, one entry per panel. Every list must
            be the same length
        default_kwargs (dict): the baseline passed to render_side_scan_image. Its 'db'/'db_floor'
            entries also set the stitched figure's display, so SONAR_PAPER_BASELINE is the one
            place that decides both
        experiment_name (str): names the saved files, and picks out this sweep's .npy files
        custom_title_strings (list[str]): panel titles, built from the varied values when None
    outputs:
        path (str): the stitched figure written
    '''
    db = default_kwargs.get('db', False)
    db_floor = default_kwargs.get('db_floor', -60.0)

    lengths = [len(vals) for vals in param_dict.values()]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError("All parameter arrays must have the same length")
    n_experiments = lengths[0]

    os.makedirs('figures', exist_ok=True)

    # clear this sweep's earlier output, so a stale panel cannot survive into the new figure
    for f in os.listdir('figures'):
        if experiment_name in f and (f.endswith('.png') or f.endswith('.npy')):
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

    # generate the image for each parameter value
    for i in range(n_experiments):
        kwargs = default_kwargs.copy()
        for param_name, param_vals in param_dict.items():
            kwargs[param_name] = param_vals[i]

        # a numeric id keeps the panels in sweep order once they are read back off disk, and the
        # title is stripped down to filename-safe characters before it joins the suffix
        safe_title = ''.join(c if c.isalnum() or c in '-._' else '_' for c in experiment_strings[i])
        kwargs['suffix'] = '%s_%03d_%s' % (experiment_name, i, safe_title)

        print('=== %s [%d/%d] %s ===' % (experiment_name, i + 1, n_experiments, experiment_strings[i]))
        render_side_scan_image(**kwargs)

    # find all raw side scan amplitude arrays saved for this experiment, in sweep order
    npy_files = [f for f in os.listdir('figures')
                 if 'side_scan_amp_%s' % experiment_name in f and f.endswith('.npy')]
    npy_ids = [int(f.split(experiment_name + '_')[1][:3]) for f in npy_files]
    sorted_npy = [f for _, f in sorted(zip(npy_ids, npy_files))]

    panels = [_panel(np.load(os.path.join('figures', f)), db=db, db_floor=db_floor)
              for f in sorted_npy]

    # every panel is already normalized to its own peak, so the scale runs to that peak either way
    vmin, vmax = (db_floor, 0.0) if db else (0.0, 1.0)

    n_image = len(panels)
    fig, axes = plt.subplots(1, n_image, figsize=(2.2 * n_image, 2.7), squeeze=False)
    for ax, panel, title in zip(axes.flat, panels, experiment_strings):
        im = ax.imshow(panel, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    fig.subplots_adjust(right=0.9, wspace=0.05)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    if db:
        db_ticks = [t for t in [0, -10, -20, -30, -40, -50, -60] if t >= db_floor]
        cbar.set_ticks(db_ticks)
        cbar.set_ticklabels(['%d dB' % t for t in db_ticks])
        cbar.set_label('dB below panel peak', fontsize=8)
    else:
        cbar.set_label('amplitude / panel peak', fontsize=8)

    path = 'figures/side_scan_stitched_%s.png' % experiment_name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved stitched image to: %s' % path)
    return path


def run_sonar_paper_experiments(experiments=SONAR_PAPER_EXPERIMENTS,
                                baseline=SONAR_PAPER_BASELINE):
    paths = []
    for exp in experiments:
        kwargs = {**baseline, **exp.get('overrides', {})}
        paths.append(multi_param_sonar_experiment(
            exp['vary'],
            kwargs,
            exp['name'],
            custom_title_strings=exp.get('custom_title_strings'),
        ))
    return paths


if __name__ == '__main__':
    run_sonar_paper_experiments()
