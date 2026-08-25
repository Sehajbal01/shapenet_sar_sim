"""Side scan sonar paper figure experiments. `main` runs the full suite via one call."""
import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the
# second to initialize aborts with "OMP: Error #15". Allow the duplicate.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from matplotlib import pyplot as plt

from sidescansonar import render_side_scan_image
from display_compression import asinh_compress


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

    # image plane geometry
    image_width  = 128,
    image_height = 128,
    image_plane_width  = 2,
    image_plane_height = 2,

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

    # display -- the one place compression/db_floor/asinh_k_ratio are decided; both the paper
    # sweeps' stitched figures and debug_side_scan.py's render_side_scan_image call read these
    # off the baseline
    compression = 'asinh',  # 'linear' | 'db' | 'asinh'
    db_floor = -60.0,
    asinh_k_ratio = 0.005,  # k = asinh_k_ratio * ref; ref is each image's own 99.9th-percentile amplitude

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

    # asinh softening scale sweep -- k = asinh_k_ratio * ref sets where the display rolls from
    # linear to logarithmic. Small k pushes almost everything above the seafloor into the log
    # regime (dB-like, texture-heavy); k near 1 keeps most of the image linear, close to the
    # 'linear' panel. Logarithmic spacing for the same reason as beam width. The raw amplitude
    # this sweep renders is identical panel to panel -- only display changes -- but it still goes
    # through render_side_scan_image so this sweep reuses multi_param_sonar_experiment like the
    # others instead of a one-off display-only path.
    asinh_k_vals = np.logspace(-3, 0, 5).tolist()
    asinh_k = dict(
        name='asinh_k',
        vary={'asinh_k_ratio': asinh_k_vals},
        overrides={'compression': 'asinh'},
        custom_title_strings=['asinh k/ref: %.2g' % k for k in asinh_k_vals],
    )

    return [
        beam_width,
        tvg,
        asinh_k,
    ]


SONAR_PAPER_EXPERIMENTS = _sonar_experiments()


def _panel(amplitude, compression='linear', db_floor=-60.0, asinh_k_ratio=0.1):
    '''
    One panel normalized to its own peak, as linear amplitude, in dB, or asinh-compressed.

    Each panel is normalized to itself rather than to a scale shared across the figure, because
    both sweeps move the absolute level by orders of magnitude and neither one means anything.
    The beam width sweep narrows the ray fan with the beam while the energy divisor stays at the
    transmitted ray count, and the gain sweep multiplies the whole image by R^n. Against a shared
    scale most panels would come out black, so the figures compare shape and not level.

    inputs:
        amplitude (H,W): raw side scan amplitude, as saved by render_side_scan_image
        compression (str): 'linear' peak-normalizes. 'db' shows dB below the panel's own peak --
            the returns span ~100 dB, so linear is all specular glint and near range, and dB is
            what shows the seafloor and the shadow. 'asinh' arcsinh-compresses (asinh_compress)
            referenced to the panel's own 99.9th percentile amplitude -- stays linear near zero
            and logarithmic past asinh_k_ratio * that reference, so it shows seafloor texture
            like dB does but without a hard floor clipping the faint end to black
        db_floor (float): black point of the dB display, ignored unless compression == 'db'
        asinh_k_ratio (float): asinh softening scale as a fraction of the panel's own reference
            level, ignored unless compression == 'asinh'
    outputs:
        panel (H,W): amplitude in [0,1] for 'linear'/'asinh', or dB below the panel's own peak
            clipped to [db_floor,0] for 'db'
    '''
    amplitude = np.asarray(amplitude, dtype=np.float32)
    peak = float(amplitude.max())
    if peak <= 0.0:  # an all-dark panel has no peak to normalize against
        return np.full_like(amplitude, db_floor if compression == 'db' else 0.0)
    if compression == 'db':
        return np.clip(20.0 * np.log10(np.clip(amplitude / peak, 1e-12, None)), db_floor, 0.0)
    if compression == 'asinh':
        ref = float(np.percentile(amplitude, 99.9))
        if ref <= 0.0:  # nearly all-dark panel: 99.9th percentile can round to 0 even with peak > 0
            return np.zeros_like(amplitude)
        return asinh_compress(amplitude, asinh_k_ratio * ref, ref)
    return amplitude / peak


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
        default_kwargs (dict): the baseline passed to render_side_scan_image. Its
            'compression'/'db_floor'/'asinh_k_ratio' entries also set the stitched figure's
            display, so SONAR_PAPER_BASELINE is the one place that decides all three -- unless
            param_dict itself varies one of those three (e.g. an asinh_k_ratio sweep), in which
            case each panel is displayed with its own swept value instead of the baseline's
        experiment_name (str): names the saved files, and picks out this sweep's .npy files
        custom_title_strings (list[str]): panel titles, built from the varied values when None
    outputs:
        path (str): the stitched figure written
    '''
    compression = default_kwargs.get('compression', 'linear')
    db_floor = default_kwargs.get('db_floor', -60.0)
    asinh_k_ratio = default_kwargs.get('asinh_k_ratio', 0.1)

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

    panels = [_panel(np.load(os.path.join('figures', f)), **panel_display_kwargs[idx])
              for idx, f in zip(sorted_ids, sorted_npy)]

    # every panel is already normalized to its own peak, so the scale runs to that peak either way
    vmin, vmax = (db_floor, 0.0) if compression == 'db' else (0.0, 1.0)

    n_image = len(panels)
    fig, axes = plt.subplots(1, n_image, figsize=(2.2 * n_image, 2.7), squeeze=False)
    for ax, panel, title in zip(axes.flat, panels, experiment_strings):
        im = ax.imshow(panel, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    fig.subplots_adjust(right=0.9, wspace=0.05)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    if compression == 'db':
        db_ticks = [t for t in [0, -10, -20, -30, -40, -50, -60] if t >= db_floor]
        cbar.set_ticks(db_ticks)
        cbar.set_ticklabels(['%d dB' % t for t in db_ticks])
        cbar.set_label('dB below panel peak', fontsize=8)
    elif compression == 'asinh':
        if 'asinh_k_ratio' in param_dict:  # k varies per panel -- see panel titles for each value
            cbar.set_label('asinh-compressed amplitude (k/ref varies by panel)', fontsize=8)
        else:
            cbar.set_label('asinh-compressed amplitude (k=%.2g x ref)' % asinh_k_ratio, fontsize=8)
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
