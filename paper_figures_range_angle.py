"""Paper figure experiments for range angle imaging. `main` runs the full suite via one call.

The range angle counterpart of paper_figures.py. Same object, same pose, same materials, same
mesh orientation -- the only things that change are the ones that have to, because a range angle
image comes from a single pulse fanned over a field of view rather than from an aperture:

    aperture parameters (azimuth_spread, num_pulse, trajectory_*, imaging_algorithm)
        no analogue, a range angle image is one pulse from one place
    spatial_bw / spatial_fs / window_func
        no analogue, range comes straight from the scatter delay with no pulse compression
    snr_db
        no analogue, sar_render_range_angle_image has no receiver noise model
    grid_width / grid_height = 1.2
        -> fov_width_deg / fov_height_deg, the angular extent of the ray fan. 1.2 units across
           at the srn_cars camera distance of 1.3 subtends 2*atan(0.6/1.3) = 49.4 deg, so the
           50 deg fan below covers the same scene the paper's ray grid does
    image_width / image_height = 128
        -> n_angle_bins / n_range_bins
    use_sig_magnitude = False
        -> True. In render_images this flag selects coherent CBP integration over
           magnitude-detected integration. Here the angle spreading always sums the complex
           scatter energies coherently, so the flag only controls whether the final image is
           made displayable by taking its magnitude; False would return raw complex values.

num_bounce is 1 rather than the paper's 2, and the beam pattern is 0.1 deg FWHM.
"""
import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the
# second to initialize aborts with "OMP: Error #15". Allow the duplicate.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from range_angle_images import sar_render_range_angle_image, plot_range_angle_image
from paper_figure_layout import stitch_panels
from utils import extract_pose_info


RANGE_ANGLE_BASELINE = dict(
    # beam / field of view
    fov_width_deg=50.0,   # angular equivalent of the paper's 1.2 wide ray grid at range 1.3
    fov_height_deg=50.0,
    beam_width_deg=0.1,
    n_ray_width=128,
    n_ray_height=128,

    # image size stuff
    n_range_bins=128,
    n_angle_bins=128,
    region_radius=1.7,

    # scene / physics
    wavelength=0.5,
    use_sig_magnitude=True,
    num_bounce=1,
    object_x_flip=False,
    object_rotate_xyz=(90.0, 0.0, 0.0),

    # material properties
    obj_raids=(0.8, 0.0, 0.9, 0.1, 0.2),
    ground_raids=(0.5, 0.0, 0.8, 0.2, 0.5),
)


def _range_angle_experiments():

    # Beam width sweep -- how the angular resolution of the sensor smears each scatter.
    # The beam pattern peaks at 1 instead of integrating to 1, so a wider beam deposits a
    # scatter's energy into more pixels without dividing it up; each panel gets its own dB
    # reference so the sweep shows the smearing rather than that level shift.
    beam_vals = np.logspace(-1, 1, 5).tolist()
    beam_width = dict(
        name='beam_width',
        vary={'beam_width_deg': beam_vals},
        custom_title_strings=['Beam width: %.2f deg' % b for b in beam_vals],
        shared_db_reference=False,
    )

    # Field of view sweep -- how much of the scene the fan covers. angle_span_deg follows
    # fov_width_deg, so the image gets wider in angle along with the fan.
    fov_vals = [10.0, 25.0, 50.0, 75.0, 90.0]
    fov = dict(
        name='fov',
        vary={'fov_width_deg': fov_vals, 'fov_height_deg': fov_vals},
        custom_title_strings=['FOV: %.0f deg' % f for f in fov_vals],
    )

    # Ray count sweep -- how densely the fan samples the scene. The counterpart of the pulse
    # count sweep in paper_figures: too few rays and the scene is sampled as isolated streaks.
    # Scatter energies are normalized by the ray count, so the level drops as the rays go up
    # whenever the beam is too narrow for neighboring rays to overlap in a pixel; each panel
    # gets its own dB reference so the sweep shows the sampling rather than that level shift.
    ray_vals = [16, 32, 64, 128, 512]
    n_ray = dict(
        name='n_ray',
        vary={'n_ray_width': ray_vals, 'n_ray_height': ray_vals},
        custom_title_strings=['Rays: %d x %d' % (r, r) for r in ray_vals],
        shared_db_reference=False,
    )

    # Range bin sweep -- range resolution of the image, the counterpart of the Fs/BW sweep.
    range_bin_vals = [8, 16, 32, 64, 128]
    n_range_bins = dict(
        name='n_range_bins',
        vary={'n_range_bins': range_bin_vals},
        custom_title_strings=['Range bins: %d' % n for n in range_bin_vals],
    )

    # Wavelength sweep -- how carrier wavelength shapes the coherent sum within a pixel.
    wavelength_vals = [0.01, 0.05, 0.2, 0.5, 2]
    wavelength = dict(
        name='wavelength',
        vary={'wavelength': wavelength_vals},
        custom_title_strings=['Wavelength: %.2f' % w for w in wavelength_vals],
    )

    # Bounce count sweep -- what multipath off the ground adds beyond the direct return. Each
    # extra bounce only re-traces the rays that hit, so the added returns get fainter and sparser
    # rather than more numerous.
    bounce_vals = [1, 2, 3, 4, 5]
    num_bounce = dict(
        name='num_bounce',
        vary={'num_bounce': bounce_vals},
        custom_title_strings=['%d bounce' % b if b == 1 else '%d bounces' % b for b in bounce_vals],
    )

    # sphere
    scale_vals = [1/8, 1/16, 1/32, 1/64, 1/128]
    override_obj_path = os.path.join('/workspace', 'berian', 'sphere.obj')
    sphere = dict(
        name='sphere_size',
        vary={'mesh_scale': scale_vals},
        overrides={'override_obj_path': override_obj_path,
                   'make_ground': False,
                   },
        custom_title_strings=['Scale: 1', 'Scale: 1/2', 'Scale: 1/4', 'Scale: 1/8', 'Scale: 1/16'],
    )

    return [
        beam_width,
        fov,
        n_ray,
        n_range_bins,
        wavelength,
        num_bounce,
        sphere,
    ]


RANGE_ANGLE_EXPERIMENTS = _range_angle_experiments()


def render_random_range_angle_image(
        suffix=None,

        # beam / field of view
        fov_width_deg=50.0,
        fov_height_deg=50.0,
        beam_width_deg=0.1,
        n_ray_width=128,
        n_ray_height=128,

        # image size stuff
        n_range_bins=128,
        n_angle_bins=128,
        angle_span_deg=None,
        range_near=None,
        range_far=None,
        region_radius=1.7,

        # scene / physics
        wavelength=None,
        use_sig_magnitude=True,
        num_bounce=1,
        second_bounce_batch_size=2**9,
        mesh_scale=None,
        make_ground=True,
        level_with_ground=True,
        object_x_flip=False,
        object_rotate_xyz=(0.0, 0.0, 0.0),

        override_obj_path=None,

        # material properties
        obj_raids=(1.0, 1.0, 100.0, 0.1, 0.9),
        ground_raids=(1.0, 1.0, 1.0, 0.9, 0.1),

        plot=True,
        db_floor=-40.0,
        verbose=False,
    ):
    """
    Renders a range angle image of a random ShapeNet object from a random pose.

    The object and pose are drawn with np.random in the same order as
    render_images.render_random_image, so a given seed picks the same car and the same viewpoint
    as the SAR paper figures.

    outputs:
        npz_path (str): where the image, its range bins, and its angle bins were saved
    """

    # cluster dirs
    dataset_dir = '/workspace/data/srncars/cars_train/'
    models_dir = '/workspace/data/srncars/02958343'

    all_obj_id = os.listdir(dataset_dir)  # list all object IDs in the dataset
    obj_id     = np.random.choice(all_obj_id, 1)[0]  # randomly select an object ID from the dataset
    print('Selected object ID: ', obj_id)

    all_pose_paths = os.path.join(dataset_dir, obj_id, 'pose')
    all_pose_nums  = os.listdir(all_pose_paths)
    pose_num       = np.random.choice(all_pose_nums, 1)[0].split('.')[0]
    print('Selected pose number: ', pose_num)

    if suffix is None:
        suffix = '%s_%s' % (pose_num, obj_id)

    pose_path = os.path.join(dataset_dir, obj_id, 'pose', '%s.txt' % pose_num)
    mesh_path = os.path.join(models_dir, obj_id, 'models', 'model_normalized.obj')
    if override_obj_path is not None:
        print('Overriding object path to %s.' % override_obj_path)
        mesh_path = override_obj_path

    pose = np.loadtxt(pose_path).reshape(1, 4, 4).astype(np.float32)
    target_poses = torch.tensor(pose, device='cuda')  # (1,4,4)

    # print the center azimuth and elevation for the selected pose
    pose_info = extract_pose_info(target_poses)
    center_az, center_el = pose_info[6].item(), pose_info[5].item()
    print('Center azimuth (deg):   ', center_az)
    print('Center elevation (deg): ', center_el)

    images, range_bins, angle_bins_deg = sar_render_range_angle_image(
        mesh_path,
        target_poses,

        fov_width_deg = fov_width_deg,
        fov_height_deg = fov_height_deg,
        beam_width_deg = beam_width_deg,
        n_ray_width = n_ray_width,
        n_ray_height = n_ray_height,

        n_range_bins = n_range_bins,
        n_angle_bins = n_angle_bins,
        angle_span_deg = angle_span_deg,
        range_near = range_near,
        range_far = range_far,
        region_radius = region_radius,

        wavelength = wavelength,
        use_sig_magnitude = use_sig_magnitude,
        num_bounce = num_bounce,
        second_bounce_batch_size = second_bounce_batch_size,
        mesh_scale = mesh_scale,
        make_ground = make_ground,
        level_with_ground = level_with_ground,
        object_x_flip = object_x_flip,
        object_rotate_xyz = object_rotate_xyz,

        obj_raids = obj_raids,
        ground_raids = ground_raids,

        verbose = verbose,
    )
    image = images[0]  # (n_range_bins, n_angle_bins)

    # save the raw amplitude and its axes so the stitching below can share a color scale
    npz_path = 'figures/range_angle_amp_%s.npz' % suffix
    np.savez(
        npz_path,
        image=image.detach().cpu().numpy(),
        range_bins=range_bins.detach().cpu().numpy(),
        angle_bins_deg=angle_bins_deg.detach().cpu().numpy(),
    )
    print('Saved range angle image to: ', npz_path)

    if plot:
        png_path = 'figures/range_angle_image_%s.png' % suffix
        plot_range_angle_image(
            image, range_bins, angle_bins_deg, png_path,
            title='Range angle image\nAz: %.1f, El: %.1f' % (center_az, center_el),
            db=True, db_floor=db_floor,
        )
        print('Saved range angle plot to:  ', png_path)

    return npz_path


def _prepare_range_angle_plot_arrays(images, db_floor=-40.0, shared_db_reference=True):
    """
    Convert raw range angle amplitudes into dB for stitched figures.

    shared_db_reference keeps every panel against the peak of the brightest panel, so panel
    brightness is comparable -- the point of the sphere size sweep. Set it False when the sweep
    moves the absolute level by construction rather than by scene physics, in which case each
    panel is referenced to its own peak, as plot_range_angle_image does.
    """
    floor = 10 ** (db_floor / 20)
    if shared_db_reference:
        references = [float(max(im.max() for im in images))] * len(images)
    else:
        references = [float(im.max()) for im in images]

    plot_arrays = []
    for image, reference in zip(images, references):
        if reference <= 0.0:
            plot_arrays.append(np.full_like(image, db_floor, dtype=np.float32))
            continue
        amplitude = np.asarray(image, dtype=np.float32) / reference
        plot_arrays.append(20.0 * np.log10(np.clip(amplitude, floor, None)))
    return plot_arrays


def multi_param_range_angle_experiment(param_dict, default_kwargs, experiment_name="experiment",
                                       seed=8134, custom_title_strings=None, db_floor=-40.0,
                                       shared_db_reference=True):
    """
    A modular function to run range angle experiments by varying multiple parameters together.

    The range angle clone of render_images.multi_param_experiment. Panels keep their range and
    angle axes, since a sweep over the field of view or the range bins changes what those axes
    cover, and they are always plotted in dB against the peak of the brightest panel.

    Args:
        param_dict (dict): Dictionary where each key is a parameter name and value is a list/array
                           of values. All lists/arrays must have the same length.
        default_kwargs (dict): Default arguments for render_random_range_angle_image
        experiment_name (str): Name of the experiment for saving files
        seed (int): Random seed for reproducibility; fixes the object and pose across panels
        db_floor (float): darkest dB shown, relative to the dB reference
        shared_db_reference (bool): reference every panel to the brightest panel's peak, rather
                                    than to its own peak
    """
    # Verify all parameter arrays have the same length
    lengths = [len(vals) for vals in param_dict.values()]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError("All parameter arrays must have the same length")
    n_experiments = lengths[0]

    # remove this experiment's files from a previous run. Matching on the range_angle_ prefix
    # as well as the name keeps a shared name (e.g. 'wavelength') from deleting the SAR suite's
    # figures out of the same directory.
    for f in os.listdir('figures'):
        if f.startswith('range_angle_') and experiment_name in f and (f.endswith('.png') or f.endswith('.npz')):
            os.remove(os.path.join('figures', f))

    # create strings to title each experiment
    if custom_title_strings is None:
        experiment_strings = []
        for i in range(n_experiments):
            param_str_parts = []
            for param_name, param_vals in param_dict.items():
                try:
                    try:
                        val = float(param_vals[i])
                        if val < 0.1:
                            param_str_parts.append("%s%.2e" % (param_name, val))
                        else:
                            param_str_parts.append("%s%.2f" % (param_name, val))
                    except(ValueError):
                        param_str_parts.append(f"{param_name}{param_vals[i]}")

                except(TypeError):
                    param_str_parts.append(f"{param_name}{param_vals[i]}")
            experiment_strings.append('_'.join(param_str_parts))
    else:
        experiment_strings = custom_title_strings

    # generate the images for each parameter combination
    for i in range(n_experiments):
        # set the random seed, so every panel images the same car from the same pose
        np.random.seed(seed)
        torch.manual_seed(seed)

        # update the kwargs with the current parameter values
        kwargs = default_kwargs.copy()
        for param_name, param_vals in param_dict.items():
            kwargs[param_name] = param_vals[i]

        # Add a numeric ID to ensure correct sorting.
        # Sanitize the title for use in a filename: '/' (and '\\') are path
        # separators, so a title like "Scale: 1/2" must not leak into the path.
        safe_title = experiment_strings[i].replace('/', '_').replace('\\', '_')
        kwargs['suffix'] = f"{experiment_name}_{i:03d}_{safe_title}"

        # render the image with the current parameters
        render_random_range_angle_image(**kwargs)

    # find all raw range angle amplitude arrays saved for this experiment
    npz_files = [f for f in os.listdir('figures')
                 if f'range_angle_amp_{experiment_name}' in f and f.endswith('.npz')]

    # Extract the figure ID (the 3-digit number after experiment_name_) and sort by it
    npz_ids = [int(f.split(experiment_name + '_')[1][:3]) for f in npz_files]
    sorted_npz = [f for _, f in sorted(zip(npz_ids, npz_files))]

    # Load raw amplitudes; use a shared dB reference so brightness is comparable across panels
    loaded = [np.load(os.path.join('figures', f)) for f in sorted_npz]
    images = [d['image'] for d in loaded]
    plot_arrays = _prepare_range_angle_plot_arrays(
        images, db_floor=db_floor, shared_db_reference=shared_db_reference)

    # extent puts near range at the bottom, matching the row order of the image, and every panel
    # keeps its own axes because a field of view or range bin sweep changes what those axes cover
    extents = [[d['angle_bins_deg'][0], d['angle_bins_deg'][-1], d['range_bins'][-1], d['range_bins'][0]]
               for d in loaded]

    # Each panel gets its own colorbar and its own top end, so a sweep whose panels differ by tens
    # of dB shows all five rather than one bright panel beside four black ones. The floor is shared
    # and the dB reference is whatever _prepare_range_angle_plot_arrays used, so the numbers on the
    # colorbars still say how the panels compare.
    reference_label = 'dB re brightest panel' if shared_db_reference else 'dB re panel peak'

    path = f'figures/range_angle_stitched_{experiment_name}.png'
    stitch_panels(
        plot_arrays,
        experiment_strings,
        path,
        cmap='inferno',
        vmin=db_floor,
        vmax=None,
        min_span=6.0,
        cbar_label=reference_label,
        cbar_tick_fmt='%.0f dB',
        extents=extents,
        xlabel='Azimuth angle (deg)',
        ylabel='Range',
        show_axes=True,
        panel_width=2.4,
        panel_height=2.4,
    )


def run_range_angle_experiments(experiments=RANGE_ANGLE_EXPERIMENTS, baseline=RANGE_ANGLE_BASELINE,
                                db_floor=-40.0):
    for exp in experiments:
        kwargs = {**baseline, **exp.get('overrides', {})}
        multi_param_range_angle_experiment(
            exp['vary'],
            kwargs,
            exp['name'],
            custom_title_strings=exp.get('custom_title_strings'),
            db_floor=exp.get('db_floor', db_floor),
            shared_db_reference=exp.get('shared_db_reference', True),
        )


if __name__ == '__main__':
    run_range_angle_experiments()
