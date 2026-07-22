"""Paper figure experiments. `main` runs the full suite via one call."""
import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the
# second to initialize aborts with "OMP: Error #15". Allow the duplicate.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import PIL
import torch
from matplotlib import pyplot as plt

from render_images import multi_param_experiment, sar_render_image
from utils import extract_pose_info, generate_pose_mat


PAPER_BASELINE = dict(
    azimuth_spread=90,
    debug_gif=False,
    num_pulse=64,
    spatial_bw=3650 / 50, # the denominator is in mm
    spatial_fs=3650 / 50, # the denominator is in mm
    wavelength=0.5,
    use_sig_magnitude=False,
    snr_db=50,
    image_width=128,
    image_height=128,
    image_plane_width=1,
    image_plane_height=1,
    grid_width=1.2,
    grid_height=1.2,
    n_ray_width=128,
    n_ray_height=128,
    region_radius=1.7,
    obj_raids=(0.8, 0.0, 0.9, 0.1, 0.2),
    ground_raids=(0.5, 0.0, 0.8, 0.2, 0.5),
    imaging_algorithm='cbp',
    cbp_batch_size=4096,
    trajectory_type='circular',
    trajectory_noise_var=0,
    num_bounce=2,
    object_x_flip=False,
    object_rotate_xyz=(90.0, 0.0, 0.0),
)


def _paper_experiments():

    # Synthetic aperture arc length sweep — how azimuth coverage shapes the image.
    az_spread = dict(
        name='az_spread',
        vary={'azimuth_spread': np.linspace(0, 360, 8).tolist()},
    )

    # Pulse count sweep — how along-track sampling density affects the image.
    num_pulse = dict(
        name='num_pulse',
        vary={'num_pulse': np.linspace(2, 32, 8).astype(np.int32).tolist()},
    )

    # Spatial bandwidth / sample rate sweep — how BW=Fs affects range resolution.
    bwfs_vals = [4, 8, 16, 32, 64, 128, 256, 512]
    fsbw = dict(
        name='fsbw',
        vary={'spatial_bw': bwfs_vals, 'spatial_fs': bwfs_vals},
        custom_title_strings=['Fs: %d' % v for v in bwfs_vals],
        plot_db_scale=True,
    )

    # SNR sweep — sensitivity of the reconstruction to additive receiver noise.
    snr_db_vals = np.linspace(0, 22, 8).tolist()
    snrdb = dict(
        name='snrdb',
        vary={'snr_db': snr_db_vals},
        custom_title_strings=['SNR dB: %.1f' % s for s in snr_db_vals],
    )

    # Wavelength sweep with magnitude-only CBP — how carrier wavelength shapes the image.
    wavelength_vals = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2]
    wavelength = dict(
        name='wavelength',
        vary={'wavelength': wavelength_vals},
        custom_title_strings=['wavelength: %.2f' % w for w in wavelength_vals],
    )

    # Trajectory geometry comparison — linear (stripmap-like) vs circular (spotlight).
    trajectory_type = dict(
        name='trajectory_type',
        vary={'trajectory_type': ['linear', 'circular']},
        custom_title_strings=['Linear Trajectory', 'Circular Trajectory'],
    )

    # Trajectory noise sweep — how sensor position error along the path degrades the image.
    noise_vals = [0] + (10 ** np.linspace(-5, -2, 7, endpoint=True)).tolist()
    trajectory_noise_var = dict(
        name='trajectory_noise_var',
        vary={'trajectory_noise_var': noise_vals},
        custom_title_strings=['Turbulence: %.2e' % v for v in noise_vals],
    )

    # Transmit-waveform comparison — how the pulse / range-compression window shapes
    # the image. window_func selects the effective range window used inside
    # interpolate_signal: an ideal sinc, a Gaussian pulse, and the matched-filter
    # responses of an LFM chirp and a Barker-13 phase code.
    waveform_vals = ['sinc', 'gaussian', 'lfm', 'barker13']
    waveform = dict(
        name='waveform',
        vary={'window_func': waveform_vals},
        custom_title_strings=['Sinc Interpolation', 'Gaussian Pulse', 'LFM Chirp', 'Barker 13'],
    )

    return [
        # az_spread,
        # num_pulse,
        # fsbw,
        # snrdb,
        # wavelength,
        # trajectory_type,
        # trajectory_noise_var,
        waveform,
    ]


PAPER_EXPERIMENTS = _paper_experiments()


def _normalize_sar_for_display(sar_image, rgb_shape):
    sar_image = sar_image.squeeze(0).detach().cpu().numpy()
    sar_image = np.asarray(sar_image, dtype=np.float32)
    if sar_image.ndim == 2:
        sar_image = np.repeat(sar_image[..., None], 3, axis=2)
    elif sar_image.ndim == 3 and sar_image.shape[2] == 1:
        sar_image = np.repeat(sar_image, 3, axis=2)

    sar_image = np.clip(sar_image, 0.0, None)
    if sar_image.size:
        sar_min = sar_image.min()
        sar_max = sar_image.max()
        if sar_max > sar_min:
            sar_image = (sar_image - sar_min) / (sar_max - sar_min)
        else:
            sar_image = np.zeros_like(sar_image)
    else:
        sar_image = np.zeros_like(sar_image)

    sar_image = (sar_image * 255.0).astype(np.uint8)
    sar_image = cv2.resize(
        sar_image,
        (rgb_shape[1], rgb_shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    return sar_image


_DEFAULT_TITLE = object()


def render_obj_sar_image(
    obj_path,
    elevation_deg,
    azimuth_deg,
    scale=None,
    distance=1.3,
    output_path='figures/obj_sar_render.png',
    baseline=PAPER_BASELINE,
    imaging_algorithm=None,
    title=_DEFAULT_TITLE,
):
    """Render a SAR image of an arbitrary .obj file from a chosen viewing angle.

    Args:
        obj_path (str): path to the .obj mesh to render.
        elevation_deg (float): camera elevation angle in degrees.
        azimuth_deg (float): camera azimuth angle in degrees.
        scale (float or None): optional uniform scale applied to the mesh
            (forwarded to sar_render_image's mesh_scale). None leaves it unscaled.
        distance (float): camera distance from the origin.
        output_path (str): where to save the rendered SAR PNG.
        baseline (dict): default rendering parameters (defaults to PAPER_BASELINE).
        imaging_algorithm (str or None): override the baseline imaging algorithm.
        title (str or None): axis title. Defaults to an "Az/El" string; pass
            None or '' to render without any title.

    Returns:
        The output path the SAR image was saved to.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # build a single camera pose from the chosen viewing angles
    target_poses = generate_pose_mat(
        azimuth_deg, elevation_deg, distance, device=device
    ).reshape(1, 4, 4)

    # pull the shared rendering parameters from the baseline config
    render_kwargs = {
        'spatial_bw': baseline['spatial_bw'],
        'spatial_fs': baseline['spatial_fs'],
        'snr_db': baseline['snr_db'],
        'wavelength': baseline['wavelength'],
        'use_sig_magnitude': baseline['use_sig_magnitude'],
        'cbp_batch_size': baseline['cbp_batch_size'],
        'trajectory_type': baseline['trajectory_type'],
        'trajectory_noise_var': baseline['trajectory_noise_var'],
        'num_bounce': baseline['num_bounce'],
        'object_x_flip': baseline['object_x_flip'],
        'object_rotate_xyz': baseline['object_rotate_xyz'],
        'image_width': baseline['image_width'],
        'image_height': baseline['image_height'],
        'image_plane_width': baseline['image_plane_width'],
        'image_plane_height': baseline['image_plane_height'],
        'grid_width': baseline['grid_width'],
        'grid_height': baseline['grid_height'],
        'n_ray_width': baseline['n_ray_width'],
        'n_ray_height': baseline['n_ray_height'],
        'region_radius': baseline['region_radius'],
        'obj_raids': baseline['obj_raids'],
        'ground_raids': baseline['ground_raids'],
    }

    sar_image = sar_render_image(
        obj_path,
        baseline['num_pulse'],
        target_poses,
        baseline['azimuth_spread'],
        mesh_scale=scale,
        imaging_algorithm=imaging_algorithm or baseline['imaging_algorithm'],
        **render_kwargs,
    )

    # normalize to an 8-bit grayscale image at native resolution and save
    sar_vis = _normalize_sar_for_display(
        sar_image, (baseline['image_height'], baseline['image_width'])
    )

    if title is _DEFAULT_TITLE:
        title = 'Az: %.1f, El: %.1f' % (azimuth_deg, elevation_deg)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(sar_vis, cmap='gray')
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis('off')
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved SAR render to: %s' % output_path)
    return output_path


def render_sphere_size_comparison(
    obj_path='/workspace/berian/sphere.obj',
    elevation_deg=30.0,
    azimuth_deg=45.0,
    output_path='figures/sphere_size_comparison.png',
    baseline=PAPER_BASELINE,
    num_sizes=5,
):
    """Render an .obj at successively halved scales and stitch them side-by-side.

    The first (largest) render uses quarter scale (scale=0.25); each subsequent
    render halves the scale (1/8, 1/16, 1/32, 1/64, ...), for num_sizes renders
    total. The absolute scale is arbitrary (quarter scale is simply the largest
    that stays fully within the frame), so the paper presents these panels as
    the relative sequence 1, 1/2, 1/4, 1/8, 1/16. The ground plane is dropped
    and the baseline pulse count is quadrupled for these renders. The panels are
    concatenated horizontally into a single comparison figure with no titles.

    Args:
        obj_path (str): path to the .obj mesh to render (defaults to sphere.obj).
        elevation_deg (float): camera elevation angle in degrees.
        azimuth_deg (float): camera azimuth angle in degrees.
        output_path (str): where to save the stitched comparison PNG.
        baseline (dict): default rendering parameters (defaults to PAPER_BASELINE).
        num_sizes (int): number of successively halved renders (defaults to 5).

    Returns:
        The output path the stitched comparison image was saved to.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Drop the ground plane (ground_raids=None) and quadruple the pulse count.
    baseline = dict(baseline, ground_raids=None, num_pulse=baseline['num_pulse'] * 4)

    # Start at quarter scale, halving for each successive render (1/4, 1/8, ...).
    # Presented in the paper as the relative sequence 1, 1/2, 1/4, 1/8, 1/16.
    scales = [0.25 * 0.5 ** i for i in range(num_sizes)]

    panel_paths = []
    for i, scale in enumerate(scales):
        panel_paths.append(render_obj_sar_image(
            obj_path, elevation_deg, azimuth_deg,
            scale=scale,
            output_path='figures/sphere_size_%d.png' % i,
            baseline=baseline,
            title='',
        ))

    # Load all renders and match heights before stitching horizontally.
    imgs = [cv2.imread(p) for p in panel_paths]
    target_h = max(im.shape[0] for im in imgs)

    def _match_height(img):
        h, w = img.shape[:2]
        if h == target_h:
            return img
        return cv2.resize(img, (int(round(w * target_h / h)), target_h))

    imgs = [_match_height(im) for im in imgs]

    separator = 255 * np.ones((target_h, 10, 3), dtype=np.uint8)
    panels = []
    for i, im in enumerate(imgs):
        if i > 0:
            panels.append(separator)
        panels.append(im)
    stitched = np.hstack(panels)

    cv2.imwrite(output_path, stitched)
    print('Saved sphere size comparison to: %s' % output_path)
    return output_path


def generate_linear_sar_comparison_figure(
    num_examples=4,
    output_path='figures/linear_sar_comparison.png',
    baseline=PAPER_BASELINE,
    seed=8134,
    min_elevation_deg=20,
):
    """Create a 4-row figure with RGB, spotlight, and strip-map SAR panels."""
    dataset_dir = '/workspace/data/srncars/cars_train/'
    models_dir = '/workspace/data/srncars/02958343'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rng = np.random.RandomState(seed)
    comparison_kwargs = dict(baseline)
    comparison_kwargs['trajectory_type'] = 'linear'

    fig, axes = plt.subplots(num_examples, 3, figsize=(9, 3.2 * num_examples), squeeze=False)

    for row_idx in range(num_examples):
        obj_ids = sorted(os.listdir(dataset_dir))
        obj_id = obj_ids[rng.randint(0, len(obj_ids))]

        pose_dir = os.path.join(dataset_dir, obj_id, 'pose')
        pose_files = sorted(os.listdir(pose_dir))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Keep drawing random poses until the camera elevation is high enough.
        while True:
            pose_file = pose_files[rng.randint(0, len(pose_files))]
            pose_num = os.path.splitext(pose_file)[0]
            pose_path = os.path.join(pose_dir, pose_file)
            pose = np.loadtxt(pose_path).reshape(1, 4, 4).astype(np.float32)
            target_poses = torch.tensor(pose, device=device)
            elevation_deg = extract_pose_info(target_poses)[5].item()
            if elevation_deg >= min_elevation_deg:
                break

        rgb_path = os.path.join(dataset_dir, obj_id, 'rgb', f'{pose_num}.png')
        mesh_path = os.path.join(models_dir, obj_id, 'models', 'model_normalized.obj')

        rgb = np.array(PIL.Image.open(rgb_path))[..., :3]

        render_kwargs = {
            'spatial_bw': comparison_kwargs['spatial_bw'],
            'spatial_fs': comparison_kwargs['spatial_fs'],
            'snr_db': comparison_kwargs['snr_db'],
            'wavelength': comparison_kwargs['wavelength'],
            'use_sig_magnitude': comparison_kwargs['use_sig_magnitude'],
            'imaging_algorithm': 'cbp',
            'cbp_batch_size': comparison_kwargs['cbp_batch_size'],
            'trajectory_type': 'linear',
            'trajectory_noise_var': comparison_kwargs['trajectory_noise_var'],
            'num_bounce': comparison_kwargs['num_bounce'],
            'object_x_flip': comparison_kwargs['object_x_flip'],
            'object_rotate_xyz': comparison_kwargs['object_rotate_xyz'],
            'image_width': comparison_kwargs['image_width'],
            'image_height': comparison_kwargs['image_height'],
            'image_plane_width': comparison_kwargs['image_plane_width'],
            'image_plane_height': comparison_kwargs['image_plane_height'],
            'grid_width': comparison_kwargs['grid_width'],
            'grid_height': comparison_kwargs['grid_height'],
            'n_ray_width': comparison_kwargs['n_ray_width'],
            'n_ray_height': comparison_kwargs['n_ray_height'],
            'region_radius': comparison_kwargs['region_radius'],
            'obj_raids': comparison_kwargs['obj_raids'],
            'ground_raids': comparison_kwargs['ground_raids'],
        }

        spotlight = sar_render_image(
            mesh_path,
            comparison_kwargs['num_pulse'],
            target_poses,
            comparison_kwargs['azimuth_spread'],
            **{k: v for k, v in render_kwargs.items() if k != 'imaging_algorithm'},
            imaging_algorithm='cbp',
        )
        stripmap = sar_render_image(
            mesh_path,
            comparison_kwargs['num_pulse'],
            target_poses,
            comparison_kwargs['azimuth_spread'],
            **{k: v for k, v in render_kwargs.items() if k != 'imaging_algorithm'},
            imaging_algorithm='stripmap',
        )

        spotlight_vis = _normalize_sar_for_display(spotlight, rgb.shape)
        stripmap_vis = _normalize_sar_for_display(stripmap, rgb.shape)

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].axis('off')

        axes[row_idx, 1].imshow(spotlight_vis, cmap='gray')
        axes[row_idx, 1].axis('off')

        axes[row_idx, 2].imshow(stripmap_vis, cmap='gray')
        axes[row_idx, 2].axis('off')

    fig.subplots_adjust(wspace=0.02, hspace=0.1)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved comparison figure to: {output_path}')
    return output_path


def run_paper_experiments(experiments=PAPER_EXPERIMENTS, baseline=PAPER_BASELINE, plot_db_scale=False):
    for exp in experiments:
        kwargs = {**baseline, **exp.get('overrides', {})}
        multi_param_experiment(
            exp['vary'],
            kwargs,
            exp['name'],
            custom_title_strings=exp.get('custom_title_strings'),
            plot_db_scale=exp.get('plot_db_scale', plot_db_scale),
        )


if __name__ == '__main__':
    # Render sphere.obj at five successively halved scales and stitch them.
    render_sphere_size_comparison()
    # run_paper_experiments(plot_db_scale=False)
    # generate_linear_sar_comparison_figure()
