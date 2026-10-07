"""SAR paper figure experiments. `main` runs the full suite via one call."""
import os
import zlib

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the
# second to initialize aborts with "OMP: Error #15". Allow the duplicate.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import PIL
import torch
from matplotlib import pyplot as plt

from config import CONFIG, SHAPENET_CARS_DIR, srn_split_dir
from render_images import multi_param_sar_experiment, sar_render_image
from utils import extract_pose_info, generate_pose_mat


# config.json's sar_baseline. Notes on its keys:
#   obj_id/azimuth_deg/elevation_deg pin the object and the look: render_random_image renders the
#     object's pose nearest that azimuth and elevation, e.g. one read off a generate_dataset.py
#     test-run gif, and draws both at random when None. The object may be from any split
#   spatial_bw/spatial_fs, region_radius, num_bounce and asinh_k_ratio match SSS_PAPER_BASELINE
#   wavelength can't be None, unlike the side scan baseline's: strip_map_imaging always demodulates
#     by wavelength, and az_spread_linear_stripmap below needs it
#   waveform 'gaussian', since the default sinc rings; its side lobes streak off the car
#   image_plane_width/height 1.1 frames an srn car the way the side scan images do
#   compression/db_floor/asinh_k_ratio -- the one place these are decided;
#     multi_param_sar_experiment reads these off the baseline (popping them before the rest is
#     forwarded to render_random_image) unless an experiment's overrides set one instead
SAR_PAPER_BASELINE = dict(CONFIG['sar_baseline'])


def _sar_experiments():

    # Synthetic aperture arc length sweep — how azimuth coverage shapes the image.
    az_vals = np.linspace(0, 360, 5).tolist()
    az_spread = dict(
        name='az_spread',
        vary={'azimuth_spread': az_vals},
        custom_title_strings=['Azimuth spread: %.0f deg' % a for a in az_vals],
    )

    # Pulse count sweep — how along-track sampling density affects the image.
    pulse_vals = [2 ** k for k in range(3, 11)]  # 8, 16, ..., 1024
    num_pulse = dict(
        name='num_pulse',
        vary={'num_pulse': pulse_vals},
        custom_title_strings=['Pulses: %d' % p for p in pulse_vals],
    )

    # Ray count sweep — how densely the ray grid samples the scene, from half to five times the
    # baseline's rays per side, linearly spaced. Width and height move together, as in the range
    # angle suite's n_ray sweep.
    base_n_ray = SAR_PAPER_BASELINE['n_ray_width']
    n_ray_vals = np.linspace(base_n_ray / 2, 5 * base_n_ray, 5).round().astype(int).tolist()
    n_ray = dict(
        name='n_ray',
        vary={'n_ray_width': n_ray_vals, 'n_ray_height': n_ray_vals},
        custom_title_strings=['Rays: %d x %d' % (r, r) for r in n_ray_vals],
    )

    # Spatial bandwidth sweep — range resolution goes as 1/BW. Fs = 2*BW throughout, as in the
    # baseline, so the panels differ by bandwidth alone and not by how finely each pulse is sampled.
    bwfs_vals = [4, 16, 64, 128, 512]
    fsbw = dict(
        name='fsbw',
        vary={'spatial_bw': bwfs_vals, 'spatial_fs': [2 * bw for bw in bwfs_vals]},
        custom_title_strings=['BW: %d, Fs: %d' % (bw, 2 * bw) for bw in bwfs_vals],
    )

    # The same bandwidth sweep imaged with strip-map instead of the baseline's CBP. strip-map
    # imaging needs a linear track, so the trajectory is pinned to linear here too.
    fsbw_stripmap = dict(
        name='fsbw_stripmap',
        vary={'spatial_bw': bwfs_vals, 'spatial_fs': [2 * bw for bw in bwfs_vals]},
        overrides={'trajectory_type': 'linear', 'imaging_algorithm': 'stripmap'},
        custom_title_strings=['Strip-map, BW: %d, Fs: %d' % (bw, 2 * bw) for bw in bwfs_vals],
    )

    # SNR sweep — sensitivity of the reconstruction to additive receiver noise.
    snr_db_vals = np.linspace(0, 22, 5).tolist()
    snrdb = dict(
        name='snrdb',
        vary={'snr_db': snr_db_vals},
        custom_title_strings=['SNR: %.1f dB' % s for s in snr_db_vals],
    )

    # Wavelength sweep with magnitude-only CBP — how carrier wavelength shapes the image.
    wavelength_vals = [0.01, 0.05, 0.2, 0.5, 2]
    wavelength = dict(
        name='wavelength',
        vary={'wavelength': wavelength_vals},
        custom_title_strings=['Wavelength: %.2f' % w for w in wavelength_vals],
    )

    # Trajectory geometry comparison — linear (stripmap-like) vs circular (spotlight). Only two
    # trajectory types exist, so the panels pair them at matched angular spreads and close with
    # the full circle, which is the aperture a linear track cannot fly: generate_trajectory
    # asserts a linear spread below 180 deg, since the track runs off to infinity at 180.
    trajectory_types = ['linear', 'circular', 'linear', 'circular', 'circular']
    trajectory_spreads = [45, 45, 135, 135, 360]
    trajectory_type = dict(
        name='trajectory_type',
        vary={'trajectory_type': trajectory_types, 'azimuth_spread': trajectory_spreads},
        custom_title_strings=['%s, %d deg' % (t.capitalize(), a)
                              for t, a in zip(trajectory_types, trajectory_spreads)],
    )

    # Azimuth spread sweep for strip-map imaging on a linear trajectory. Capped at 135 deg
    # (not 180) since generate_trajectory asserts a linear spread strictly below 180 deg, where
    # the track runs off to infinity.
    az_spread_linear_vals = np.linspace(0, 135, 5).tolist()
    az_spread_linear_stripmap = dict(
        name='az_spread_linear_stripmap',
        vary={'azimuth_spread': az_spread_linear_vals},
        overrides={'trajectory_type': 'linear', 'imaging_algorithm': 'stripmap'},
        custom_title_strings=['Linear strip-map, azimuth spread: %.1f deg' % a
                              for a in az_spread_linear_vals],
    )

    # Trajectory noise sweep — how sensor position error along the path degrades the image.
    noise_vals = [0] + (10 ** np.linspace(-4, -2, 4, endpoint=True)).tolist()
    trajectory_noise_var = dict(
        name='trajectory_noise_var',
        vary={'trajectory_noise_var': noise_vals},
        custom_title_strings=['Turbulence: %.2e' % v for v in noise_vals],
    )

    # Transmit-waveform comparison — how the pulse / range-compression window shapes
    # the image. waveform selects the effective range window used inside
    # interpolate_signal: an ideal sinc, a Gaussian pulse, and the matched-filter
    # responses of an LFM chirp and a Barker-13 phase code. Those are four of the five waveforms
    # interpolate_signal implements — the sonar suite's waveform figure covers the fifth, a Hamming
    # window — so the last panel here is the chirp again at twice the bandwidth, the knob that
    # actually sets range resolution once a waveform is chosen. Fs stays at 2*BW as in the
    # baseline, so it follows the wider pulse instead of aliasing it. Twice and not more: past that
    # the range resolution outruns what 64 pulses of aperture resolve in cross range, and the panel
    # turns into grating lobes rather than a sharper car.
    base_bw = SAR_PAPER_BASELINE['spatial_bw']
    waveform_vals = ['sinc', 'gaussian', 'lfm', 'barker13', 'lfm']
    waveform_bw_vals = [base_bw] * 4 + [2 * base_bw]
    waveform = dict(
        name='waveform',
        vary={'waveform': waveform_vals,
              'spatial_bw': waveform_bw_vals,
              'spatial_fs': [2 * bw for bw in waveform_bw_vals]},
        custom_title_strings=['Sinc Interpolation', 'Gaussian Pulse', 'LFM Chirp', 'Barker 13',
                              'LFM Chirp, 2x BW'],
    )

    # sphere
    scale_vals = [1/8, 1/16, 1/32, 1/64, 1/128]
    override_obj_path = os.path.join('/workspace','berian','sphere.obj')
    sphere = dict(
        name='sphere_size',
        vary={'mesh_scale': scale_vals},
        overrides = {'override_obj_path': override_obj_path,
                     # 'debug_gif': True,
                     'make_ground': False,
                    },
        custom_title_strings=['Scale: 1','Scale: 1/2','Scale: 1/4','Scale: 1/8','Scale: 1/16'],
    )

    return [
        # az_spread,
        # num_pulse,
        n_ray,
        # fsbw,
        # fsbw_stripmap,
        # snrdb,
        # wavelength,
        # trajectory_type,
        # az_spread_linear_stripmap,
        # trajectory_noise_var,
        # waveform,
        # sphere,
    ]


SAR_PAPER_EXPERIMENTS = _sar_experiments()


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



def generate_linear_sar_comparison_figure(
    num_examples=4,
    output_path='figures/linear_sar_comparison.png',
    baseline=SAR_PAPER_BASELINE,
    seed=8134,
    min_elevation_deg=20,
):
    """Create a 4-row figure with RGB, spotlight, and strip-map SAR panels."""
    dataset_dir = srn_split_dir('cars_train')
    models_dir = SHAPENET_CARS_DIR

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
            'waveform': comparison_kwargs['waveform'],
            'snr_db': comparison_kwargs['snr_db'],
            'wavelength': comparison_kwargs['wavelength'],
            'use_sig_magnitude': comparison_kwargs['use_sig_magnitude'],
            'imaging_algorithm': 'cbp',
            'cbp_batch_size': comparison_kwargs['cbp_batch_size'],
            'signal_interpolation': comparison_kwargs['signal_interpolation'],
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


def generate_modality_comparison_figure(
    num_examples=4,
    output_path='figures/modality_comparison.png',
    split='cars_train',
    seed=8134,
):
    """Create a 4-row figure with RGB, CBP SAR, stripmap SAR, side scan and forward looking sonar panels, rendered as the dataset is."""
    # lazy: generate_dataset imports this module
    import generate_dataset as gd

    modalities = ('cbp_sar', 'side_scan_sonar', 'forward_looking_sonar')
    columns = ('cbp_sar', 'stripmap_sar', 'side_scan_sonar', 'forward_looking_sonar')
    dataset_dir = srn_split_dir(split)
    obj_ids = sorted(os.listdir(dataset_dir))
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(num_examples, 1 + len(columns), figsize=(15, 3.2 * num_examples),
                             squeeze=False)

    for row_idx in range(num_examples):
        obj_id = obj_ids[rng.randint(0, len(obj_ids))]
        object_dir = os.path.join(dataset_dir, obj_id)
        mesh_path = os.path.join(SHAPENET_CARS_DIR, obj_id, 'models', 'model_normalized.obj')

        # one random pose from the dataset's elevation band
        pose_nums, poses, _, _ = gd.plan_object(obj_id, split, True, None, True,
                                                (gd.MIN_ELEVATION_DEG, gd.MAX_ELEVATION_DEG), modalities)
        pose_num = pose_nums[rng.randint(0, len(pose_nums))]
        gd.render_poses(obj_id, object_dir, mesh_path, {pose_num: modalities}, 1, poses,
                        'cuda', False, True, modalities, False)

        # stripmap is not a dataset modality: image the cbp panel's aperture and noise seed with it
        seed_pose = zlib.crc32(('%s/%s' % (obj_id, pose_num)).encode())
        np.random.seed(seed_pose)
        torch.manual_seed(seed_pose)
        stripmap = gd._quiet(sar_render_image, mesh_path,
                             gd.SAR_PAPER_BASELINE['num_pulse'],
                             torch.tensor(poses[pose_num][0], device='cuda'),
                             gd.AZIMUTH_SPREAD_DEG,
                             imaging_algorithm = 'stripmap',
                             trajectory_type   = gd.TRAJECTORY_TYPE,
                             **{k: gd.SAR_PAPER_BASELINE[k] for k in gd.SAR_KEYS})  # (1,H,W)
        gd.save_gray_png(stripmap[0].detach().cpu().numpy(),
                         gd.output_path(object_dir, obj_id, pose_num, 'stripmap_sar', True),
                         **gd.DISPLAY['cbp_sar'])

        rgb = np.array(PIL.Image.open(os.path.join(object_dir, 'rgb', f'{pose_num}.png')))[..., :3]
        panels = [rgb] + [np.array(PIL.Image.open(gd.output_path(object_dir, obj_id, pose_num, m, True)))
                          for m in columns]
        for col_idx, panel in enumerate(panels):
            axes[row_idx, col_idx].imshow(panel, cmap='gray', vmin=0, vmax=255)
            axes[row_idx, col_idx].axis('off')
        print(f'row {row_idx}: {obj_id} pose {pose_num}, elevation {poses[pose_num][2]:.1f} deg')

    fig.subplots_adjust(wspace=0.02, hspace=0.1)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved comparison figure to: {output_path}')
    return output_path


def generate_overview_figure(
    obj_id='e6846796e15020e02bc9f17412005422',
    pose_num='000046',
    baseline=SAR_PAPER_BASELINE,
    output_dir='figures/overview',
    seed=0,
):
    """Figure 1 (b)-(f) of the paper: the SAR image of one srn_cars pose, then the first-bounce
    range and energy maps, the return scatter and the clean signal of the pulse at that pose.
    The default object and pose are panel (a)'s rgb image."""
    # lazy: generate_dataset imports this module
    import generate_dataset as gd
    from accumulate_scatters import accumulate_scatters
    from render_images import _find_split_dir
    from signal_simulation import interpolate_signal, load_mesh

    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(seed)
    np.random.seed(seed)

    pose_path = os.path.join(_find_split_dir(obj_id), obj_id, 'pose', f'{pose_num}.txt')
    pose = torch.tensor(np.loadtxt(pose_path).reshape(1, 4, 4).astype(np.float32), device=device)
    mesh_path = os.path.join(SHAPENET_CARS_DIR, obj_id, 'models', 'model_normalized.obj')
    scene = load_mesh(mesh_path, device=device, make_ground=True, obj_raids=baseline['obj_raids'],
                      ground_raids=baseline['ground_raids'], x_flip=baseline['object_x_flip'],
                      rotate_xyz=baseline['object_rotate_xyz'])

    # (b), saved as the dataset saves it
    sar = sar_render_image(mesh_path, baseline['num_pulse'], pose, baseline['azimuth_spread'],
                           imaging_algorithm='cbp', trajectory_type=baseline['trajectory_type'],
                           preloaded_mesh=scene, **{k: baseline[k] for k in gd.SAR_KEYS})  # (1,H,W)
    gd.save_gray_png(sar[0].detach().cpu().numpy(), os.path.join(output_dir, 'sar_overview.png'),
                     **gd.DISPLAY['cbp_sar'])

    # (c)-(f): one pulse from the rgb camera's position, at the center of the aperture
    sensor = pose[:, :3, 3].reshape(1, 1, 3)
    n_ray = baseline['n_ray_width'] * baseline['n_ray_height']
    ranges, energies, maps = accumulate_scatters(
        *scene, sensor, wavelength=baseline['wavelength'], debug_gif=True,
        grid_width=baseline['grid_width'], grid_height=baseline['grid_height'],
        n_ray_width=baseline['n_ray_width'], n_ray_height=baseline['n_ray_height'],
        num_bounce=baseline['num_bounce'], second_bounce_batch_size=2**9)
    signal, sample_z = interpolate_signal(
        ranges[0][0].unsqueeze(0) / 2, energies[0][0].unsqueeze(0), baseline['region_radius'],
        torch.linalg.norm(sensor[0, 0]).reshape(1), spatial_bw=baseline['spatial_bw'],
        spatial_fs=baseline['spatial_fs'], waveform=baseline['waveform'], batch_size=None)

    depth = maps[(0, 0)]['depth'].cpu().numpy()                       # (H,W), misses are -1
    depth = np.where(depth < 0, np.nan, depth)
    energy = maps[(0, 0)]['energy'].cpu().numpy() / n_ray             # (H,W), as E_k
    scatter_range = ranges[0][0].cpu().numpy() / 2                    # (R',) half round trip
    scatter_energy = energies[0][0].abs().cpu().numpy()               # (R',)
    signal = signal[0].abs().cpu().numpy()                            # (Z,)
    sample_z = sample_z[0].cpu().numpy()                              # (Z,)
    print('depth %.3f..%.3f, %d misses; energy max %.3e, 99.9th pct %.3e; %d returns'
          % (np.nanmin(depth), np.nanmax(depth), np.isnan(depth).sum(), energy.max(),
             np.percentile(energy, 99.9), scatter_range.size))

    # the ray grid's own coordinates, with +y up, so the maps read like the rgb image
    gw, gh = baseline['grid_width'], baseline['grid_height']
    extent = (-gw / 2, gw / 2, -gh / 2, gh / 2)
    figsize = (3.0, 2.6)
    plt.rcParams.update({'font.size': 13})  # readable at 0.3 of the paper's text width

    def save(fig, name):
        fig.savefig(os.path.join(output_dir, name), dpi=200, bbox_inches='tight')
        plt.close(fig)

    def scaled(values, label):  # values and label in units of the peak's power of ten
        power = int(np.floor(np.log10(np.nanmax(values))))
        return values / 10.0**power, r'%s ($\times 10^{%d}$)' % (label, power)

    energy, energy_label = scaled(energy, r'first-bounce $E_k$')
    for name, image, label, vmax in (
            ('sar_overview_depth.png', depth, r'first-hit range ($\ell$)', None),
            ('sar_overview_energy.png', energy, energy_label, np.percentile(energy, 99.9))):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(image, cmap='gray', extent=extent, vmin=0 if vmax else None, vmax=vmax)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_xlabel(r'$x$ ($\ell$)')
        ax.set_ylabel(r'$y$ ($\ell$)')
        fig.colorbar(im, ax=ax, label=label, extend='max' if vmax else 'neither')
        save(fig, name)

    # the far multipath returns fall outside the signal window, so the axis stops where it does
    scatter_energy, scatter_label = scaled(scatter_energy, r'$|E_k|$')
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(scatter_range, scatter_energy, s=1, linewidths=0, rasterized=True)
    ax.set_xlim(sample_z[0], sample_z[-1])
    ax.set_xlabel(r'range ($\ell$)')
    ax.set_ylabel(scatter_label)
    save(fig, 'sar_overview_scatters.png')

    signal, signal_label = scaled(signal, r'$|s(z)|$')
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sample_z, signal)
    ax.set_xlim(sample_z[0], sample_z[-1])
    ax.set_xlabel(r'range ($\ell$)')
    ax.set_ylabel(signal_label)
    save(fig, 'sar_overview_signal.png')
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
    print(f'Saved overview panels to: {output_dir}')
    return output_dir


def run_sar_paper_experiments(experiments=SAR_PAPER_EXPERIMENTS, baseline=SAR_PAPER_BASELINE):
    for exp in experiments:
        kwargs = {**baseline, **exp.get('overrides', {})}
        multi_param_sar_experiment(
            exp['vary'],
            kwargs,
            exp['name'],
            custom_title_strings=exp.get('custom_title_strings'),
        )


if __name__ == '__main__':
    run_sar_paper_experiments()
    # generate_linear_sar_comparison_figure()
