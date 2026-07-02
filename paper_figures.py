"""Paper figure experiments. `main` runs the full suite via one call."""
import numpy as np

from render_images import multi_param_experiment


PAPER_BASELINE = dict(
    azimuth_spread=90,
    debug_gif=False,
    num_pulse=64,
    spatial_bw=3650 / 50, # the denominator is in mm
    spatial_fs=3650 / 50, # the denominator is in mm
    wavelength=0.5,
    use_sig_magnitude=True,
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
    )

    # SNR sweep — sensitivity of the reconstruction to additive receiver noise.
    snr_db_vals = np.linspace(0, 22, 8).tolist()
    snrdb = dict(
        name='snrdb',
        vary={'snr_db': snr_db_vals},
        custom_title_strings=['SNR dB: %.1f' % s for s in snr_db_vals],
    )

    # Wavelength sweep with magnitude-only CBP — how carrier wavelength shapes the image.
    wavelength_vals = [0.01, 0.02, 0.05, 0.1]
    wavelengthmagnitude = dict(
        name='wavelengthmagnitude',
        vary={'wavelength': wavelength_vals},
        custom_title_strings=['wavelength: %.2f' % w for w in wavelength_vals],
    )

    # Wavelength sweep with coherent (complex) CBP — same sweep, phase preserved.
    wavelengthcomplex = dict(
        name='wavelengthcomplex',
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
        custom_title_strings=['%.2e' % v for v in noise_vals],
    )

    return [
        az_spread,
        num_pulse,
        fsbw,
        snrdb,
        wavelengthmagnitude,
        wavelengthcomplex,
        trajectory_type,
        trajectory_noise_var,
    ]


PAPER_EXPERIMENTS = _paper_experiments()


def run_paper_experiments(experiments=PAPER_EXPERIMENTS, baseline=PAPER_BASELINE):
    for exp in experiments:
        kwargs = {**baseline, **exp.get('overrides', {})}
        multi_param_experiment(
            exp['vary'],
            kwargs,
            exp['name'],
            custom_title_strings=exp.get('custom_title_strings'),
        )


if __name__ == '__main__':
    run_paper_experiments()
