"""
A1 - point-scatterer validation harness.

Validates the coherent signal-simulation + imaging chain against the closed-form
response of a single ideal point scatterer at a known location. The ray tracer is
bypassed on purpose: one analytic scatterer is injected directly into
`interpolate_signal`, so anything that fails here is a signal/imaging defect and
not a geometry defect. (A2/A3 exercise the tracer.)

Reported quantities, matching the reviewer requests in claudes_plan.md sec. 0:

    aperture phase residual   arg( s_hat(R_p) * conj(A e^{j4 pi R_p / lambda}) )
                              across pulses -- the direct evidence R1.1 asks for
    peak location error       sub-pixel image peak vs the projected target position
    mainlobe width            -3 dB widths vs 1/(B_s cos el) and lambda/(4 sin(dtheta/2))
    PSLR                      vs the -13.26 dB unweighted reference
    ISLR                      integrated sidelobe ratio

Run:
    python validation_point_target.py
"""
import os

# see paper_figures.py -- MKL and torch each bring their own OpenMP runtime
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from matplotlib import pyplot as plt

from imaging_algorithms import projected_CBP
from signal_simulation import apply_snr, generate_trajectory, interpolate_signal
from utils import extract_pose_info, generate_pose_mat, get_next_path, savefig


# Baseline mirrors PAPER_BASELINE in paper_figures.py so the numbers reported here
# describe the configuration the paper's figures were generated with.
BASELINE = dict(
    target_xyz=(0.20, -0.15, 0.0),
    azimuth_deg=0.0,
    elevation_deg=30.0,
    sensor_distance=1.3,
    azimuth_spread=90.0,
    num_pulses=64,
    wavelength=0.5,
    spatial_bw=3650 / 50,
    spatial_fs=3650 / 50,
    region_radius=1.7,
    window_func='sinc',
    amplitude=1.0,
    snr_db=None,
    trajectory_type='circular',
)

# Impulse response of an unweighted (rectangular) spectral aperture.
REFERENCE_PSLR_DB = -13.26


# --------------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------------

def simulate_point_target(
    target_xyz,
    azimuth_deg=0.0,
    elevation_deg=30.0,
    sensor_distance=1.3,
    azimuth_spread=90.0,
    num_pulses=64,
    wavelength=0.5,
    spatial_bw=73.0,
    spatial_fs=73.0,
    region_radius=1.7,
    window_func='sinc',
    amplitude=1.0,
    snr_db=None,
    trajectory_type='circular',
    range_model='spherical',
    device='cpu',
):
    """
    Build the exact phase history of one ideal point scatterer.

    Follows the conventions of the full pipeline so this measures the same code
    path the paper's figures use:
      - `accumulate_scatters` attaches exp(+j 2 pi / lambda * two_way_range)
        to the scatter energy (accumulate_scatters.py:318-328)
      - `render_images` hands `interpolate_signal` the *one-way* range,
        two_way/2, and the sensor's distance to the origin (render_images.py:123-131)

    `range_model` selects the propagation geometry:
      'spherical' -- the true range |traj - q| (correct physics, the default)
      'planar'    -- |traj| + q . forward, i.e. the same far-field approximation
                     projected_CBP assumes internally (imaging_algorithms.py:59-60).
    Imaging a 'planar' history isolates the imager's approximation error: any
    residual peak displacement under 'planar' is an implementation bug, whereas
    displacement present only under 'spherical' is far-field model error.

    Returns a dict of the signal, its range samples, the trajectory, and the
    per-pulse ground-truth range to the target.
    """
    target = torch.tensor(target_xyz, device=device, dtype=torch.float32)  # (3,)

    # one pose at the requested look angle, then the pulse trajectory around it
    pose = generate_pose_mat(
        azimuth_deg, elevation_deg, sensor_distance, device=device
    ).reshape(1, 4, 4)
    true_trajectory, perceived_trajectory, cam_azimuth_deg = generate_trajectory(
        pose,
        trajectory_type=trajectory_type,
        n_pulses=num_pulses,
        azimuth_spread_deg=azimuth_spread,
    )  # (1,P,3), (1,P,3), (1,)
    P = true_trajectory.shape[1]

    # one-way range from each pulse to the target
    sensor_dist = torch.linalg.norm(true_trajectory[0], dim=-1)          # (P,)
    if range_model == 'spherical':
        target_range = torch.linalg.norm(
            true_trajectory[0] - target.reshape(1, 3), dim=-1
        )  # (P,)
    elif range_model == 'planar':
        forward = -true_trajectory[0] / sensor_dist.unsqueeze(-1)        # (P,3)
        target_range = sensor_dist + torch.sum(
            target.reshape(1, 3) * forward, dim=-1
        )  # (P,)
    else:
        raise ValueError("range_model should be 'spherical' or 'planar', but got %s"
                         % range_model)

    # the ideal scatterer: unit energy carrying the two-way propagation phase
    scatter_z = target_range.reshape(P, 1)                       # (P,1) one-way
    scatter_e = amplitude * torch.exp(
        1j * 2 * np.pi / wavelength * (2 * scatter_z)
    )                                                            # (P,1) complex

    signals, sample_z = interpolate_signal(
        scatter_z,
        scatter_e,
        region_radius,
        sensor_dist,                                             # (P,)
        spatial_bw=spatial_bw,
        spatial_fs=spatial_fs,
        window_func=window_func,
        batch_size=None,
    )  # (P,Z), (P,Z)

    if snr_db is not None:
        signals = apply_snr(signals.reshape(1, -1), snr_db).reshape(signals.shape)

    return dict(
        signals=signals.reshape(1, P, -1),
        sample_z=sample_z.reshape(1, P, -1),
        true_trajectory=true_trajectory,
        perceived_trajectory=perceived_trajectory,
        cam_azimuth_deg=cam_azimuth_deg,
        target_range=target_range.reshape(1, P),
        target=target,
        wavelength=wavelength,
        spatial_bw=spatial_bw,
        spatial_fs=spatial_fs,
        amplitude=amplitude,
        elevation_deg=elevation_deg,
        azimuth_spread=azimuth_spread,
    )


def reconstruct_at_range(signals, sample_z, query_z, spatial_fs):
    """
    Band-limited (sinc) reconstruction of the sampled signal at arbitrary ranges.

    This is the same interposummation operator CBP_2D applies internally
    (imaging_algorithms.py:147-151), so the residual it produces is a direct
    measurement of interposummation's reconstruction fidelity.

    inputs:
        signals  (T,P,Z)
        sample_z (T,P,Z)
        query_z  (T,P)
    outputs:
        (T,P) complex
    """
    return torch.sum(
        signals * torch.sinc(spatial_fs * (query_z.unsqueeze(-1) - sample_z)),
        dim=-1,
    )


# --------------------------------------------------------------------------------
# metric: aperture phase residual  (the R1.1 deliverable)
# --------------------------------------------------------------------------------

def aperture_phase_residual(sim):
    """
    Compare the simulated phase history against the analytic exp(-j4 pi R/lambda).

    Reconstructs the sampled signal back at the target's true range and divides out
    the ideal response. A coherent chain leaves a constant residual; drift or
    structure across the aperture is a coherence defect.
    """
    s_hat = reconstruct_at_range(
        sim['signals'], sim['sample_z'], sim['target_range'], sim['spatial_fs']
    )[0]  # (P,)

    R = sim['target_range'][0]                                        # (P,)
    ideal = sim['amplitude'] * torch.exp(1j * 4 * np.pi * R / sim['wavelength'])

    residual = s_hat * torch.conj(ideal) / (torch.abs(ideal) ** 2)
    phase_err_deg = torch.angle(residual) * 180 / np.pi               # (P,)
    mag_ratio = torch.abs(s_hat) / torch.abs(ideal)                   # (P,)

    # remove the constant part: a fixed offset is a harmless global phase, whereas
    # variation across the aperture is what destroys coherent integration
    centered = torch.angle(residual * torch.conj(torch.mean(residual)))
    centered_deg = centered * 180 / np.pi

    return dict(
        s_hat=s_hat,
        ideal=ideal,
        phase_err_deg=phase_err_deg,
        phase_err_centered_deg=centered_deg,
        mag_ratio=mag_ratio,
        rms_phase_err_deg=float(torch.sqrt(torch.mean(centered_deg ** 2))),
        max_phase_err_deg=float(torch.max(torch.abs(centered_deg))),
        mag_ripple_db=float(
            20 * torch.log10(torch.max(mag_ratio) / torch.min(mag_ratio))
        ),
        mean_mag_ratio=float(torch.mean(mag_ratio)),
    )


# --------------------------------------------------------------------------------
# imaging + image-domain metrics
# --------------------------------------------------------------------------------

def image_point_target(
    sim,
    image_width=512,
    image_height=512,
    image_plane_width=1.0,
    image_plane_height=1.0,
    coherent=True,
    cbp_batch_size=4096,
):
    """Form the SAR image, matching render_images.py:152-166."""
    signals = sim['signals'] if coherent else sim['signals'].abs()
    return projected_CBP(
        signals,
        sim['sample_z'],
        sim['perceived_trajectory'],
        sim['spatial_fs'],
        image_plane_rotation_deg=sim['cam_azimuth_deg'] + 90,
        image_width=image_width,
        image_height=image_height,
        image_plane_width=image_plane_width,
        image_plane_height=image_plane_height,
        batch_size=cbp_batch_size,
        coherent_integration=coherent,
        wavelength=sim['wavelength'],
    )[0]  # (H,W)


def expected_peak_pixel(sim, image_width, image_height,
                        image_plane_width, image_plane_height):
    """
    Where the target should land, in (row, col).

    CBP_2D builds the pixel grid then rotates it into the world:
    world_xy = R(theta) @ pixel_xy (imaging_algorithms.py:136-140), so the
    inverse map is pixel_xy = R(theta)^T @ world_xy.
    """
    theta = float(sim['cam_azimuth_deg'][0] + 90) * np.pi / 180.0
    c, s = np.cos(theta), np.sin(theta)
    wx, wy = float(sim['target'][0]), float(sim['target'][1])

    px = c * wx + s * wy       # R^T @ world
    py = -s * wx + c * wy

    # x spans [-w/2, w/2] over columns; y spans [+h/2, -h/2] over rows
    col = (px + image_plane_width / 2) / image_plane_width * (image_width - 1)
    row = (image_plane_height / 2 - py) / image_plane_height * (image_height - 1)
    return row, col


def _subpixel_peak(cut, idx):
    """Quadratic-fit refinement of a discrete peak; returns the offset in samples."""
    if idx <= 0 or idx >= len(cut) - 1:
        return 0.0
    a, b, c = float(cut[idx - 1]), float(cut[idx]), float(cut[idx + 1])
    denom = a - 2 * b + c
    if abs(denom) < 1e-30:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))


def _minus_3db_width(cut, peak_idx, pixel_size):
    """-3 dB (intensity) width of the mainlobe, linearly interpolated, in scene units."""
    peak = float(cut[peak_idx])
    if peak <= 0:
        return float('nan')
    half = peak / np.sqrt(2.0)

    def crossing(direction):
        i = peak_idx
        while 0 < i < len(cut) - 1:
            j = i + direction
            if cut[j] <= half:
                # linear interpolation between samples i and j
                num = float(cut[i]) - half
                den = float(cut[i]) - float(cut[j])
                return abs(i - peak_idx) + (num / den if den != 0 else 0.0)
            i = j
        return float('nan')

    return (crossing(-1) + crossing(+1)) * pixel_size


def _first_nulls(cut, peak_idx):
    """Indices of the first local minima either side of the peak (mainlobe extent)."""
    def null(direction):
        i = peak_idx
        while 0 < i + direction < len(cut) - 1:
            j = i + direction
            if cut[j] > cut[i]:      # started climbing again -> i was the null
                return i
            i = j
        return i
    return null(-1), null(+1)


def _pslr_islr(cut, peak_idx):
    """Peak- and integrated-sidelobe ratio of a 1-D cut, in dB."""
    lo, hi = _first_nulls(cut, peak_idx)
    peak = float(cut[peak_idx])

    sidelobes = np.concatenate([cut[:lo + 1], cut[hi:]])
    pslr = 20 * np.log10(np.max(sidelobes) / peak) if sidelobes.size else float('nan')

    power = cut.astype(np.float64) ** 2
    main_e = power[lo:hi + 1].sum()
    side_e = power.sum() - main_e
    islr = 10 * np.log10(side_e / main_e) if main_e > 0 else float('nan')
    return pslr, islr, (lo, hi)


def impulse_response_metrics(image, sim, image_plane_width, image_plane_height):
    """
    Peak location, resolution, PSLR and ISLR from the image.

    Range maps to the image row axis and cross-range to the column axis: with the
    image plane rotated by (azimuth + 90), the CBP projection coordinate reduces to
    r_coord = -py, so py (rows) is range and px (columns) is cross-range.
    """
    img = image.detach().cpu().numpy()
    H, W = img.shape
    peak_row, peak_col = np.unravel_index(np.argmax(img), img.shape)

    row_cut = img[:, peak_col]   # range
    col_cut = img[peak_row, :]   # cross-range

    row_px = image_plane_height / (H - 1)
    col_px = image_plane_width / (W - 1)

    sub_row = peak_row + _subpixel_peak(row_cut, peak_row)
    sub_col = peak_col + _subpixel_peak(col_cut, peak_col)
    exp_row, exp_col = expected_peak_pixel(
        sim, W, H, image_plane_width, image_plane_height
    )

    rng_res = _minus_3db_width(row_cut, peak_row, row_px)
    crs_res = _minus_3db_width(col_cut, peak_col, col_px)

    rng_pslr, rng_islr, rng_nulls = _pslr_islr(row_cut, peak_row)
    crs_pslr, crs_islr, crs_nulls = _pslr_islr(col_cut, peak_col)

    # closed-form predictions
    el = sim['elevation_deg'] * np.pi / 180
    dtheta = sim['azimuth_spread'] * np.pi / 180
    pred_rng = 1.0 / (sim['spatial_bw'] * np.cos(el))
    pred_crs = sim['wavelength'] / (4 * np.sin(dtheta / 2))

    return dict(
        image=img,
        peak_row=sub_row, peak_col=sub_col,
        expected_row=exp_row, expected_col=exp_col,
        row_err_px=sub_row - exp_row, col_err_px=sub_col - exp_col,
        peak_err_px=float(np.hypot(sub_row - exp_row, sub_col - exp_col)),
        peak_err_scene=float(np.hypot((sub_row - exp_row) * row_px,
                                      (sub_col - exp_col) * col_px)),
        range_res=rng_res, crossrange_res=crs_res,
        predicted_range_res=pred_rng, predicted_crossrange_res=pred_crs,
        range_res_ratio=rng_res / pred_rng,
        crossrange_res_ratio=crs_res / pred_crs,
        range_pslr_db=rng_pslr, crossrange_pslr_db=crs_pslr,
        range_islr_db=rng_islr, crossrange_islr_db=crs_islr,
        row_cut=row_cut, col_cut=col_cut,
        row_nulls=rng_nulls, col_nulls=crs_nulls,
        row_px=row_px, col_px=col_px,
    )


# --------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------

def plot_phase_validation(sim, phase, out_path):
    """Phase history vs analytic, and the residual across the aperture."""
    P = sim['signals'].shape[1]
    pulse = np.arange(P)
    az = np.linspace(-sim['azimuth_spread'] / 2, sim['azimuth_spread'] / 2, P)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(az, np.unwrap(np.angle(phase['ideal'].cpu().numpy())),
               'k-', lw=2, label=r'analytic $4\pi R(\theta)/\lambda$')
    ax[0].plot(az, np.unwrap(np.angle(phase['s_hat'].cpu().numpy())),
               'r--', lw=1.2, label='simulated')
    ax[0].set_xlabel('azimuth offset (deg)')
    ax[0].set_ylabel('unwrapped phase (rad)')
    ax[0].set_title('Phase history')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(az, phase['phase_err_centered_deg'].cpu().numpy(), 'b.-', ms=4)
    ax[1].set_xlabel('azimuth offset (deg)')
    ax[1].set_ylabel('phase residual (deg)')
    ax[1].set_title('Residual (mean removed)\nRMS = %.3e deg'
                    % phase['rms_phase_err_deg'])
    ax[1].grid(alpha=0.3)

    ax[2].plot(az, 20 * np.log10(phase['mag_ratio'].cpu().numpy()), 'g.-', ms=4)
    ax[2].set_xlabel('azimuth offset (deg)')
    ax[2].set_ylabel('amplitude error (dB)')
    ax[2].set_title('Reconstruction amplitude\nripple = %.3e dB'
                    % phase['mag_ripple_db'])
    ax[2].grid(alpha=0.3)

    fig.suptitle('A1: point-scatterer phase validation  '
                 r'($\lambda$=%.3g, $B_s$=%.3g, %d pulses over %g$^\circ$)'
                 % (sim['wavelength'], sim['spatial_bw'], P, sim['azimuth_spread']))
    fig.tight_layout()
    savefig(out_path)
    plt.close(fig)


def plot_impulse_response(metrics, title, out_path):
    """Image with the expected peak marked, plus range/cross-range cuts in dB."""
    img = metrics['image']
    peak = img.max()
    db = 20 * np.log10(np.maximum(img / peak, 1e-8))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    im = ax[0].imshow(db, cmap='inferno', vmin=-60, vmax=0)
    ax[0].plot(metrics['expected_col'], metrics['expected_row'],
               'c+', ms=14, mew=2, label='expected')
    ax[0].plot(metrics['peak_col'], metrics['peak_row'],
               'wx', ms=9, mew=1.6, label='measured')
    ax[0].set_title('Impulse response (dB)\npeak error = %.3f px'
                    % metrics['peak_err_px'])
    ax[0].legend(loc='upper right', fontsize=8)
    fig.colorbar(im, ax=ax[0], label='dB')

    for a, cut, px, res, pred, pslr, islr, name in (
        (ax[1], metrics['row_cut'], metrics['row_px'], metrics['range_res'],
         metrics['predicted_range_res'], metrics['range_pslr_db'],
         metrics['range_islr_db'], 'range'),
        (ax[2], metrics['col_cut'], metrics['col_px'], metrics['crossrange_res'],
         metrics['predicted_crossrange_res'], metrics['crossrange_pslr_db'],
         metrics['crossrange_islr_db'], 'cross-range'),
    ):
        n = len(cut)
        axis = (np.arange(n) - np.argmax(cut)) * px
        a.plot(axis, 20 * np.log10(np.maximum(cut / cut.max(), 1e-8)), lw=1.2)
        a.axhline(-3, color='0.6', ls=':', lw=1)
        a.axhline(REFERENCE_PSLR_DB, color='r', ls='--', lw=1,
                  label='%.2f dB reference' % REFERENCE_PSLR_DB)
        a.set_ylim(-60, 2)
        a.set_xlabel('%s offset (scene units)' % name)
        a.set_ylabel('dB')
        a.set_title('%s cut\n-3 dB: %.4g (predicted %.4g, x%.2f)\nPSLR %.2f dB, '
                    'ISLR %.2f dB' % (name, res, pred, res / pred, pslr, islr))
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    savefig(out_path)
    plt.close(fig)


def coherent_sampling_headroom(wavelength, spatial_bw, spatial_fs):
    """
    Required sampling rate for demodulate-then-interpolate to be reconstructable.

    projected_CBP multiplies the *sampled* signal by exp(-j4 pi z/lambda), shifting its
    spectrum from baseband to a band of width B_s centred at 2/lambda. Sinc
    interpolation at rate F_s can only represent that if the shifted band still fits
    inside [-F_s/2, F_s/2]:

        2/lambda + B_s/2  <  F_s/2      =>      F_s  >  B_s + 4/lambda

    With F_s == B_s (the paper default) the condition can never be met, so the carrier
    is always partly aliased; severity is set by (4/lambda) / B_s.
    """
    required = spatial_bw + 4.0 / wavelength
    return dict(required_fs=required, actual_fs=spatial_fs,
                satisfied=spatial_fs > required,
                headroom_ratio=spatial_fs / required)


def reference_backprojection(
    sim, image_width=512, image_height=512,
    image_plane_width=1.0, image_plane_height=1.0, pixel_batch=8192,
    demod_first=False,
):
    """
    Textbook time-domain backprojection, used as an independent reference.

    Differs from `projected_CBP` in exactly two ways, both deliberate:

      1. exact spherical range |traj - pixel| instead of the plane-wave projection
         about the origin, so there is no far-field error;
      2. the carrier is removed AFTER interpolating to the pixel range, using that
         pixel's own range -- exp(-j4 pi R_pixel / lambda) -- rather than being
         removed from the sampled signal beforehand.

    Order matters for (2). Demodulating the *samples* shifts their spectrum by 2/lambda
    before reconstruction; with spatial_fs == spatial_bw there is no headroom for that
    shift, so the sinc interpolation can no longer represent the carrier. Demodulating
    after interpolation, as strip_map_imaging does (imaging_algorithms.py:244-256),
    avoids the problem.

    No ramp filter: coherent summation over the aperture needs no Radon inversion.

    `demod_first=True` reverts (2) to projected_CBP's ordering while keeping everything
    else identical, which isolates the ordering as a cause.
    """
    signals = sim['signals'][0]                     # (P,Z) complex
    sample_z = sim['sample_z'][0]                   # (P,Z)
    traj = sim['true_trajectory'][0]                # (P,3)
    lam = sim['wavelength']
    fs = sim['spatial_fs']
    device = signals.device
    P = signals.shape[0]
    H, W = image_height, image_width

    if demod_first:
        signals = signals * torch.exp(-1j * 4 * np.pi * sample_z / lam)

    # same pixel grid and rotation convention as CBP_2D
    x = torch.linspace(-image_plane_width / 2, image_plane_width / 2, W, device=device)
    y = torch.linspace(image_plane_height / 2, -image_plane_height / 2, H, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')    # (H,W)
    theta = float(sim['cam_azimuth_deg'][0] + 90) * np.pi / 180.0
    c, s = np.cos(theta), np.sin(theta)
    wx = c * xx - s * yy
    wy = s * xx + c * yy
    pix = torch.stack([wx, wy, torch.zeros_like(wx)], dim=-1).reshape(-1, 3)  # (T,3)

    out = torch.zeros(pix.shape[0], dtype=signals.dtype, device=device)
    for start in range(0, pix.shape[0], pixel_batch):
        block = pix[start:start + pixel_batch]                       # (b,3)
        R = torch.linalg.norm(
            traj.reshape(P, 1, 3) - block.reshape(1, -1, 3), dim=-1
        )                                                            # (P,b) exact range
        interp = torch.sum(
            signals.reshape(P, 1, -1) *
            torch.sinc(fs * (R.unsqueeze(-1) - sample_z.reshape(P, 1, -1))),
            dim=-1,
        )                                                            # (P,b)
        if not demod_first:
            interp = interp * torch.exp(-1j * 4 * np.pi * R / lam)
        out[start:start + pixel_batch] = torch.sum(interp, dim=0)
    return out.reshape(H, W).abs()


# --------------------------------------------------------------------------------
# A1b: far-field validity of the CBP projection
# --------------------------------------------------------------------------------

def run_farfield_sweep(
    # kept inside image_plane_width/2 so the peak never runs off the image edge
    radii=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
    image_width=512, image_height=512,
    image_plane_width=1.0, image_plane_height=1.0,
    device='cpu', make_figures=True, verbose=True, **overrides
):
    """
    Peak displacement vs the target's distance from the scene origin.

    `projected_CBP` maps range to the ground plane with a plane-wave projection about
    the origin (imaging_algorithms.py:59-60), while the simulator propagates the true
    spherical range. The mismatch is the second-order term

        dR  =  |traj - q| - (|traj| + q . forward)  ~  rho^2 / (2 D)

    which displaces a scatterer at radius rho from the origin. Running each radius
    under both range models attributes the displacement: 'planar' should image
    perfectly, 'spherical' should follow rho^2/(2D).
    """
    cfg = dict(BASELINE)
    cfg.update(overrides)
    D = cfg['sensor_distance']
    el = cfg['elevation_deg'] * np.pi / 180
    range_res = 1.0 / (cfg['spatial_bw'] * np.cos(el))

    rows = []
    for rho in radii:
        entry = dict(rho=rho, predicted=rho ** 2 / (2 * D),
                     cells_predicted=rho ** 2 / (2 * D) / range_res)
        for model in ('spherical', 'planar'):
            c = dict(cfg)
            c['target_xyz'] = (rho, 0.0, 0.0)
            c['range_model'] = model
            sim = simulate_point_target(device=device, **c)
            image = image_point_target(
                sim, image_width=image_width, image_height=image_height,
                image_plane_width=image_plane_width,
                image_plane_height=image_plane_height, coherent=True,
            )
            m = impulse_response_metrics(image, sim, image_plane_width,
                                         image_plane_height)
            entry[model] = m['peak_err_scene']
            entry[model + '_cells'] = m['peak_err_scene'] / range_res
        rows.append(entry)

    # largest radius still imaged to better than one resolution cell
    within = [r['rho'] for r in rows if r['spherical_cells'] <= 1.0]
    rho_max = max(within) if within else 0.0

    if verbose:
        print()
        print('-- A1b far-field validity of the CBP plane-wave projection ' + '-' * 15)
        print('  D = %g,  ground range resolution = %.5g' % (D, range_res))
        print()
        print('   rho    spherical    (cells)     planar   (cells)   rho^2/2D   (cells)')
        for r in rows:
            print('  %5.2f   %9.5f  %8.2f  %9.2e  %7.2f  %9.5f  %7.2f'
                  % (r['rho'], r['spherical'], r['spherical_cells'],
                     r['planar'], r['planar_cells'],
                     r['predicted'], r['cells_predicted']))
        print()
        print('  largest rho imaged within one resolution cell: %.3g' % rho_max)
        print('  analytic bound rho < sqrt(2 D / (B_s cos el))  = %.3g'
              % np.sqrt(2 * D * range_res))

    if make_figures:
        os.makedirs('figures/validation', exist_ok=True)
        rho = np.array([r['rho'] for r in rows])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(rho, [r['spherical_cells'] for r in rows], 'o-',
                label='spherical range (true physics)')
        ax.plot(rho, [r['planar_cells'] for r in rows], 's-',
                label='planar range (matches imager)')
        ax.plot(rho, [r['cells_predicted'] for r in rows], 'k--',
                label=r'$\rho^2/(2D)$ prediction')
        ax.axhline(1.0, color='r', ls=':', label='one resolution cell')
        ax.set_xlabel(r'target distance from scene origin $\rho$ (scene units)')
        ax.set_ylabel('peak displacement (resolution cells)')
        ax.set_title('A1b: far-field validity of the CBP plane-wave projection')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        savefig(get_next_path('figures/validation/a1b_farfield.png'))
        plt.close(fig)

    return dict(rows=rows, rho_max=rho_max, range_res=range_res)


# --------------------------------------------------------------------------------
# A1d: why coherent imaging loses cross-range resolution
# --------------------------------------------------------------------------------

def run_demod_order_study(
    wavelengths=(0.02, 0.05, 0.10, 0.25, 0.50),
    oversample_ratios=(1, 2, 4, 8),
    probe_wavelength=0.02,
    image_width=1024, image_height=1024,
    image_plane_width=0.5, image_plane_height=0.5,
    device='cpu', make_figures=True, verbose=True, **overrides
):
    """
    Establishes why complex-valued imaging under-performs (reviewers R2.3, R3).

    Part 1 sweeps wavelength through three imagers: projected_CBP, the reference
    backprojector, and the reference backprojector forced into projected_CBP's
    demodulation order. A coherent imager's cross-range resolution should follow
    lambda/(4 sin(dtheta/2)) until the range envelope takes over.

    Part 2 gives projected_CBP progressively more sampling headroom at a short
    wavelength and checks recovery against the F_s > B_s + 4/lambda condition.
    """
    cfg = dict(BASELINE)
    cfg.update(overrides)
    cfg['target_xyz'] = (0.0, 0.0, 0.0)          # origin: no far-field error in play
    dtheta = cfg['azimuth_spread'] * np.pi / 180

    def measure(sim, kind):
        if kind == 'cbp':
            img = image_point_target(
                sim, image_width=image_width, image_height=image_height,
                image_plane_width=image_plane_width,
                image_plane_height=image_plane_height, coherent=True)
        else:
            img = reference_backprojection(
                sim, image_width=image_width, image_height=image_height,
                image_plane_width=image_plane_width,
                image_plane_height=image_plane_height,
                demod_first=(kind == 'bp_demod_first'))
        return impulse_response_metrics(
            img, sim, image_plane_width, image_plane_height)['crossrange_res']

    lam_rows = []
    for lam in wavelengths:
        c = dict(cfg, wavelength=lam)
        sim = simulate_point_target(device=device,
                                    **{k: v for k, v in c.items()})
        lam_rows.append(dict(
            wavelength=lam,
            carrier_limit=lam / (4 * np.sin(dtheta / 2)),
            cbp=measure(sim, 'cbp'),
            reference=measure(sim, 'bp'),
            reference_demod_first=measure(sim, 'bp_demod_first'),
        ))

    os_rows = []
    for osr in oversample_ratios:
        c = dict(cfg, wavelength=probe_wavelength,
                 spatial_fs=cfg['spatial_bw'] * osr)
        sim = simulate_point_target(device=device, **c)
        head = coherent_sampling_headroom(
            probe_wavelength, cfg['spatial_bw'], cfg['spatial_bw'] * osr)
        os_rows.append(dict(ratio=osr, crossrange=measure(sim, 'cbp'), **head))

    carrier = probe_wavelength / (4 * np.sin(dtheta / 2))

    if verbose:
        base = coherent_sampling_headroom(
            cfg['wavelength'], cfg['spatial_bw'], cfg['spatial_fs'])
        print()
        print('-- A1d coherent cross-range: demodulation order ' + '-' * 26)
        print('  baseline F_s = %.4g, required B_s + 4/lambda = %.4g  -> %s'
              % (base['actual_fs'], base['required_fs'],
                 'OK' if base['satisfied'] else 'VIOLATED (carrier aliased)'))
        print()
        print('  lambda   projected_CBP   reference BP   BP w/ CBP order   carrier limit')
        for r in lam_rows:
            print('  %6.3f   %13.5f  %13.5f  %16.5f  %13.5f'
                  % (r['wavelength'], r['cbp'], r['reference'],
                     r['reference_demod_first'], r['carrier_limit']))
        print()
        print('  projected_CBP with sampling headroom at lambda = %g '
              '(carrier limit %.5f):' % (probe_wavelength, carrier))
        print('    F_s/B_s   F_s      required   cross-range   satisfied')
        for r in os_rows:
            print('    %5d   %7.1f  %9.1f   %11.5f   %s'
                  % (r['ratio'], r['actual_fs'], r['required_fs'],
                     r['crossrange'], 'yes' if r['satisfied'] else 'no'))

    if make_figures:
        os.makedirs('figures/validation', exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        lam = np.array([r['wavelength'] for r in lam_rows])
        ax[0].loglog(lam, [r['carrier_limit'] for r in lam_rows], 'k--',
                     label=r'$\lambda/(4\sin(\Delta\theta/2))$')
        ax[0].loglog(lam, [r['reference'] for r in lam_rows], 'o-',
                     label='reference BP (demod after interp)')
        ax[0].loglog(lam, [r['reference_demod_first'] for r in lam_rows], '^-',
                     label='reference BP (demod before interp)')
        ax[0].loglog(lam, [r['cbp'] for r in lam_rows], 's-',
                     label='projected_CBP')
        ax[0].set_xlabel(r'wavelength $\lambda$ (scene units)')
        ax[0].set_ylabel('cross-range -3 dB width')
        ax[0].set_title('Cross-range resolution vs wavelength')
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3, which='both')

        ratio = np.array([r['ratio'] for r in os_rows])
        ax[1].semilogy(ratio, [r['crossrange'] for r in os_rows], 'o-',
                       label='projected_CBP')
        ax[1].axhline(carrier, color='k', ls='--',
                      label=r'$\lambda/(4\sin(\Delta\theta/2))$')
        need = (cfg['spatial_bw'] + 4 / probe_wavelength) / cfg['spatial_bw']
        ax[1].axvline(need, color='r', ls=':',
                      label=r'$F_s = B_s + 4/\lambda$')
        ax[1].set_xlabel(r'oversampling ratio $F_s/B_s$')
        ax[1].set_ylabel('cross-range -3 dB width')
        ax[1].set_title(r'Recovery with sampling headroom ($\lambda$=%g)'
                        % probe_wavelength)
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, which='both')

        fig.suptitle('A1d: coherent cross-range resolution is lost to '
                     'demodulate-before-interpolate')
        fig.tight_layout()
        savefig(get_next_path('figures/validation/a1d_demod_order.png'))
        plt.close(fig)

    return dict(wavelength_rows=lam_rows, oversample_rows=os_rows)


# --------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------

def run_point_validation(
    image_width=512,
    image_height=512,
    image_plane_width=1.0,
    image_plane_height=1.0,
    device='cpu',
    make_figures=True,
    verbose=True,
    **overrides
):
    """Run A1 end to end and return every measured quantity."""
    cfg = dict(BASELINE)
    cfg.update(overrides)

    if make_figures:
        os.makedirs('figures/validation', exist_ok=True)

    sim = simulate_point_target(device=device, **cfg)
    phase = aperture_phase_residual(sim)

    results = dict(config=cfg, phase=phase)

    for coherent in (True, False):
        image = image_point_target(
            sim,
            image_width=image_width, image_height=image_height,
            image_plane_width=image_plane_width,
            image_plane_height=image_plane_height,
            coherent=coherent,
        )
        m = impulse_response_metrics(image, sim, image_plane_width, image_plane_height)
        results['coherent' if coherent else 'magnitude'] = m

        if make_figures:
            mode = 'coherent' if coherent else 'magnitude-only'
            plot_impulse_response(
                m, 'A1 point target, %s imaging' % mode,
                get_next_path('figures/validation/a1_ipr_%s.png'
                              % ('coherent' if coherent else 'magnitude')),
            )

    if make_figures:
        plot_phase_validation(
            sim, phase, get_next_path('figures/validation/a1_phase.png')
        )

    if verbose:
        _report(results)
    return results


def _report(results):
    cfg = results['config']
    p = results['phase']
    el = cfg['elevation_deg'] * np.pi / 180

    print()
    print('=' * 74)
    print('A1  point-scatterer validation')
    print('=' * 74)
    print('target %s   %d pulses over %g deg   el %g deg   dist %g'
          % (cfg['target_xyz'], cfg['num_pulses'], cfg['azimuth_spread'],
             cfg['elevation_deg'], cfg['sensor_distance']))
    print('lambda %.4g   B_s %.4g   F_s %.4g   window %s'
          % (cfg['wavelength'], cfg['spatial_bw'], cfg['spatial_fs'],
             cfg['window_func']))
    print('slant range resolution 1/B_s          = %.5g' % (1 / cfg['spatial_bw']))
    print('ground range resolution 1/(B_s cos el) = %.5g'
          % (1 / (cfg['spatial_bw'] * np.cos(el))))
    print('lambda / (2 * ground range resolution) = %.4g   '
          '(>> 1 means the carrier is coarser than the bandwidth)'
          % (cfg['wavelength'] / (2 / (cfg['spatial_bw'] * np.cos(el)))))

    head = coherent_sampling_headroom(cfg['wavelength'], cfg['spatial_bw'],
                                      cfg['spatial_fs'])
    print('coherent sampling condition F_s > B_s + 4/lambda: %.4g > %.4g  -> %s'
          % (head['actual_fs'], head['required_fs'],
             'satisfied' if head['satisfied'] else
             'VIOLATED, carrier partly aliased (see A1d)'))

    print()
    print('-- aperture phase residual vs analytic exp(-j4 pi R/lambda) ' + '-' * 15)
    print('  RMS phase error        %12.4e deg' % p['rms_phase_err_deg'])
    print('  max phase error        %12.4e deg' % p['max_phase_err_deg'])
    print('  amplitude ripple       %12.4e dB' % p['mag_ripple_db'])
    print('  mean amplitude ratio   %12.6f     (1.0 = lossless reconstruction)'
          % p['mean_mag_ratio'])

    for key, label in (('coherent', 'COHERENT'), ('magnitude', 'MAGNITUDE-ONLY')):
        m = results[key]
        print()
        print('-- %s imaging %s' % (label, '-' * (58 - len(label))))
        print('  peak location error    %12.4f px  (%.4g scene units)'
              % (m['peak_err_px'], m['peak_err_scene']))
        print('    row (range)          %12.4f px' % m['row_err_px'])
        print('    col (cross-range)    %12.4f px' % m['col_err_px'])
        print('  range -3 dB width      %12.5g     predicted %.5g   (x%.3f)'
              % (m['range_res'], m['predicted_range_res'], m['range_res_ratio']))
        # labelled "carrier limit" rather than "predicted": A1d shows projected_CBP
        # never reaches it, so the ratio is a coherence diagnostic, not an error
        print('  cross-range -3 dB      %12.5g     carrier limit %.5g   (x%.3f)'
              % (m['crossrange_res'], m['predicted_crossrange_res'],
                 m['crossrange_res_ratio']))
        print('  range PSLR             %12.2f dB  reference %.2f dB'
              % (m['range_pslr_db'], REFERENCE_PSLR_DB))
        print('  cross-range PSLR       %12.2f dB  reference %.2f dB'
              % (m['crossrange_pslr_db'], REFERENCE_PSLR_DB))
        print('  range ISLR             %12.2f dB' % m['range_islr_db'])
        print('  cross-range ISLR       %12.2f dB' % m['crossrange_islr_db'])
    print('=' * 74)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_point_validation(device=device)
    run_farfield_sweep(device=device)
    run_demod_order_study(device=device)


if __name__ == '__main__':
    main()
