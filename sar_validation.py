"""
Validation of the SAR forward model against closed-form references, for the paper's Validation
subsection. Reuses the harnesses in validation_point_target.py (A1) and validation_plate.py (A2).

    point target  an ideal scatterer injected into interposummation at the paper's SAR defaults,
                  bypassing the ray tracer: phase history vs exp(j 4 pi R / lambda), image peak
                  position over a grid of targets, and the impulse response vs the same filtered
                  backprojection evaluated without sampling
    flat plate    a ray-traced square plate vs the physical-optics RCS: area law, glint pattern,
                  absolute level vs wavelength, and the ray spacing that samples the phase across it

Prints every number the paper quotes and writes figures/sar_validation_NN.png.

Run:
    python sar_validation.py
"""
import os

# see sar_paper_figures.py -- MKL and torch each bring their own OpenMP runtime
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from matplotlib import pyplot as plt

from config import CONFIG
from imaging_algorithms import projected_CBP
from signal_simulation import make_transmit_waveform
from utils import get_next_path
import validation_plate as plate
from validation_point_target import (
    aperture_phase_residual,
    expected_peak_pixel,
    impulse_response_metrics,
    simulate_point_target,
)


SAR = CONFIG['sar_baseline']

# point-target settings taken from the paper's SAR defaults; srncars poses sit at distance 1.3
POINT = dict(
    azimuth_deg=SAR['azimuth_deg'],
    elevation_deg=SAR['elevation_deg'],
    sensor_distance=1.3,
    azimuth_spread=SAR['azimuth_spread'],
    num_pulses=SAR['num_pulse'],
    wavelength=SAR['wavelength'],
    spatial_bw=SAR['spatial_bw'],
    spatial_fs=SAR['spatial_fs'],
    region_radius=SAR['region_radius'],
    waveform=SAR['waveform'],
    trajectory_type=SAR['trajectory_type'],
    # the ray tracer's ranges are planar-wavefront (accumulate_scatters), so the injected one is too
    range_model='planar',
)

WAVEFORMS = ('sinc', 'gaussian', 'lfm', 'barker13')
GRID_SPAN = 0.8          # 5 x 5 targets over +/- 0.4, inside the 1.2 image plane
IRF_PLANE = 0.3          # zoomed image for impulse-response cuts
IRF_PIXELS = 601


# --------------------------------------------------------------------------------
# point target
# --------------------------------------------------------------------------------

def image_cbp(signals, sim, plane, pixels, interpolation):
    """Magnitude-only CBP, as render_images.py forms the paper's SAR images."""
    return projected_CBP(
        signals.abs(), sim['sample_z'], sim['perceived_trajectory'], sim['spatial_fs'],
        image_plane_rotation_deg=sim['cam_azimuth_deg'] + 90,
        image_width=pixels, image_height=pixels,
        image_plane_width=plane, image_plane_height=plane,
        batch_size=4096, coherent_integration=False,
        signal_interpolation=interpolation,
    )[0]  # (H,W)


def effective_window(waveform, spatial_bw, oversample=64):
    """|w(z)| on a fine grid: the window itself, or the transmit pulse's autocorrelation."""
    x, z, dz = make_transmit_waveform(waveform, spatial_bw, oversample=oversample)
    w = np.correlate(x, x, mode='same') if waveform in ('lfm', 'barker13') else x
    w = np.abs(w)
    return w / w.max(), z, dz


def reference_cbp(sim, waveform, plane, pixels):
    """
    The backprojection of projected_CBP evaluated analytically: each pulse's projection is the
    continuous |w| of the target, ramp-filtered up to the sampling Nyquist with the same per-pulse
    weight CBP_2D applies, and read at each pixel's exact projected range. No interposummation
    sampling and no readout interpolation, so the gap to the simulator is discretization alone.
    """
    w, z, dz = effective_window(waveform, sim['spatial_bw'])
    fs = sim['spatial_fs']

    traj = sim['perceived_trajectory'][0].detach().cpu().numpy()       # (P,3)
    forward = -traj / np.linalg.norm(traj, axis=-1, keepdims=True)
    ground = np.linalg.norm(forward[:, :2], axis=-1)                   # (P,) cos(elevation)
    line = forward[:, :2] / ground[:, None]                            # (P,2)

    # pixel grid and rotation, as in CBP_2D
    theta = float(sim['cam_azimuth_deg'][0] + 90) * np.pi / 180.0
    x = np.linspace(-plane / 2, plane / 2, pixels)
    y = np.linspace(plane / 2, -plane / 2, pixels)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    wx = np.cos(theta) * xx - np.sin(theta) * yy
    wy = np.sin(theta) * xx + np.cos(theta) * yy
    q = sim['target'].detach().cpu().numpy()[:2]

    spectrum = np.fft.fft(np.fft.ifftshift(w))
    image = np.zeros(xx.shape, dtype=np.complex128)
    for p in range(len(ground)):
        k = np.fft.fftfreq(len(z), d=dz / ground[p])                   # ground-range frequency
        # CBP_2D's |r| ramp is |k| / ground^2 up to a shared constant, cut at the sample Nyquist
        ramp = np.abs(k) * (np.abs(k) <= fs * ground[p] / 2) / ground[p] ** 2
        filtered = np.fft.fftshift(np.fft.ifft(spectrum * ramp))
        r = (wx - q[0]) * line[p, 0] + (wy - q[1]) * line[p, 1]
        rho = z / ground[p]
        image += np.interp(r, rho, filtered.real) + 1j * np.interp(r, rho, filtered.imag)
    return torch.tensor(np.abs(image))


def _peak_offset(cut, idx):
    """Quadratic sub-pixel refinement of a discrete peak, in samples."""
    if idx <= 0 or idx >= len(cut) - 1:
        return 0.0
    a, b, c = float(cut[idx - 1]), float(cut[idx]), float(cut[idx + 1])
    denom = a - 2 * b + c
    return float(np.clip(0.5 * (a - c) / denom, -1, 1)) if abs(denom) > 1e-30 else 0.0


def run_single_target(pixels=1201, device='cuda'):
    """One point target at the scene origin, imaged over the default image plane."""
    plane = SAR['image_plane_width']
    sim = simulate_point_target((0.0, 0.0, 0.0), device=device, **POINT)
    image = image_cbp(sim['signals'], sim, plane, pixels, SAR['signal_interpolation'])
    metrics = impulse_response_metrics(image, sim, plane, plane)
    return dict(image=metrics['image'], plane=plane, metrics=metrics)


def run_peak_grid(n_side=5, pixels=1201, device='cuda'):
    """Image a grid of point targets in one scene and measure each peak against its true position."""
    plane = SAR['image_plane_width']
    offsets = np.linspace(-GRID_SPAN / 2, GRID_SPAN / 2, n_side)

    signals, sim = 0, None
    targets = [(float(tx), float(ty), 0.0) for tx in offsets for ty in offsets]
    for t in targets:
        sim = simulate_point_target(t, device=device, **POINT)
        signals = signals + sim['signals']
    image = image_cbp(signals, sim, plane, pixels, SAR['signal_interpolation']).detach().cpu().numpy()

    px = plane / (pixels - 1)
    half = int(round(0.5 * (offsets[1] - offsets[0]) / px))            # search window, half a spacing
    errors = []
    for t in targets:
        sim['target'] = torch.tensor(t)
        er, ec = expected_peak_pixel(sim, pixels, pixels, plane, plane)
        r0, c0 = int(round(er)) - half, int(round(ec)) - half
        patch = image[r0:r0 + 2 * half + 1, c0:c0 + 2 * half + 1]
        pr, pc = np.unravel_index(np.argmax(patch), patch.shape)
        row = r0 + pr + _peak_offset(patch[:, pc], pr)
        col = c0 + pc + _peak_offset(patch[pr, :], pc)
        errors.append(np.hypot(row - er, col - ec) * px)
    errors = np.array(errors)
    return dict(image=image, plane=plane, targets=targets, errors=errors,
                default_pixel=plane / SAR['image_width'])


def run_impulse_response(fs_ratios=(2, 4, 8, 16), device='cuda'):
    """
    Impulse response at the scene origin per waveform and sample rate F_s = ratio * B_s: both CBP
    readouts vs the analytic CBP at the same F_s, whose ramp is cut at the same Nyquist.
    """
    rows = []
    for wf in WAVEFORMS:
        for ratio in fs_ratios:
            cfg = dict(POINT, waveform=wf, spatial_fs=ratio * POINT['spatial_bw'])
            sim = simulate_point_target((0.0, 0.0, 0.0), device=device, **cfg)
            entry = dict(waveform=wf, fs_ratio=ratio)
            for name in ('bilinear', 'sinc'):
                img = image_cbp(sim['signals'], sim, IRF_PLANE, IRF_PIXELS, name)
                entry[name] = impulse_response_metrics(img, sim, IRF_PLANE, IRF_PLANE)
            entry['reference'] = impulse_response_metrics(
                reference_cbp(sim, wf, IRF_PLANE, IRF_PIXELS), sim, IRF_PLANE, IRF_PLANE)
            rows.append(entry)
    return rows


def run_phase_history(device='cuda'):
    """Aperture phase residual of an off-center target, on the band-limited sinc so it reconstructs exactly."""
    sim = simulate_point_target((0.2, -0.15, 0.0), device=device, **dict(POINT, waveform='sinc'))
    return aperture_phase_residual(sim)


# --------------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------------

def _db(a):
    a = np.asarray(a, dtype=np.float64)
    return 20 * np.log10(np.clip(a / a.max(), 1e-12, None))


def plot_validation(single, grid, glint, out_path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) one point target, (b) grid of point targets
    for a, d, title in ((ax[0], single, '(a) single point target'),
                        (ax[1], grid, '(b) 5x5 point targets')):
        h = d['plane'] / 2
        im = a.imshow(_db(d['image']), cmap='gray', vmin=-40, vmax=0, extent=(-h, h, -h, h))
        a.set_title(title)
        a.set_xlabel('cross-range')
        a.set_ylabel('range')
        fig.colorbar(im, ax=a, label='dB')

    # (c) plate glint pattern vs physical optics
    for i, c in enumerate(glint['curves']):
        lab = r'$\lambda$ = %.3g' % c['wavelength']
        ax[2].plot(c['phi_deg'], _db(c['sigma']) / 2, 'C%d-' % i, lw=1.5, label=lab + ', simulator')
        ax[2].plot(c['phi_deg'], _db(c['po']) / 2, 'k--', lw=0.9,
                   label='physical optics' if i == 0 else None)
    ax[2].set_ylim(-45, 2)
    ax[2].set_xlabel('aspect off normal (deg)')
    ax[2].set_ylabel('normalized RCS (dB)')
    ax[2].set_title('(c) flat plate, L = %.2f' % glint['side'])
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------

def _rule(title):
    print('\n' + '=' * 78 + '\n' + title + '\n' + '=' * 78)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    os.makedirs('figures', exist_ok=True)

    _rule('point target, SAR defaults: ' + '  '.join('%s=%s' % kv for kv in POINT.items()))

    phase = run_phase_history(device)
    print('phase history vs exp(j 4 pi R / lambda), sinc waveform, target (0.2, -0.15)')
    print('  RMS residual %.3e deg   max %.3e deg   amplitude ripple %.3e dB'
          % (phase['rms_phase_err_deg'], phase['max_phase_err_deg'], phase['mag_ripple_db']))

    single = run_single_target(device=device)
    m = single['metrics']
    print('single target at the origin, %s readout: -3 dB width range %.5f  cross-range %.5f   '
          'PSLR range %.2f dB  cross-range %.2f dB   peak error %.5f'
          % (SAR['signal_interpolation'], m['range_res'], m['crossrange_res'],
             m['range_pslr_db'], m['crossrange_pslr_db'], m['peak_err_scene']))

    grid = run_peak_grid(device=device)
    e = grid['errors']
    print('\npeak position over %d targets within +/- %.1f, %s readout'
          % (len(e), GRID_SPAN / 2, SAR['signal_interpolation']))
    print('  error mean %.5f  max %.5f scene units  = max %.3f of a default %d px pixel (%.5f)'
          % (e.mean(), e.max(), e.max() / grid['default_pixel'], SAR['image_width'],
             grid['default_pixel']))

    irf = run_impulse_response(device=device)
    print('\nimpulse response at the origin, -3 dB widths in scene units (x analytic CBP)')
    print('%-9s %5s %-10s %9s %7s %9s %7s %9s %9s'
          % ('waveform', 'Fs/Bs', 'readout', 'range', '', 'cross', '', 'PSLR rng', 'PSLR crs'))
    for r in irf:
        ref = r['reference']
        for key in ('reference', 'sinc', 'bilinear'):
            m = r[key]
            print('%-9s %5d %-10s %9.5f %6.3fx %9.5f %6.3fx %8.2f %9.2f'
                  % (r['waveform'], r['fs_ratio'], key, m['range_res'],
                     m['range_res'] / ref['range_res'], m['crossrange_res'],
                     m['crossrange_res'] / ref['crossrange_res'],
                     m['range_pslr_db'], m['crossrange_pslr_db']))

    _rule('flat plate vs physical optics, baseline: '
          + '  '.join('%s=%s' % kv for kv in plate.BASELINE.items()))

    area = plate.run_area_sweep(device=device)
    print('RCS vs area: log-log slope %.4f   (physical optics 2)' % area['slope'])

    glint = plate.run_glint_sweep(device=device)
    print('\nglint pattern       -3 dB width (deg)      first null (deg)       PSLR (dB)')
    for c in glint['curves']:
        print('  lambda %.4f   %7.4f vs %7.4f     %7.4f vs %7.4f     %6.2f vs %6.2f'
              % (c['wavelength'], c['measured']['width_deg'], c['reference']['width_deg'],
                 c['measured']['null_deg'], c['po_null_deg'],
                 c['measured']['pslr_db'], c['reference']['pslr_db']))

    wav = plate.run_wavelength_sweep(device=device)
    print('\nspecular RCS vs wavelength: log-log slope %.4f   (physical optics %.1f)'
          % (wav['slope'], wav['po_slope']))

    sampling = plate.run_phase_sampling_sweep(device=device)
    print('\nplate at %.1f deg, %d^2 rays: RCS relative to specular vs physical optics'
          % (sampling['tilt_deg'], sampling['n_ray']))
    print('%10s %12s %10s' % ('lambda', 'rays/cycle', 'error dB'))
    for r in sampling['rows']:
        print('%10.5f %12.2f %10.2f' % (r['wavelength'], r['rays_per_cycle'], r['error_db']))

    out = get_next_path('figures/sar_validation.png')
    plot_validation(single, grid, glint, out)
    print('\nfigure written to %s' % out)


if __name__ == '__main__':
    main()
