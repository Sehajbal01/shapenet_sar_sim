"""
A2 - flat-plate validation harness.

Where A1 (`validation_point_target.py`) bypasses the ray tracer to isolate the
signal/imaging chain, A2 drives the *whole* forward model -- ray tracer,
`accumulate_scatters`, `interpolate_signal` -- against the physical-optics closed
form for a flat rectangular plate, the canonical primitive every SAR simulator is
expected to reproduce.

Geometry. The plate sits at the scene origin with its normal along the line of
sight; the sensor sits at elevation 0 so that sweeping sensor azimuth by phi is
*exactly* a rotation about an in-plane axis of the plate. That makes the PO
reference one-dimensional and exact:

    sigma(phi) = 4 pi A^2 / lambda^2 * cos^2(phi) * sinc^2( 2 L sin(phi) / lambda )

with A = L^2 and np.sinc(x) = sin(pi x)/(pi x), so the first null is at
sin(phi) = lambda / (2 L).

Observable. Every experiment below reports

    sigma_hat = | sum_r E_r exp(j 2 pi z_r / lambda) |^2

i.e. the squared magnitude of the coherent sum of the scatter phasors that
`accumulate_scatters` returns. That is the monostatic RCS in arbitrary units, and
it is taken *before* image formation, so none of the imaging defects recorded in
claudes_plan.md sec. 2.5-2.7 contaminate it.

Experiments:
    A2a  RCS vs plate area          -- expects sigma ~ A^2
    A2b  RCS vs aspect angle        -- expects the PO sinc^2 glint pattern
    A2c  RCS vs wavelength          -- PO expects sigma ~ 1/lambda^2
    A2d  energy vs amplitude phasor -- resolves claudes_plan.md sec. 2.2
    A2e  ray-density convergence    -- checks the sec. 2.1 normalization
    A2f  far-field consistency      -- checks the sec. 2.6 consequence end to end

Run:
    python validation_plate.py
"""
import contextlib
import io
import os

# see paper_figures.py -- MKL and torch each bring their own OpenMP runtime
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch3d.structures import Meshes

from accumulate_scatters import accumulate_scatters
from imaging_algorithms import projected_CBP
from signal_simulation import generate_trajectory, interpolate_signal
from utils import (
    correct_material_properties,
    generate_pose_mat,
    get_next_path,
    savefig,
    spherical_to_cartesian,
)


BASELINE = dict(
    plate_side=0.40,        # scene units
    elevation_deg=0.0,      # sensor elevation; 0 makes the azimuth sweep an exact in-plane rotation
    sensor_distance=1.3,
    wavelength=0.05,        # L/lambda = 8, so the glint mainlobe is well inside the sweep
    grid_width=1.0,         # ray grid, held FIXED so hit count tracks plate area
    n_ray=512,
    n_subdiv=8,             # triangles per plate edge
    scatter_fraction=0.9,   # s in the r+s+a=1 budget
)

# Peak sidelobe of an unwindowed sinc, the PO pattern's own sidelobe level.
SINC_PSLR_DB = -13.26

FIG_DIR = 'figures/validation'


# --------------------------------------------------------------------------------
# scene construction
# --------------------------------------------------------------------------------

def _plate_frame(tilt_deg, elevation_deg, device):
    """
    Orthonormal frame for a plate whose normal is tilted `tilt_deg` off the line of
    sight, rotated about the plate's own vertical in-plane axis.

    Returns (n, u, v): n the plate normal, u the in-plane axis the tilt rotates
    within (so range varies linearly along u), v the in-plane axis it does not.
    """
    az = torch.tensor(0.0, device=device)
    el = torch.tensor(float(elevation_deg), device=device)
    one = torch.tensor(1.0, device=device)

    n0 = spherical_to_cartesian(az, el, one)                      # (3,) origin -> sensor
    z = torch.tensor([0.0, 0.0, 1.0], device=device)
    u0 = torch.nn.functional.normalize(torch.linalg.cross(z, n0), dim=0)
    v0 = torch.linalg.cross(n0, u0)                               # completes a right-handed frame

    phi = float(tilt_deg) * np.pi / 180.0
    n = float(np.cos(phi)) * n0 + float(np.sin(phi)) * u0
    u = float(np.cos(phi)) * u0 - float(np.sin(phi)) * n0
    return n, u, v0


def ground_frame(device):
    """Frame for a patch lying flat in the ground plane (normal +z, spanning x and y)."""
    e = torch.eye(3, device=device)
    return e[2], e[0], e[1]


def make_plate(side, tilt_deg=0.0, center=(0.0, 0.0, 0.0), scatter_fraction=0.9,
               diffuse=False, n_subdiv=8, elevation_deg=0.0, frame=None, device='cuda'):
    """
    One square plate as (verts, faces, raids).

    Materials use the (r, a, i, d, s) convention of `correct_material_properties`:
    r + s + a = 1 and i + d = 1. `scatter_fraction` is s, the fraction of incident
    *power* the surface scatters -- the quantity A2d asks whether the coherent sum
    weights correctly. r = 0 because these experiments are single-bounce.

    `frame` overrides the tilt-derived (n, u, v) basis, e.g. `ground_frame` for a
    horizontal patch.
    """
    n, u, v = frame if frame is not None else _plate_frame(tilt_deg, elevation_deg, device)

    t = torch.linspace(-side / 2, side / 2, n_subdiv + 1, device=device)
    su, sv = torch.meshgrid(t, t, indexing='ij')                  # (K,K)
    c = torch.tensor(center, device=device, dtype=torch.float32)
    verts = (c.reshape(1, 1, 3)
             + su.unsqueeze(-1) * u.reshape(1, 1, 3)
             + sv.unsqueeze(-1) * v.reshape(1, 1, 3))             # (K,K,3)
    K = n_subdiv + 1
    verts = verts.reshape(-1, 3)

    # two triangles per grid square, wound consistently (as in make_big_ground)
    i, j = torch.meshgrid(torch.arange(K - 1, device=device),
                          torch.arange(K - 1, device=device), indexing='ij')
    a_ = (i * K + j).reshape(-1)
    b_ = ((i + 1) * K + j).reshape(-1)
    c_ = (i * K + j + 1).reshape(-1)
    d_ = ((i + 1) * K + j + 1).reshape(-1)
    faces = torch.cat([torch.stack([a_, b_, c_], dim=-1),
                       torch.stack([b_, d_, c_], dim=-1)], dim=0).long()

    s = float(scatter_fraction)
    # (r, a, i, d, s): pure specular unless `diffuse`, which gives an aspect-independent
    # return -- useful when the target has to stay visible across a wide aperture (A2f)
    raids = (0.0, 1.0 - s, 0.0 if diffuse else 1.0, 1.0 if diffuse else 0.0, s)
    raids = torch.tensor(raids, device=device, dtype=torch.float32).reshape(1, 5).repeat(faces.shape[0], 1)
    return verts, faces, raids


def build_scene(plates, device='cuda'):
    """Concatenate plate primitives into the (mesh, face_normals, materials) triple."""
    all_verts, all_faces, all_raids = [], [], []
    offset = 0
    for verts, faces, raids in plates:
        all_verts.append(verts)
        all_faces.append(faces + offset)
        all_raids.append(raids)
        offset += verts.shape[0]

    verts = torch.cat(all_verts, dim=0)
    faces = torch.cat(all_faces, dim=0)
    raids = correct_material_properties(torch.cat(all_raids, dim=0))

    face_verts = verts[faces]
    edge_1 = face_verts[:, 1] - face_verts[:, 0]
    edge_2 = face_verts[:, 2] - face_verts[:, 0]
    face_normals = torch.nn.functional.normalize(torch.linalg.cross(edge_1, edge_2, dim=1), dim=1)

    mesh = Meshes(verts=[verts], faces=[faces])
    mesh.edge_1 = edge_1
    mesh.edge_2 = edge_2
    return mesh, face_normals, raids


# --------------------------------------------------------------------------------
# forward model
# --------------------------------------------------------------------------------

def _quiet(fn, *args, **kwargs):
    """Run fn with stdout swallowed -- accumulate_scatters prints a timing line per call."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def scatter_from_plates(plates, sensor_az_deg, elevation_deg=0.0, sensor_distance=1.3,
                        wavelength=0.05, grid_width=1.0, n_ray=512,
                        num_bounce=1, device='cuda'):
    """
    Ray-trace a plate scene from one or more sensor azimuths.

    `sensor_az_deg` is array-like; each entry becomes one pulse, so the whole aspect
    sweep is a single accumulate_scatters call over one octree.

    Returns (ranges, energies, trajectory): lists of P 1-D tensors plus the (1,P,3)
    sensor positions. `energies` already carries the exp(+j 2 pi z / lambda) phasor.
    """
    mesh, face_normals, materials = build_scene(plates, device=device)

    az = torch.as_tensor(np.atleast_1d(sensor_az_deg), device=device, dtype=torch.float32)
    el = torch.full_like(az, float(elevation_deg))
    dist = torch.full_like(az, float(sensor_distance))
    trajectory = spherical_to_cartesian(az, el, dist).reshape(1, -1, 3)   # (1,P,3)

    ranges, energies, _ = _quiet(
        accumulate_scatters,
        mesh, face_normals, materials, trajectory,
        wavelength=wavelength,
        grid_width=grid_width, grid_height=grid_width,
        n_ray_width=n_ray, n_ray_height=n_ray,
        num_bounce=num_bounce,
        second_bounce_batch_size=2 ** 18,
    )
    return ranges[0], energies[0], trajectory


def coherent_rcs(energy):
    """sigma_hat = |sum of scatter phasors|^2, in arbitrary (uncalibrated) units."""
    if energy.numel() == 0:
        return 0.0
    return float(torch.abs(energy.sum()) ** 2)


def po_plate_rcs(side, tilt_rad, wavelength):
    """Monostatic physical-optics RCS of a square plate, rotated about an in-plane axis."""
    area = side ** 2
    tilt_rad = np.asarray(tilt_rad, dtype=np.float64)
    return (4 * np.pi * area ** 2 / wavelength ** 2
            * np.cos(tilt_rad) ** 2
            * np.sinc(2 * side * np.sin(tilt_rad) / wavelength) ** 2)


def _loglog_slope(x, y):
    """Least-squares slope of log(y) vs log(x), ignoring non-positive samples."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float('nan')
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


# --------------------------------------------------------------------------------
# A2a - RCS vs plate area
# --------------------------------------------------------------------------------

def run_area_sweep(sides=(0.10, 0.14, 0.20, 0.28, 0.40), device='cuda', **overrides):
    """
    PO says sigma ~ A^2, i.e. field amplitude ~ area.

    The ray grid is held fixed while the plate grows, so the hit count tracks plate
    area directly; a grid scaled with the plate would divide the area back out.
    """
    cfg = dict(BASELINE); cfg.update(overrides)
    rows = []
    for side in sides:
        plate = make_plate(side, scatter_fraction=cfg['scatter_fraction'],
                           n_subdiv=cfg['n_subdiv'], elevation_deg=cfg['elevation_deg'],
                           device=device)
        _, energy, _ = scatter_from_plates(
            [plate], [0.0], elevation_deg=cfg['elevation_deg'],
            sensor_distance=cfg['sensor_distance'], wavelength=cfg['wavelength'],
            grid_width=cfg['grid_width'], n_ray=cfg['n_ray'], device=device,
        )
        rows.append(dict(side=side, area=side ** 2, n_hit=int(energy[0].numel()),
                         sigma=coherent_rcs(energy[0]),
                         po=float(po_plate_rcs(side, 0.0, cfg['wavelength']))))

    slope = _loglog_slope([r['area'] for r in rows], [r['sigma'] for r in rows])
    return dict(rows=rows, slope=slope, expected_slope=2.0, cfg=cfg)


# --------------------------------------------------------------------------------
# A2b - glint pattern vs aspect
# --------------------------------------------------------------------------------

def _pattern_metrics(phi_deg, sigma):
    """-3 dB width, first-null angle and peak sidelobe of an angular pattern."""
    phi = np.asarray(phi_deg, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    k = int(np.argmax(s))
    peak = s[k]
    if peak <= 0:
        return dict(width_deg=float('nan'), null_deg=float('nan'), pslr_db=float('nan'))

    def cross(direction, level):
        i = k
        while 0 < i + direction < len(s) - 1:
            j = i + direction
            if s[j] <= level:
                f = (s[i] - level) / (s[i] - s[j]) if s[i] != s[j] else 0.0
                return phi[i] + f * (phi[j] - phi[i])
            i = j
        return float('nan')

    width = abs(cross(+1, peak / 2) - cross(-1, peak / 2))       # -3 dB in power

    def null(direction):
        i = k
        while 0 < i + direction < len(s) - 1:
            j = i + direction
            if s[j] > s[i]:
                return phi[i]
            i = j
        return float('nan')

    lo_i = int(np.argmin(np.abs(phi - null(-1))))
    hi_i = int(np.argmin(np.abs(phi - null(+1))))
    side = np.concatenate([s[:lo_i + 1], s[hi_i:]])
    pslr = 10 * np.log10(side.max() / peak) if side.size else float('nan')
    return dict(width_deg=width,
                null_deg=0.5 * (abs(null(-1)) + abs(null(+1))),
                pslr_db=pslr)


def run_glint_sweep(wavelengths=(0.05, 0.025), phi_max_deg=12.0, n_angles=481,
                    device='cuda', **overrides):
    """
    Sweep the sensor azimuth through specular and compare the glint pattern to PO.

    With elevation 0 the azimuth offset *is* the plate's off-normal angle, so the
    reference is the exact 1-D PO expression rather than a small-angle stand-in.
    """
    cfg = dict(BASELINE); cfg.update(overrides)
    side = cfg['plate_side']
    phi = np.linspace(-phi_max_deg, phi_max_deg, n_angles)

    curves = []
    for lam in wavelengths:
        plate = make_plate(side, scatter_fraction=cfg['scatter_fraction'],
                           n_subdiv=cfg['n_subdiv'], elevation_deg=cfg['elevation_deg'],
                           device=device)
        _, energies, _ = scatter_from_plates(
            [plate], phi, elevation_deg=cfg['elevation_deg'],
            sensor_distance=cfg['sensor_distance'], wavelength=lam,
            grid_width=cfg['grid_width'], n_ray=cfg['n_ray'], device=device,
        )
        sigma = np.array([coherent_rcs(e) for e in energies])
        po = po_plate_rcs(side, phi * np.pi / 180.0, lam)

        m_meas = _pattern_metrics(phi, sigma)
        m_po = _pattern_metrics(phi, po)
        curves.append(dict(
            wavelength=lam, phi_deg=phi, sigma=sigma, po=po,
            measured=m_meas, reference=m_po,
            po_null_deg=float(np.degrees(np.arcsin(min(1.0, lam / (2 * side))))),
        ))

    return dict(curves=curves, side=side, cfg=cfg)


def plot_glint(result, out_path):
    curves = result['curves']
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 4.0), squeeze=False)
    for ax, c in zip(axes.flat, curves):
        norm = lambda a: 10 * np.log10(np.clip(a / a.max(), 1e-12, None))
        ax.plot(c['phi_deg'], norm(c['sigma']), lw=1.6, label='simulator')
        ax.plot(c['phi_deg'], norm(c['po']), lw=1.2, ls='--', label='physical optics')
        ax.axvline(c['po_null_deg'], color='0.6', lw=0.8, ls=':')
        ax.axvline(-c['po_null_deg'], color='0.6', lw=0.8, ls=':')
        ax.set_ylim(-45, 2)
        ax.set_xlabel('aspect off specular (deg)')
        ax.set_ylabel('normalized RCS (dB)')
        ax.set_title('L = %.2f, $\\lambda$ = %.3f  (L/$\\lambda$ = %.0f)'
                     % (result['side'], c['wavelength'], result['side'] / c['wavelength']))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('A2b  flat-plate glint pattern vs physical optics', y=1.02)
    fig.tight_layout()
    savefig(out_path)


# --------------------------------------------------------------------------------
# A2c - RCS vs wavelength
# --------------------------------------------------------------------------------

def run_wavelength_sweep(wavelengths=(0.2, 0.1, 0.05, 0.025, 0.0125), device='cuda', **overrides):
    """
    At specular every ray on a plate normal to the line of sight returns at the same
    range, so the coherent sum is wavelength-independent by construction. PO says
    sigma ~ 1/lambda^2. The gap is the aperture-diffraction factor geometric optics
    does not carry, and it is what an absolute RCS calibration would have to absorb.
    """
    cfg = dict(BASELINE); cfg.update(overrides)
    side = cfg['plate_side']
    rows = []
    for lam in wavelengths:
        plate = make_plate(side, scatter_fraction=cfg['scatter_fraction'],
                           n_subdiv=cfg['n_subdiv'], elevation_deg=cfg['elevation_deg'],
                           device=device)
        _, energy, _ = scatter_from_plates(
            [plate], [0.0], elevation_deg=cfg['elevation_deg'],
            sensor_distance=cfg['sensor_distance'], wavelength=lam,
            grid_width=cfg['grid_width'], n_ray=cfg['n_ray'], device=device,
        )
        rows.append(dict(wavelength=lam, sigma=coherent_rcs(energy[0]),
                         po=float(po_plate_rcs(side, 0.0, lam))))

    return dict(rows=rows,
                slope=_loglog_slope([r['wavelength'] for r in rows], [r['sigma'] for r in rows]),
                po_slope=_loglog_slope([r['wavelength'] for r in rows], [r['po'] for r in rows]),
                cfg=cfg)


# --------------------------------------------------------------------------------
# A2d - radiometric scaling of the returned energy (claudes_plan.md sec. 2.2)
# --------------------------------------------------------------------------------

def run_radiometric_scaling(fractions=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
                            contrast=(1.0, 0.1), device='cuda', **overrides):
    """
    Measure how the pipeline's RCS responds to the material scattering fraction `s`,
    and how that shows up as contrast between two targets.

    Reference: `s` is a fraction of incident *power* (it comes from the r+s+a=1
    budget), so RCS is linear in s and two targets with fractions s1, s2 stand apart
    by 10*log10(s1/s2) dB.

    (a) log-log slope of sigma_hat vs s
    (b) measured range-profile contrast between two plates at different ranges

    Both are reported as measured, against the closed form. This is a measurement of
    the current model, not a proposal to change it.
    """
    cfg = dict(BASELINE); cfg.update(overrides)
    side = cfg['plate_side']

    # (a) scaling law
    sigmas = []
    for s in fractions:
        plate = make_plate(side, scatter_fraction=s, n_subdiv=cfg['n_subdiv'],
                           elevation_deg=cfg['elevation_deg'], device=device)
        _, energy, _ = scatter_from_plates(
            [plate], [0.0], elevation_deg=cfg['elevation_deg'],
            sensor_distance=cfg['sensor_distance'], wavelength=cfg['wavelength'],
            grid_width=cfg['grid_width'], n_ray=cfg['n_ray'], device=device,
        )
        sigmas.append(coherent_rcs(energy[0]))
    scaling = dict(fractions=list(fractions), sigma=sigmas,
                   slope=_loglog_slope(fractions, sigmas))

    # (b) two plates, separated in range and laterally so neither occludes the other
    s1, s2 = contrast
    n0, u0, v0 = _plate_frame(0.0, cfg['elevation_deg'], device)

    # Sampling is chosen so both plates land exactly on range samples: region_radius
    # gives an odd sample count (so one sample sits at offset 0, where plate A is) and
    # the separation is an integer number of sample spacings. Otherwise each plate
    # picks up its own straddle loss -- up to 3.9 dB at half-sample offset -- and that
    # bias lands directly on the contrast being measured.
    fs = bw = 3650 / 50
    region_radius = 0.65                   # -> Z = int(2*0.65*73) + 1 = 95, odd
    sep_range = round(0.30 * fs) / fs      # along the line of sight, integer samples
    sep_lateral = 0.55 * side              # along the in-plane vertical axis

    small = side * 0.6
    plate_a = make_plate(small, scatter_fraction=s1, n_subdiv=cfg['n_subdiv'],
                         center=tuple((+sep_lateral * v0).tolist()),
                         elevation_deg=cfg['elevation_deg'], device=device)
    offset_b = (-sep_lateral * v0 - sep_range * n0)
    plate_b = make_plate(small, scatter_fraction=s2, n_subdiv=cfg['n_subdiv'],
                         center=tuple(offset_b.tolist()),
                         elevation_deg=cfg['elevation_deg'], device=device)

    truth_db = 10 * np.log10(s1 / s2)      # RCS ratio equals the power-fraction ratio
    ranges, energies, traj = scatter_from_plates(
        [plate_a, plate_b], [0.0], elevation_deg=cfg['elevation_deg'],
        sensor_distance=cfg['sensor_distance'], wavelength=cfg['wavelength'],
        grid_width=cfg['grid_width'], n_ray=cfg['n_ray'], device=device,
    )
    sig, sample_z = interpolate_signal(
        (ranges[0] / 2).reshape(1, -1), energies[0].reshape(1, -1),
        region_radius, torch.linalg.norm(traj[0, 0]).reshape(1),
        spatial_bw=bw, spatial_fs=fs, window_func='sinc',
    )
    mag = sig[0].abs().detach().cpu().numpy()
    z = sample_z[0].detach().cpu().numpy() - float(torch.linalg.norm(traj[0, 0]))

    # plate A sits at range offset 0; plate B is displaced sep_range along the line
    # of sight, which is exactly its one-way range offset after the /2 above
    near = mag[np.abs(z - 0.0) < sep_range / 3].max()
    far = mag[np.abs(z - sep_range) < sep_range / 3].max()
    profile = dict(z=z, mag=mag, measured_db=20 * np.log10(near / far))

    return dict(scaling=scaling, profile=profile, truth_db=truth_db,
                contrast=(s1, s2), cfg=cfg)


def plot_radiometric_scaling(result, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    d = result['scaling']
    f = np.array(d['fractions'], dtype=float)
    ax = axes[0]
    ax.loglog(f, np.array(d['sigma']) / d['sigma'][-1], 'o-',
              label='simulator: slope %.2f' % d['slope'])
    ax.loglog(f, f / f[-1], 'k--', lw=1.0, label='$\\sigma \\propto s$: slope 1')
    ax.set_xlabel('scattering fraction $s$ (power)')
    ax.set_ylabel('normalized RCS')
    ax.set_title('A2d(a)  RCS vs scattered power fraction')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    p = result['profile']
    ax.plot(p['z'], 20 * np.log10(np.clip(p['mag'] / p['mag'].max(), 1e-12, None)),
            lw=1.4, label='simulator: %.1f dB' % p['measured_db'])
    ax.axhline(-result['truth_db'], color='k', ls='--', lw=1.0,
               label='$10\\log_{10}(s_1/s_2)$ = %.1f dB' % result['truth_db'])
    ax.set_xlabel('one-way range offset (scene units)')
    ax.set_ylabel('normalized range profile (dB)')
    ax.set_title('A2d(b)  two plates, $s$ = %.2f and %.2f' % result['contrast'])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    savefig(out_path)


# --------------------------------------------------------------------------------
# A2e - ray-density convergence
# --------------------------------------------------------------------------------

def run_ray_density(n_rays=(64, 128, 256, 512, 1024), tilts=(0.0, 1.5, 4.0),
                    device='cuda', **overrides):
    """
    Check that the sec. 2.1 normalization by the transmitted ray count actually makes
    the answer ray-density invariant.

    Specular (tilt 0) is trivially converged -- every ray shares a range. Off
    specular, adjacent rays differ in range and the ray grid has to resolve that.
    Residual scatter is plate-edge quantization: the hit count steps in whole rays,
    so sigma_hat ~ (round(L / d_ray) * d_ray)^4.
    """
    cfg = dict(BASELINE); cfg.update(overrides)
    out = []
    for tilt in tilts:
        sigmas = []
        for n in n_rays:
            plate = make_plate(cfg['plate_side'], tilt_deg=0.0,
                               scatter_fraction=cfg['scatter_fraction'],
                               n_subdiv=cfg['n_subdiv'],
                               elevation_deg=cfg['elevation_deg'], device=device)
            _, energy, _ = scatter_from_plates(
                [plate], [tilt], elevation_deg=cfg['elevation_deg'],
                sensor_distance=cfg['sensor_distance'], wavelength=cfg['wavelength'],
                grid_width=cfg['grid_width'], n_ray=n, device=device,
            )
            sigmas.append(coherent_rcs(energy[0]))
        d_ray = cfg['grid_width'] / np.array(n_rays, dtype=float)
        nyquist = cfg['wavelength'] / (4 * max(np.sin(np.radians(tilt)), 1e-9))
        out.append(dict(tilt_deg=tilt, n_rays=list(n_rays), sigma=sigmas,
                        d_ray=d_ray, nyquist_d_ray=nyquist,
                        converged=sigmas[-1],
                        rel_err=[abs(s - sigmas[-1]) / sigmas[-1] if sigmas[-1] else float('nan')
                                 for s in sigmas]))
    return dict(sweeps=out, cfg=cfg)


def run_phase_sampling_sweep(u_values=(0.5, 1.5, 2.5, 4.5, 8.5, 12.5, 16.5,
                                       20.5, 24.5, 30.5, 40.5, 60.5),
                             tilt_deg=4.0, n_ray=128, device='cuda', **overrides):
    """
    Cross the sec. 2.3 phase-sampling limit Delta_ray < lambda / (4 sin theta) with an
    exact reference on the other side of it.

    The aspect is held fixed and the wavelength shrunk, so the number of phase cycles
    across the plate, u = 2 L sin(theta) / lambda, grows while the ray grid does not.
    Each lambda is chosen so u lands halfway between PO nulls -- on a sidelobe peak --
    which keeps the reference away from the zeros where any relative error blows up.

    Rays per phase cycle is lambda / (2 Delta_ray sin theta), so the sec. 2.3 rule is
    exactly "more than 2 rays per cycle". Below that the measured pattern must depart
    from PO, and the departure is aliasing, not physics.
    """
    cfg = dict(BASELINE); cfg.update(overrides)
    side = cfg['plate_side']
    phi = float(tilt_deg) * np.pi / 180.0
    d_ray = cfg['grid_width'] / (n_ray - 1)

    plate = make_plate(side, scatter_fraction=cfg['scatter_fraction'],
                       n_subdiv=cfg['n_subdiv'], elevation_deg=cfg['elevation_deg'],
                       device=device)

    rows = []
    for u in u_values:
        lam = 2 * side * np.sin(phi) / u
        _, energies, _ = scatter_from_plates(
            [plate], [0.0, tilt_deg], elevation_deg=cfg['elevation_deg'],
            sensor_distance=cfg['sensor_distance'], wavelength=lam,
            grid_width=cfg['grid_width'], n_ray=n_ray, device=device,
        )
        spec, off = coherent_rcs(energies[0]), coherent_rcs(energies[1])
        po_norm = float(np.cos(phi) ** 2 * np.sinc(u) ** 2)
        rows.append(dict(
            u=u, wavelength=lam,
            rays_per_cycle=lam / (2 * d_ray * np.sin(phi)),
            nyquist_ratio=d_ray / (lam / (4 * np.sin(phi))),
            measured_norm=off / spec if spec else float('nan'),
            po_norm=po_norm,
            error_db=10 * np.log10((off / spec) / po_norm) if spec and po_norm else float('nan'),
        ))
    return dict(rows=rows, tilt_deg=tilt_deg, n_ray=n_ray, d_ray=d_ray, cfg=cfg)


def plot_phase_sampling(result, out_path):
    rows = result['rows']
    rpc = np.array([r['rays_per_cycle'] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    ax.loglog(rpc, [r['measured_norm'] for r in rows], 'o-', label='simulator')
    ax.loglog(rpc, [r['po_norm'] for r in rows], 's--', label='physical optics')
    ax.axvline(2.0, color='r', lw=1.0, ls=':', label='$\\Delta_{ray} = \\lambda/(4\\sin\\theta)$')
    ax.set_xlabel('rays per phase cycle across the plate')
    ax.set_ylabel('RCS normalized to specular')
    ax.set_title('A2e(b)  crossing the phase-sampling limit')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ax.semilogx(rpc, [r['error_db'] for r in rows], 'o-')
    ax.axvline(2.0, color='r', lw=1.0, ls=':')
    ax.axhline(0.0, color='k', lw=0.8)
    ax.set_xlabel('rays per phase cycle across the plate')
    ax.set_ylabel('simulator $-$ physical optics (dB)')
    ax.set_title('aspect %.1f deg, %d$^2$ rays' % (result['tilt_deg'], result['n_ray']))
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    savefig(out_path)


# --------------------------------------------------------------------------------
# A2f - far-field consistency, end to end (claudes_plan.md sec. 2.6)
# --------------------------------------------------------------------------------

def run_farfield_consistency(radii=(0.0, 0.15, 0.30, 0.45), device='cuda'):
    """
    A1b measured a peak displacement ~ rho^2 / (2D) by feeding `projected_CBP` a
    *spherical* phase history. But `accumulate_scatters` computes range as
    2 * dot(hit - trajectory, forward), which is itself a planar-wavefront model
    (accumulate_scatters.py:260-265), and `projected_CBP` inverts exactly that
    (imaging_algorithms.py:63). If the two agree, a ray-traced target must land at
    the geometrically correct pixel no matter how far off origin it sits.

    This runs a small diffuse patch out to increasing radius through the real
    pipeline and measures the residual. Magnitude imaging is used (the paper's
    default) so the sec. 2.5 carrier-aliasing defect stays out of the measurement.
    """
    elevation_deg, sensor_distance = 30.0, 1.3
    azimuth_spread, num_pulses = 90.0, 64
    spatial_bw = spatial_fs = 3650 / 50
    region_radius = 1.7
    image_width = image_height = 512
    image_plane = 1.0
    patch = 0.04

    pose = generate_pose_mat(0.0, elevation_deg, sensor_distance, device=device).reshape(1, 4, 4)
    true_traj, perceived_traj, cam_az = generate_trajectory(
        pose, trajectory_type='circular', n_pulses=num_pulses, azimuth_spread_deg=azimuth_spread,
    )

    rows = []
    for rho in radii:
        # a flat, ground-lying, purely diffuse patch stays visible over the full aperture
        center = (float(rho), 0.0, 0.0)
        verts, faces, raids = make_plate(patch, center=center,
                                         scatter_fraction=0.9, diffuse=True,
                                         n_subdiv=4, frame=ground_frame(device), device=device)
        mesh, normals, materials = build_scene([(verts, faces, raids)], device=device)

        ranges, energies, _ = _quiet(
            accumulate_scatters, mesh, normals, materials, true_traj,
            wavelength=0.5, grid_width=1.2, grid_height=1.2,
            n_ray_width=512, n_ray_height=512, num_bounce=1,
            second_bounce_batch_size=2 ** 18,
        )

        sig, sz = [], []
        for p in range(num_pulses):
            s_p, z_p = interpolate_signal(
                (ranges[0][p] / 2).reshape(1, -1), energies[0][p].reshape(1, -1),
                region_radius, torch.linalg.norm(true_traj[0, p]).reshape(1),
                spatial_bw=spatial_bw, spatial_fs=spatial_fs, window_func='sinc',
            )
            sig.append(s_p[0]); sz.append(z_p[0])
        signals = torch.stack(sig).unsqueeze(0).abs()      # (1,P,Z) magnitude imaging
        sample_z = torch.stack(sz).unsqueeze(0)

        image = projected_CBP(
            signals, sample_z, perceived_traj, spatial_fs,
            image_plane_rotation_deg=cam_az + 90,
            image_width=image_width, image_height=image_height,
            image_plane_width=image_plane, image_plane_height=image_plane,
            batch_size=4096, coherent_integration=False, wavelength=0.5,
        )[0].detach().cpu().numpy()

        peak_row, peak_col = np.unravel_index(np.argmax(image), image.shape)

        # invert CBP_2D's pixel -> world rotation (imaging_algorithms.py:150-154)
        theta = float(cam_az[0] + 90) * np.pi / 180.0
        px = np.cos(theta) * rho
        py = -np.sin(theta) * rho
        exp_col = (px + image_plane / 2) / image_plane * (image_width - 1)
        exp_row = (image_plane / 2 - py) / image_plane * (image_height - 1)

        row_px = image_plane / (image_height - 1)
        rng_res = 1.0 / (spatial_bw * np.cos(np.radians(elevation_deg)))
        rows.append(dict(
            radius=rho,
            row_err_px=float(peak_row - exp_row), col_err_px=float(peak_col - exp_col),
            row_err_cells=float(abs(peak_row - exp_row) * row_px / rng_res),
            farfield_pred_cells=float(rho ** 2 / (2 * sensor_distance) / rng_res),
        ))
    return dict(rows=rows, range_res=1.0 / (spatial_bw * np.cos(np.radians(elevation_deg))))


# --------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------

def _rule(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(FIG_DIR, exist_ok=True)
    torch.manual_seed(0)

    print('A2 - flat-plate validation   (device: %s)' % device)
    print('baseline: ' + '  '.join('%s=%s' % kv for kv in BASELINE.items()))

    # ---------------------------------------------------------------- A2a
    _rule('A2a  RCS vs plate area      (physical optics: sigma ~ A^2)')
    area = run_area_sweep(device=device)
    print('%8s %10s %10s %14s %14s' % ('side', 'area', 'n_hit', 'sigma_hat', 'PO sigma'))
    for r in area['rows']:
        print('%8.3f %10.4f %10d %14.6e %14.6e' % (r['side'], r['area'], r['n_hit'],
                                                   r['sigma'], r['po']))
    print('log-log slope of sigma vs area: %.4f   (physical optics: %.1f)'
          % (area['slope'], area['expected_slope']))

    # ---------------------------------------------------------------- A2b
    _rule('A2b  glint pattern vs aspect')
    glint = run_glint_sweep(device=device)
    print('%10s %14s %14s %14s %14s %12s' % ('lambda', '-3dB meas', '-3dB PO',
                                             'null meas', 'null PO', 'PSLR meas'))
    for c in glint['curves']:
        print('%10.4f %13.4f%s %13.4f%s %13.4f%s %13.4f%s %11.2f dB' % (
            c['wavelength'],
            c['measured']['width_deg'], ' deg', c['reference']['width_deg'], ' deg',
            c['measured']['null_deg'], ' deg', c['po_null_deg'], ' deg',
            c['measured']['pslr_db']))
    print('sinc reference PSLR: %.2f dB' % SINC_PSLR_DB)
    plot_glint(glint, get_next_path(os.path.join(FIG_DIR, 'a2b_glint_pattern.png')))

    # ---------------------------------------------------------------- A2c
    _rule('A2c  RCS vs wavelength      (physical optics: sigma ~ 1/lambda^2)')
    wav = run_wavelength_sweep(device=device)
    print('%10s %16s %16s' % ('lambda', 'sigma_hat', 'PO sigma'))
    for r in wav['rows']:
        print('%10.4f %16.6e %16.6e' % (r['wavelength'], r['sigma'], r['po']))
    print('log-log slope vs lambda: simulator %.4f, physical optics %.4f'
          % (wav['slope'], wav['po_slope']))

    # ---------------------------------------------------------------- A2d
    _rule('A2d  radiometric scaling of the returned energy   (claudes_plan.md sec. 2.2)')
    ea = run_radiometric_scaling(device=device)
    print('(a) RCS vs scattered power fraction s')
    print('    simulator slope %.4f     sigma ~ s predicts 1' % ea['scaling']['slope'])
    print('(b) two plates, s = %.2f and %.2f -- 10*log10(s1/s2) = %.2f dB'
          % (ea['contrast'][0], ea['contrast'][1], ea['truth_db']))
    print('    simulator measured %.2f dB   (difference %+.2f dB)'
          % (ea['profile']['measured_db'], ea['profile']['measured_db'] - ea['truth_db']))
    plot_radiometric_scaling(ea, get_next_path(os.path.join(FIG_DIR, 'a2d_radiometric_scaling.png')))

    # ---------------------------------------------------------------- A2e
    _rule('A2e(a)  ray-density convergence')
    rd = run_ray_density(device=device)
    for sw in rd['sweeps']:
        print('tilt %.1f deg   Nyquist ray spacing lambda/(4 sin theta) = %.4g'
              % (sw['tilt_deg'], sw['nyquist_d_ray']))
        print('    %8s %12s %14s %12s' % ('n_ray', 'd_ray', 'sigma_hat', 'rel. err'))
        for n, d, s, e in zip(sw['n_rays'], sw['d_ray'], sw['sigma'], sw['rel_err']):
            print('    %8d %12.5f %14.6e %11.2f%%' % (n, d, s, 100 * e))

    _rule('A2e(b)  crossing the phase-sampling limit   (claudes_plan.md sec. 2.3)')
    ps = run_phase_sampling_sweep(device=device)
    print('aspect %.1f deg, %d^2 rays, d_ray = %.5f' % (ps['tilt_deg'], ps['n_ray'], ps['d_ray']))
    print('%8s %10s %14s %14s %14s %12s' % ('cycles', 'lambda', 'rays/cycle',
                                            'measured', 'phys. optics', 'error'))
    for r in ps['rows']:
        print('%8.1f %10.5f %14.2f %14.4e %14.4e %10.2f dB'
              % (r['u'], r['wavelength'], r['rays_per_cycle'],
                 r['measured_norm'], r['po_norm'], r['error_db']))
    plot_phase_sampling(ps, get_next_path(os.path.join(FIG_DIR, 'a2e_phase_sampling.png')))

    # ---------------------------------------------------------------- A2f
    _rule('A2f  far-field consistency, end to end   (claudes_plan.md sec. 2.6)')
    ff = run_farfield_consistency(device=device)
    print('range resolution %.5f scene units' % ff['range_res'])
    print('%8s %12s %12s %14s %18s' % ('radius', 'row err px', 'col err px',
                                       'row err cells', 'A1b prediction'))
    for r in ff['rows']:
        print('%8.2f %12.2f %12.2f %14.3f %18.3f' % (
            r['radius'], r['row_err_px'], r['col_err_px'],
            r['row_err_cells'], r['farfield_pred_cells']))

    print('\nfigures written to %s/' % FIG_DIR)


if __name__ == '__main__':
    main()
