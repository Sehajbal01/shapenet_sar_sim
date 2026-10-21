"""
Monostatic radar cross section of the SAR baseline car over sensor azimuth and elevation.

RCS is |sum_r E_r|^2, the squared magnitude of the coherent sum of the scatter phasors that
`accumulate_scatters` returns for one pulse (as in validation_plate.py), in arbitrary units.
Every look direction is one pulse of one orthographic ray grid aimed at the origin, so the
scene and octree are built once for the whole sweep.

Modes:
    polar    full az/el grid, drawn in polar coordinates with theta = az and r = 90 - el,
             so the zenith is the center and the horizon the rim
    az_cut   fixed elevation, varying azimuth (polar, theta = az, r = RCS in dB)
    el_cut   fixed azimuth, varying elevation (polar, theta = el, r = RCS in dB)

-cartesian draws the same on cartesian axes. -wavelength takes several values, which a cut overlays.

The car, materials, wavelength, bounce count and ray grid come from config.json's sar_baseline.
The ground is left out by default, so this is the car's free-space RCS; -ground adds it.
-flatplate SIZE swaps the car for a SIZE x SIZE plate on the xy plane (make_big_ground), with the
car's materials; the ray grid widens to the plate's corner-on width at the baseline ray spacing.
A plate's cuts carry its physical-optics RCS, dotted, with every curve normalized to its own peak.

    python calculate_rcs.py
    python calculate_rcs.py -mode az_cut -elevation 30
    python calculate_rcs.py -mode el_cut -azimuth 90
    python calculate_rcs.py -flatplate 1
    python calculate_rcs.py -flatplate 1 -mode el_cut -el_step 0.1 -wavelength 0.05 0.025 -cartesian
    python calculate_rcs.py -mode polar -replot      # redraw from the saved .npz
"""
import argparse
import os

# see sar_paper_figures.py -- MKL and torch each bring their own OpenMP runtime
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt

from accumulate_scatters import _quiet, accumulate_scatters
from config import CONFIG, SHAPENET_CARS_DIR
from ray_tracer_v2 import build_octree
from pytorch3d.structures import Meshes
from signal_simulation import load_mesh, make_big_ground
from utils import correct_material_properties, spherical_to_cartesian


RCS_BASELINE = dict(CONFIG['sar_baseline'])
OUTPUT_DIR = 'figures/rcs'


def load_scene(baseline, make_ground=False, device='cuda'):
    '''The baseline car as (mesh, normals, materials), leveled onto z = 0 as in the SAR renders.'''
    mesh_path = os.path.join(SHAPENET_CARS_DIR, baseline['obj_id'], 'models', 'model_normalized.obj')
    return _quiet(load_mesh, mesh_path, device=device,
                  make_ground=make_ground,
                  obj_raids=baseline['obj_raids'],
                  ground_raids=baseline['ground_raids'],
                  x_flip=baseline['object_x_flip'],
                  rotate_xyz=baseline['object_rotate_xyz'])


def flat_plate_scene(size, baseline, device='cuda'):
    '''A size x size plate on the xy plane (normal +z, centered at the origin) with the baseline's object materials.'''
    verts, faces = make_big_ground(size, 2, ground_level=0.0, max_triangle_len=size / 8, device=device)
    raids = torch.tensor(baseline['obj_raids'], device=device, dtype=torch.float32).reshape(1, 5).repeat(faces.shape[0], 1)

    face_verts = verts[faces]
    edge_1 = face_verts[:, 1] - face_verts[:, 0]
    edge_2 = face_verts[:, 2] - face_verts[:, 0]
    face_normals = torch.nn.functional.normalize(torch.linalg.cross(edge_1, edge_2, dim=1), dim=1)

    mesh = Meshes(verts=[verts], faces=[faces])
    mesh.edge_1 = edge_1
    mesh.edge_2 = edge_2
    return mesh, face_normals, correct_material_properties(raids)


def flat_plate_baseline(size, baseline):
    '''baseline with the ray grid widened to a size plate's corner-on width, at the same ray spacing.'''
    grid = max(baseline['grid_width'], 1.05 * np.sqrt(2) * size)
    n_ray = int(np.ceil(baseline['n_ray_width'] * grid / baseline['grid_width']))
    return {**baseline, 'grid_width': grid, 'grid_height': grid, 'n_ray_width': n_ray, 'n_ray_height': n_ray}


def calculate_rcs(az_deg, el_deg, baseline=RCS_BASELINE, make_ground=False, flatplate=None,
                  sensor_distance=1.3, pulses_per_call=64, device='cuda'):
    '''
    Monostatic RCS at each (az, el) pair.

    inputs:
        az_deg, el_deg (N,): look directions in deg, paired elementwise
        flatplate (float): image a flatplate x flatplate plate on the xy plane instead of the car
        sensor_distance (float): where the ray grid sits; the wavefront is planar, so it only
            has to clear the car
        pulses_per_call (int): look directions per accumulate_scatters call, for the progress bar

    outputs:
        sigma (N,): |sum of scatter phasors|^2, arbitrary units
    '''
    if flatplate is None:
        scene = load_scene(baseline, make_ground=make_ground, device=device)
    else:
        scene = flat_plate_scene(flatplate, baseline, device=device)
    octree = build_octree(scene[0])

    az = torch.as_tensor(np.asarray(az_deg), device=device, dtype=torch.float32)
    el = torch.as_tensor(np.asarray(el_deg), device=device, dtype=torch.float32)
    trajectory = spherical_to_cartesian(az, el, torch.full_like(az, sensor_distance))  # (N,3)

    sigma = []
    for start in tqdm.trange(0, len(az), pulses_per_call, desc='rcs'):
        _, energies, _ = _quiet(
            accumulate_scatters, *scene, trajectory[start:start + pulses_per_call].unsqueeze(0),
            wavelength=baseline['wavelength'],
            grid_width=baseline['grid_width'], grid_height=baseline['grid_height'],
            n_ray_width=baseline['n_ray_width'], n_ray_height=baseline['n_ray_height'],
            num_bounce=baseline['num_bounce'],
            second_bounce_batch_size=2 ** 18,
            octree=octree,
        )
        sigma += [float(torch.abs(e.sum()) ** 2) if e.numel() else 0.0 for e in energies[0]]
    return np.array(sigma)


def po_flat_plate(az_deg, el_deg, size, wavelength):
    '''
    Physical-optics monostatic RCS of a size x size plate on the xy plane: the square-aperture
    sinc^2 x sinc^2 times cos^2 of the angle off normal, which is sin^2(el).
    '''
    az, el = np.radians(az_deg), np.radians(el_deg)
    u, v = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az)       # look direction's x and y
    return (4 * np.pi * size ** 4 / wavelength ** 2 * np.sin(el) ** 2
            * np.sinc(2 * size * u / wavelength) ** 2 * np.sinc(2 * size * v / wavelength) ** 2)


def to_db(sigma, db_range, normalize=False):
    '''RCS in dB (arbitrary reference, or relative to its peak), floored db_range below the peak.'''
    db = 10 * np.log10(np.maximum(sigma, 1e-30))
    if normalize:
        db = db - db.max()
    return np.maximum(db, db.max() - db_range)


def draw_cut(ax, angle_deg, sigmas, wavelengths, db_range, theories=None, polar=False,
             closed=False, xlabel='angle (deg)'):
    '''
    One RCS curve per wavelength on ax, with its physical-optics curve dotted in the same color.
    With theories, every curve is normalized to its own peak, since the simulated level does not
    carry PO's 4 pi A^2 / lambda^2 gain. polar draws RCS in dB as the radius; closed wraps 360 deg.
    '''
    normalize = theories is not None
    angle = np.append(angle_deg, angle_deg[0] + 360) if closed else angle_deg
    x = np.radians(angle) if polar else angle
    wrap = (lambda a: np.append(a, a[0])) if closed else (lambda a: a)
    top = -np.inf
    for i, (sigma, lam) in enumerate(zip(sigmas, wavelengths)):
        db = wrap(to_db(sigma, db_range, normalize))
        top = max(top, db.max())
        ax.plot(x, db, color='C%d' % i, lw=1.2, label='λ = %g' % lam)
        if normalize:
            ax.plot(x, wrap(to_db(theories[i], db_range, normalize)), color='C%d' % i, lw=1.2,
                    ls=':', label='λ = %g, physical optics' % lam)
    ylabel = 'normalized RCS (dB)' if normalize else 'RCS (dB, arb.)'
    if polar:
        ax.set_rlim(top - db_range, top + 0.05 * db_range)
        ax.set_rlabel_position(22.5)
        ax.tick_params(axis='y', labelsize=8)
        if not closed:
            ax.set_thetamin(angle_deg.min())
            ax.set_thetamax(angle_deg.max())
    else:
        ax.set_ylim(top - db_range, top + 0.05 * db_range)
        ax.set_xlim(angle_deg.min(), angle_deg.max())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    if len(wavelengths) > 1 or normalize:
        ax.legend(fontsize=7, loc='lower left' if not polar else 'upper right')
    return ylabel


def plot_cut(angle_deg, sigmas, wavelengths, db_range, title, out_path, theories=None,
             cartesian=False, closed=False, xlabel='angle (deg)'):
    '''One cut to out_path; polar unless cartesian.'''
    if cartesian:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
    else:
        fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw=dict(projection='polar'))
    ylabel = draw_cut(ax, angle_deg, sigmas, wavelengths, db_range, theories=theories,
                      polar=not cartesian, closed=closed, xlabel=xlabel)
    ax.set_title(title if cartesian else title + '\n' + ylabel, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_map(az, el, sigma, db_range, out_path, cartesian=False):
    '''sigma (E,A) on the az/el grid; polar has theta = az, r = 90 - el.'''
    if cartesian:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        mesh = ax.pcolormesh(az, el, to_db(sigma, db_range), shading='nearest', cmap='viridis')
        ax.set_xlabel('azimuth (deg)')
        ax.set_ylabel('elevation (deg)')
        ax.set_title('RCS', fontsize=10)
    else:
        fig, ax = plt.subplots(figsize=(5.5, 4.6), subplot_kw=dict(projection='polar'))
        mesh = ax.pcolormesh(np.radians(az), 90 - el, to_db(sigma, db_range),
                             shading='nearest', cmap='viridis')
        r_ticks = np.arange(0, 90 - el.min() + 1e-6, 30)
        ax.set_rticks(r_ticks)
        ax.set_yticklabels(['%d°' % (90 - r) for r in r_ticks], fontsize=8)
        ax.set_rlim(0, 90 - el.min())
        ax.set_rlabel_position(22.5)
        ax.set_title('RCS, r = 90° − elevation, θ = azimuth', fontsize=10)
    fig.colorbar(mesh, ax=ax, pad=0.1 if not cartesian else 0.02, label='RCS (dB, arb.)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-mode', default='polar', choices=['polar', 'az_cut', 'el_cut'])
    parser.add_argument('-elevation', type=float, default=30.0, help='az_cut: fixed elevation (deg)')
    parser.add_argument('-azimuth', type=float, default=0.0, help='el_cut: fixed azimuth (deg)')
    parser.add_argument('-az_step', type=float, default=2.0, help='azimuth spacing (deg)')
    parser.add_argument('-el_step', type=float, default=2.0, help='elevation spacing (deg)')
    parser.add_argument('-el_range', type=float, nargs=2, default=(0.0, 90.0), metavar=('MIN', 'MAX'))
    parser.add_argument('-wavelength', type=float, nargs='+', default=None,
                        help='override the sar_baseline wavelength; several overlay on one cut')
    parser.add_argument('-num_bounce', type=int, default=None, help='override the sar_baseline num_bounce')
    parser.add_argument('-n_ray', type=int, default=None, help='override the sar_baseline ray grid, per side')
    parser.add_argument('-ground', action='store_true', help='include the ground plane')
    parser.add_argument('-flatplate', type=float, default=None, metavar='SIZE',
                        help='image a SIZE x SIZE flat plate on the xy plane instead of the car')
    parser.add_argument('-cartesian', action='store_true', help='cartesian axes instead of polar')
    parser.add_argument('-db_range', type=float, default=40.0, help='dynamic range shown below the peak')
    parser.add_argument('-replot', action='store_true', help='redraw from the saved .npz, no ray tracing')
    parser.add_argument('-output_dir', default=OUTPUT_DIR)
    args = parser.parse_args()

    baseline = dict(RCS_BASELINE)
    wavelengths = [baseline['wavelength']] if args.wavelength is None else args.wavelength
    assert args.mode != 'polar' or len(wavelengths) == 1, 'the az/el map takes one wavelength'
    if args.num_bounce is not None:
        baseline['num_bounce'] = args.num_bounce
    if args.n_ray is not None:
        baseline['n_ray_width'] = baseline['n_ray_height'] = args.n_ray
    if args.flatplate is not None:
        assert not args.ground, 'the plate sits where the ground would, at z = 0'
        baseline = flat_plate_baseline(args.flatplate, baseline)

    az_all = np.arange(0.0, 360.0, args.az_step)
    el_all = np.arange(args.el_range[0], args.el_range[1] + 1e-6, args.el_step)
    if args.mode == 'polar':
        el_grid, az_grid = np.meshgrid(el_all, az_all, indexing='ij')   # (E,A)
        tag = 'polar'
    elif args.mode == 'az_cut':
        az_grid, el_grid = az_all, np.full_like(az_all, args.elevation)
        tag = 'az_cut_el%g' % args.elevation
    else:
        el_grid, az_grid = el_all, np.full_like(el_all, args.azimuth)
        tag = 'el_cut_az%g' % args.azimuth
    if args.ground:
        tag += '_ground'
    if args.flatplate is not None:
        tag = 'flatplate%g_%s' % (args.flatplate, tag)
    if args.wavelength is not None:
        tag += '_wl' + '-'.join('%g' % w for w in wavelengths)

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, 'rcs_%s.npz' % tag)
    png_path = os.path.join(args.output_dir, 'rcs_%s%s.png' % (tag, '_cartesian' if args.cartesian else ''))

    if args.replot:
        data = np.load(npz_path)
        az_grid, el_grid, sigmas = data['az'], data['el'], data['sigma']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sigmas = []
        for lam in wavelengths:
            print('%s%s: %d look directions, wavelength %s, %d bounces, %dx%d rays, ground %s'
                  % (args.mode, '' if args.flatplate is None else ', %g flat plate' % args.flatplate,
                     az_grid.size, lam, baseline['num_bounce'],
                     baseline['n_ray_width'], baseline['n_ray_height'], args.ground))
            sigmas.append(calculate_rcs(az_grid.ravel(), el_grid.ravel(), baseline={**baseline, 'wavelength': lam},
                                        make_ground=args.ground, flatplate=args.flatplate,
                                        device=device).reshape(az_grid.shape))
        sigmas = np.stack(sigmas)                                       # (W, ...)
        np.savez(npz_path, az=az_grid, el=el_grid, sigma=sigmas,
                 obj_id='flatplate' if args.flatplate is not None else baseline['obj_id'],
                 wavelength=[np.nan if w is None else w for w in wavelengths],
                 num_bounce=baseline['num_bounce'], ground=args.ground)
    sigmas = sigmas.reshape((len(wavelengths),) + az_grid.shape)        # old single-wavelength .npz

    theories = None
    if args.flatplate is not None and None not in wavelengths:
        theories = [po_flat_plate(az_grid, el_grid, args.flatplate, lam) for lam in wavelengths]

    plate = '' if args.flatplate is None else '%g x %g plate, ' % (args.flatplate, args.flatplate)
    if args.mode == 'polar':
        plot_map(az_grid[0], el_grid[:, 0], sigmas[0], args.db_range, png_path, cartesian=args.cartesian)
    elif args.mode == 'az_cut':
        plot_cut(az_grid, sigmas, wavelengths, args.db_range,
                 plate + 'elevation %g°, varying azimuth' % el_grid[0], png_path,
                 theories=theories, cartesian=args.cartesian, closed=not args.cartesian,
                 xlabel='azimuth (deg)')
    else:
        plot_cut(el_grid, sigmas, wavelengths, args.db_range,
                 plate + 'azimuth %g°, varying elevation' % az_grid[0], png_path,
                 theories=theories, cartesian=args.cartesian, xlabel='elevation (deg)')

    for lam, sigma in zip(wavelengths, sigmas):
        k = np.unravel_index(np.argmax(sigma), sigma.shape)
        print('wavelength %s: peak %.2f dB (arb.) at az %.1f el %.1f'
              % (lam, 10 * np.log10(sigma[k]), az_grid[k], el_grid[k]))
    print('saved %s and %s' % (png_path, npz_path))


if __name__ == '__main__':
    main()
