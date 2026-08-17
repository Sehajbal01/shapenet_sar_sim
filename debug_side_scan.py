'''
Three-ping debug harness for sidescansonar.side_scan_sonar_image.

Reproduces the pipeline stage by stage instead of calling side_scan_sonar_image, so every
intermediate (poses, scatters, interpolated signals) is available to print and plot.
'''
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from accumulate_scatters import centered_linspace
from range_angle_images import beam_spread_weights
from signal_simulation import interpolate_signal, load_mesh
from sidescansonar import accumulate_scatters_side_scan, target_vertices
from utils import extract_pose_info


def debug_side_scan(
        obj_id = '100715345ee54d7ae38b52b4ee9d36a3',
        pose_num = None,
        device = 'cuda',

        num_pings = 3,
        track_length = None,
        elevation_fov_deg = 60.0,
        azimuth_beam_width_deg = 1.0,
        num_ray_width = 16,
        num_ray_height = 256,
        region_radius = None,

        wavelength = None,
        num_bounce = 1,
        spatial_bw = 64,
        spatial_fs = 64,
        window_func = 'sinc',

        make_ground = True,
        object_rotate_xyz = (90.0, 0.0, 0.0),
        obj_raids    = (1.0, 1.0, 100.0, 0.1, 0.9),
        ground_raids = (1.0, 1.0,   1.0, 0.9, 0.1),

        window_center_override = None,  # one-way range the interpolation window centers on
        tag = '',
    ):

    dataset_dir = '/workspace/data/srncars/cars_train/'
    models_dir  = '/workspace/data/srncars/02958343'

    if pose_num is None:
        pose_num = sorted(os.listdir(os.path.join(dataset_dir, obj_id, 'pose')))[0].split('.')[0]
    print('object %s  pose %s' % (obj_id, pose_num))

    pose_path = os.path.join(dataset_dir, obj_id, 'pose', '%s.txt' % pose_num)
    mesh_path = os.path.join(models_dir, obj_id, 'models', 'model_normalized.obj')
    poses_rgb = torch.tensor(np.loadtxt(pose_path).reshape(1, 4, 4).astype(np.float32), device=device)

    pose_info = extract_pose_info(poses_rgb)
    az, el = pose_info[6].item(), pose_info[5].item()
    mean_sensor_position = pose_info[0].reshape(3)
    print('rgb pose: az=%.2f deg  el=%.2f deg  dist=%.4f' % (az, el, pose_info[4].item()))
    print('mean_sensor_position:', mean_sensor_position.cpu().numpy())

    mesh, normals, material_properties = load_mesh(
        mesh_path, device=device, make_ground=make_ground,
        obj_raids=obj_raids, ground_raids=ground_raids, level_with_ground=True,
        rotate_xyz=object_rotate_xyz,
    )
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    # load_mesh renormalizes raids, so match on "same material as face 0" rather than on obj_raids
    obj_face_mask = (material_properties == material_properties[0:1]).all(dim=1)
    obj_verts = verts[faces[obj_face_mask].reshape(-1).unique()]
    print('mesh: %d verts  %d faces (%d object)' % (verts.shape[0], faces.shape[0], obj_face_mask.sum().item()))
    print('object bbox min: %s' % obj_verts.min(dim=0).values.cpu().numpy())
    print('object bbox max: %s' % obj_verts.max(dim=0).values.cpu().numpy())

    # ---------------------------------------------------------------- track geometry
    # mirrors side_scan_sonar_image; kept inline so every intermediate stays inspectable
    line_of_sight   = torch.nn.functional.normalize(-mean_sensor_position, dim=-1)
    world_up        = torch.tensor([0.0, 0.0, 1.0], device=device)
    track_direction = torch.nn.functional.normalize(torch.linalg.cross(line_of_sight, world_up), dim=-1)
    up_vector       = torch.linalg.cross(track_direction, line_of_sight)

    altitude             = mean_sensor_position[2]
    boresight_depression = torch.asin(-line_of_sight[2].clamp(-1.0, 1.0))
    half_fan             = np.pi / 180 * elevation_fov_deg / 2
    min_depression = (boresight_depression - half_fan).clamp_min(np.pi / 180)
    max_depression = (boresight_depression + half_fan).clamp_max(np.pi / 2)
    swath_near     = altitude / torch.sin(max_depression)
    swath_far      = altitude / torch.sin(min_depression)
    print('swath: one-way range [%.4f, %.4f]  depression [%.2f, %.2f] deg  altitude=%.4f'
          % (swath_near, swath_far, min_depression * 180 / np.pi, max_depression * 180 / np.pi, altitude))

    if region_radius is None:
        region_radius = float((swath_far - swath_near) / 2)
    if track_length is None:
        along_track    = target_vertices(mesh, material_properties) @ track_direction
        beam_footprint = 2 * float(torch.linalg.norm(mean_sensor_position)) * np.tan(
            np.pi / 180 * azimuth_beam_width_deg / 2)
        track_length   = float(along_track.max() - along_track.min()) + 2 * beam_footprint
        print('track_length from object extent %.4f + 2 x beam footprint %.4f = %.4f'
              % (float(along_track.max() - along_track.min()), beam_footprint, track_length))

    ping_offsets = centered_linspace(track_length, num_pings, device)
    trajectory   = (mean_sensor_position.reshape(1, 3)
                    + ping_offsets.reshape(num_pings, 1) * track_direction.reshape(1, 3))

    poses = torch.zeros(num_pings, 4, 4, device=device)
    poses[:, :3, 0] = track_direction
    poses[:, :3, 1] = up_vector
    poses[:, :3, 2] = line_of_sight
    poses[:, :3, 3] = trajectory
    poses[:,  3, 3] = 1.0

    print('\ntrack_length=%.4f  num_pings=%d  ping_offsets=%s'
          % (track_length, num_pings, ping_offsets.cpu().numpy()))
    print('line_of_sight  :', line_of_sight.cpu().numpy())
    print('track_direction:', track_direction.cpu().numpy())
    print('up_vector      :', up_vector.cpu().numpy())

    np.set_printoptions(precision=5, suppress=True)
    for p in range(num_pings):
        print('\n--- ping %d camera matrix (columns: right, up, forward, center) ---' % p)
        print(poses[p].cpu().numpy())
        info = extract_pose_info(poses[p:p+1])
        print('  |center|=%.4f  az=%.2f deg  el=%.2f deg' % (info[4].item(), info[6].item(), info[5].item()))

    # ---------------------------------------------------------------- ray trace
    fan_azimuth_fov_deg = azimuth_beam_width_deg * 3
    print('\nray fan: azimuth fov=%.2f deg (%d rays)  elevation fov=%.2f deg (%d rays)'
          % (fan_azimuth_fov_deg, num_ray_width, elevation_fov_deg, num_ray_height))

    scatter_ranges, scatter_energies, scatter_azimuths, _ = accumulate_scatters_side_scan(
        mesh, normals, material_properties, poses.unsqueeze(0),
        wavelength = wavelength,
        fov_width_deg  = fan_azimuth_fov_deg,
        fov_height_deg = elevation_fov_deg,
        n_ray_width    = num_ray_width,
        n_ray_height   = num_ray_height,
        num_bounce     = num_bounce,
    )

    raw_energies = [[e.clone() for e in et] for et in scatter_energies]
    beam_weights = [[beam_spread_weights(a, azimuth_beam_width_deg) for a in at] for at in scatter_azimuths]
    scatter_energies = [[e * w for e, w in zip(et, wt)] for et, wt in zip(scatter_energies, beam_weights)]

    if window_center_override is None:
        window_center = (swath_near + region_radius).reshape(1)
    else:
        window_center = torch.tensor([float(window_center_override)], device=device)
    print('\nrange window: center=%.4f  radius=%.4f  ->  one-way range [%.4f, %.4f]'
          % (window_center.item(), region_radius,
             window_center.item() - region_radius, window_center.item() + region_radius))

    for p in range(num_pings):
        r = scatter_ranges[0][p] / 2  # one-way
        a = scatter_azimuths[0][p]
        e = scatter_energies[0][p]
        in_win = ((r >= window_center - region_radius) & (r <= window_center + region_radius)).sum().item()
        print('ping %d: %7d scatters  one-way range [%.3f, %.3f]  %d in window (%.1f%%)  '
              'azimuth [%.3f, %.3f] deg  energy max=%.3e  post-beam sum=%.3e'
              % (p, r.numel(), r.min().item(), r.max().item(), in_win, 100 * in_win / max(r.numel(), 1),
                 a.min().item(), a.max().item(), raw_energies[0][p].max().item(), e.sum().item()))

    # ---------------------------------------------------------------- interpolate
    signals, sample_zs = [], []
    for p in range(num_pings):
        sig, sz = interpolate_signal(
            (scatter_ranges[0][p] / 2).unsqueeze(0), scatter_energies[0][p].unsqueeze(0),
            region_radius, window_center,
            spatial_bw=spatial_bw, spatial_fs=spatial_fs, window_func=window_func,
        )
        signals.append(sig.squeeze(0))
        sample_zs.append(sz.squeeze(0))
    signals   = torch.stack(signals)    # (P,Z)
    sample_zs = torch.stack(sample_zs)  # (P,Z)
    print('\nsignals (P,Z)=%s  sample_z [%.4f, %.4f]'
          % (tuple(signals.shape), sample_zs[0, 0].item(), sample_zs[0, -1].item()))
    for p in range(num_pings):
        print('ping %d signal: max=%.4e  mean=%.4e  argmax at z=%.4f'
              % (p, signals[p].abs().max().item(), signals[p].abs().mean().item(),
                 sample_zs[p, signals[p].abs().argmax()].item()))

    # how many rays land in each range bin. Below ~1 the seafloor is sampled more coarsely than
    # the range resolution, so the signal there is a sparse spike train rather than continuous.
    print('\nrays per range bin (bin width 1/spatial_fs = %.4f):' % (1.0 / spatial_fs))
    edges = torch.cat([sample_zs[0] - 0.5 / spatial_fs, sample_zs[0][-1:] + 0.5 / spatial_fs])
    for p in range(num_pings):
        counts = torch.histc((scatter_ranges[0][p] / 2).float(), bins=len(edges) - 1,
                             min=edges[0].item(), max=edges[-1].item())
        near = counts[:len(counts) // 2]
        far  = counts[len(counts) // 2:]
        print('ping %d: near half mean=%.1f  far half mean=%.1f  empty bins=%d/%d'
              % (p, near.mean().item(), far.mean().item(), int((counts == 0).sum()), len(counts)))

    os.makedirs('figures', exist_ok=True)
    plot_trajectory(obj_verts, trajectory, line_of_sight, track_direction, up_vector,
                    elevation_fov_deg, azimuth_beam_width_deg, tag)
    plot_scatters(scatter_ranges, raw_energies, scatter_energies, scatter_azimuths,
                  signals, sample_zs, window_center.item(), region_radius,
                  azimuth_beam_width_deg, tag)
    plot_image(signals, sample_zs, ping_offsets, tag)

    return poses, scatter_ranges, scatter_energies, scatter_azimuths, signals, sample_zs


def plot_trajectory(obj_verts, trajectory, line_of_sight, track_direction, up_vector,
                    elevation_fov_deg, azimuth_beam_width_deg, tag=''):
    '''Top-down and side views of where the platform is relative to the object.'''
    obj_verts = obj_verts.detach().cpu().numpy()
    traj = trajectory.detach().cpu().numpy()
    los  = line_of_sight.detach().cpu().numpy()
    trk  = track_direction.detach().cpu().numpy()
    up   = up_vector.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (i, j), name in zip(axes, [(0, 1), (0, 2)], ['x-y (top down)', 'x-z (side)']):
        ax.scatter(obj_verts[:, i], obj_verts[:, j], s=1, c='0.6', label='object verts')
        ax.plot(traj[:, i], traj[:, j], '-o', c='tab:blue', ms=8, label='trajectory (pings)')
        for p in range(traj.shape[0]):
            ax.annotate('p%d' % p, (traj[p, i], traj[p, j]), textcoords='offset points', xytext=(6, 6))
            ax.arrow(traj[p, i], traj[p, j], 0.35 * los[i], 0.35 * los[j],
                     color='tab:red', width=0.005, length_includes_head=True)
        ax.arrow(traj[0, i], traj[0, j], 0.25 * trk[i], 0.25 * trk[j],
                 color='tab:green', width=0.005, length_includes_head=True)
        ax.arrow(traj[0, i], traj[0, j], 0.25 * up[i], 0.25 * up[j],
                 color='tab:purple', width=0.005, length_includes_head=True)
        ax.plot(0, 0, 'k+', ms=14, label='origin')
        ax.set_title(name)
        ax.set_xlabel('xyz'[i]); ax.set_ylabel('xyz'[j])
        ax.axis('equal'); ax.grid(alpha=0.3)

    # elevation fan edges, drawn in the side view where they are in plane
    hw = np.deg2rad(elevation_fov_deg) / 2
    for sign in (+1, -1):
        d = np.cos(hw) * los + sign * np.sin(hw) * up
        axes[1].arrow(traj[0, 0], traj[0, 2], 2.0 * d[0], 2.0 * d[2],
                      color='tab:orange', width=0.003, alpha=0.7, length_includes_head=True)

    # azimuth beam edges (the FWHM, not the 3x-wider ray fan) in the top-down view: this is the
    # along-track slice of the scene each ping actually sees
    haz = np.deg2rad(azimuth_beam_width_deg) / 2
    for p in range(traj.shape[0]):
        for sign in (+1, -1):
            d = np.cos(haz) * los + sign * np.sin(haz) * trk
            axes[0].plot([traj[p, 0], traj[p, 0] + 2.0 * d[0]],
                         [traj[p, 1], traj[p, 1] + 2.0 * d[1]],
                         color='tab:orange', lw=1, alpha=0.8)

    axes[0].legend(loc='best', fontsize=8)
    fig.suptitle('side scan track: red=boresight, green=track/right, purple=up, orange=elevation fan')
    fig.tight_layout()
    fig.savefig('figures/debug_sss_trajectory%s.png' % tag, dpi=120)
    plt.close(fig)
    print('saved figures/debug_sss_trajectory%s.png' % tag)


def plot_scatters(scatter_ranges, raw_energies, weighted_energies, scatter_azimuths,
                  signals, sample_zs, window_center, region_radius, beam_width_deg, tag=''):
    '''Per-ping: every returned scatter in range/azimuth/energy, and the signal it interpolates to.'''
    P = len(scatter_ranges[0])
    fig, axes = plt.subplots(P, 3, figsize=(18, 4 * P), squeeze=False)

    z_lo, z_hi = window_center - region_radius, window_center + region_radius

    for p in range(P):
        r = (scatter_ranges[0][p] / 2).detach().cpu().numpy()
        a = scatter_azimuths[0][p].detach().cpu().numpy()
        e_raw = raw_energies[0][p].abs().detach().cpu().numpy()
        e_w   = weighted_energies[0][p].abs().detach().cpu().numpy()
        sz  = sample_zs[p].detach().cpu().numpy()
        sig = signals[p].abs().detach().cpu().numpy()

        ax = axes[p][0]
        ax.scatter(r, e_raw, s=2, alpha=0.4, label='raw')
        ax.scatter(r, e_w,   s=2, alpha=0.4, label='beam weighted')
        ax.axvspan(z_lo, z_hi, color='tab:green', alpha=0.12, label='range window')
        ax.set_yscale('log'); ax.set_xlabel('one-way range'); ax.set_ylabel('energy')
        ax.set_title('ping %d: %d scatters vs range' % (p, r.size))
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[p][1]
        sc = ax.scatter(a, r, s=2, c=np.log10(np.maximum(e_w, 1e-30)), cmap='viridis')
        ax.axhspan(z_lo, z_hi, color='tab:green', alpha=0.12)
        for x in (-beam_width_deg / 2, beam_width_deg / 2):
            ax.axvline(x, color='tab:red', ls='--', lw=1)
        ax.set_xlabel('azimuth (deg)'); ax.set_ylabel('one-way range')
        ax.set_title('ping %d: scatter azimuth vs range (red = FWHM)' % p)
        fig.colorbar(sc, ax=ax, label='log10 weighted energy')
        ax.grid(alpha=0.3)

        ax = axes[p][2]
        ax.plot(sz, sig, lw=1, label='interpolated signal')
        in_win = (r >= sz.min()) & (r <= sz.max())
        if in_win.any():
            ax2 = ax.twinx()
            ax2.scatter(r[in_win], e_w[in_win], s=3, c='tab:orange', alpha=0.4)
            ax2.set_ylabel('scatter energy', color='tab:orange')
            ax2.set_yscale('log')
        ax.set_xlabel('one-way range'); ax.set_ylabel('|signal|')
        ax.set_title('ping %d: signal (Z=%d)' % (p, sz.size))
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig('figures/debug_sss_scatters%s.png' % tag, dpi=110)
    plt.close(fig)
    print('saved figures/debug_sss_scatters%s.png' % tag)


def plot_image(signals, sample_zs, ping_offsets, tag=''):
    '''The waterfall the pipeline would produce, in linear and dB.'''
    img = signals.abs().transpose(0, 1).flip(0).detach().cpu().numpy()  # (Z,P) near range at bottom
    rng = sample_zs[0].flip(0).detach().cpu().numpy()
    off = ping_offsets.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    extent = [off.min(), off.max(), rng.min(), rng.max()]
    axes[0].imshow(img[::-1], aspect='auto', extent=extent, cmap='gray', origin='lower')
    axes[0].set_title('linear')
    peak = img.max()
    db = 20 * np.log10(np.clip(img / max(peak, 1e-30), 1e-6, None))
    axes[1].imshow(db[::-1], aspect='auto', extent=extent, cmap='gray', origin='lower', vmin=-60, vmax=0)
    axes[1].set_title('dB (floor -60)')
    for ax in axes:
        ax.set_xlabel('along-track offset'); ax.set_ylabel('one-way range')
    fig.tight_layout()
    fig.savefig('figures/debug_sss_image%s.png' % tag, dpi=120)
    plt.close(fig)
    print('saved figures/debug_sss_image%s.png' % tag)


if __name__ == '__main__':
    debug_side_scan()
