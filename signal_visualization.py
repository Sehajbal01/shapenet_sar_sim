import io
import tqdm
import PIL
import imageio
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from utils import get_next_path


def plot_energy_scatter(ax, fig, ranges_p, energies_p, sz_min, sz_max, e_min, e_max):
    """Plot range vs energy as a raw scatter of points."""
    ax.scatter(ranges_p, energies_p, s=1)
    ax.set_title('Scatter')
    ax.set_xlabel('Range')
    ax.set_ylabel('Energy')
    ax.set_xlim(sz_min, sz_max)
    ax.set_ylim(e_min, e_max)


def plot_energy_hexbin(ax, fig, ranges_p, energies_p, sz_min, sz_max, e_min, e_max,
                       gridsize=80, cmap='viridis', count_min=1, count_max=None):
    """Plot range vs energy as a hexbin, coloring each cell by point count.

    Makes dense regions readable by encoding how many points fall in each
    hexagonal bin (which a raw scatter can't show once points overplot).

    Pass count_min/count_max to fix the color scale across frames so the
    colorbar stays constant in an animation.
    """
    hb = ax.hexbin(ranges_p, energies_p, gridsize=gridsize, cmap=cmap,
                   mincnt=1, extent=(sz_min, sz_max, e_min, e_max),
                   vmin=count_min, vmax=count_max)
    fig.colorbar(hb, ax=ax, label='count')
    ax.set_title('Hexbin (point count)')
    ax.set_xlabel('Range')
    ax.set_ylabel('Energy')
    ax.set_xlim(sz_min, sz_max)
    ax.set_ylim(e_min, e_max)


# registry of available range-vs-energy viewing methods for signal_gif
SCATTER_VIEWS = {
    'scatter': plot_energy_scatter,
    'hexbin': plot_energy_hexbin,
}


def signal_gif(signals, sample_z, debugging_maps, all_ranges, all_energies, region_radius,
               suffix=None, use_mp4_format=True, scatter_view='hexbin'):
    signals = torch.abs(signals)  # (T, P, Z)
    T, P, Z = signals.shape

    sig_min, sig_max = signals.min().item(), signals.max().item()

    # precompute energy axis limits across all pulses
    all_energies_cat = torch.cat([torch.abs(all_energies[0][p]) for p in range(P)])
    e_min, e_max = all_energies_cat.min().item(), all_energies_cat.max().item()

    # scatter x-axis matches sample_z extent (sensor_distance ± region_radius)
    sz_min, sz_max = sample_z.min().item(), sample_z.max().item()

    # precompute neighbor error ratio for all pulses
    sigs_np = signals[0].cpu().numpy()  # (P, Z)
    neighbor_error = np.zeros(P)
    for i in range(P):
        diffs = []
        if i > 0:
            diffs.append(np.sum((sigs_np[i] - sigs_np[i - 1]) ** 2))
        if i < P - 1:
            diffs.append(np.sum((sigs_np[i] - sigs_np[i + 1]) ** 2))
        neighbor_error[i] = np.mean(diffs)
    signal_energy = np.sum(sigs_np ** 2, axis=-1)  # (P,)
    neighbor_error_ratio = neighbor_error / (signal_energy + 1e-10)
    pulse_indices = np.arange(P)
    ne_max = neighbor_error_ratio.max() * 1.1 if neighbor_error_ratio.max() > 0 else 1.0

    if scatter_view not in SCATTER_VIEWS:
        raise ValueError("scatter_view must be one of %s, got %r" % (list(SCATTER_VIEWS), scatter_view))
    plot_energy_view = SCATTER_VIEWS[scatter_view]

    # precompute global max bin count so the hexbin colorbar stays constant
    view_kwargs = {}
    if scatter_view == 'hexbin':
        tmp_fig, tmp_ax = plt.subplots()
        count_max = 0
        for p in range(P):
            ranges_p   = all_ranges[0][p].cpu().numpy() / 2
            energies_p = torch.abs(all_energies[0][p]).cpu().numpy()
            hb = tmp_ax.hexbin(ranges_p, energies_p, gridsize=80, mincnt=1,
                               extent=(sz_min, sz_max, e_min, e_max))
            counts = hb.get_array()
            if counts.size:
                count_max = max(count_max, counts.max())
            tmp_ax.clear()
        plt.close(tmp_fig)
        view_kwargs['count_max'] = count_max

    images = []
    for p in tqdm.tqdm(range(P), desc='Creating MP4' if use_mp4_format else 'Creating GIF'):
        depth_map  = debugging_maps[(0, p)]['depth'].cpu().numpy()           # (H, W)
        energy_map = debugging_maps[(0, p)]['energy'].cpu().numpy()          # (H, W)
        sig        = signals[0, p].cpu().numpy()                             # (Z,)
        sz         = sample_z[0, p].cpu().numpy()                            # (Z,)
        ranges_p   = all_ranges[0][p].cpu().numpy() / 2                      # (R',) half round-trip
        energies_p = torch.abs(all_energies[0][p]).cpu().numpy()             # (R',)

        fig = plt.figure(figsize=(12, 13))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.6], hspace=0.35)

        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
        ax_ne = fig.add_subplot(gs[2, :])  # double-wide bottom panel

        ax00.imshow(depth_map,  cmap='gray')
        ax00.set_title('Depth Map')
        ax00.axis('off')

        ax01.imshow(energy_map, cmap='gray')
        ax01.set_title('Energy Map')
        ax01.axis('off')

        plot_energy_view(ax10, fig, ranges_p, energies_p, sz_min, sz_max, e_min, e_max, **view_kwargs)

        ax11.plot(sz, sig)
        ax11.set_title('Signal')
        ax11.set_xlabel('Range')
        ax11.set_ylabel('Amplitude')
        ax11.set_ylim(sig_min, sig_max)

        ax_ne.plot(pulse_indices, neighbor_error_ratio, color='steelblue')
        ax_ne.plot(p, neighbor_error_ratio[p], '*', color='red', markersize=14, zorder=5)
        ax_ne.set_title('Neighbor Error / Signal Energy')
        ax_ne.set_xlabel('Pulse Index')
        ax_ne.set_ylabel('Neighbor Error Ratio')
        ax_ne.set_xlim(0, P - 1)
        ax_ne.set_ylim(0, ne_max)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        frame = np.array(PIL.Image.open(buf))[..., :3]
        plt.close(fig)
        images.append(frame)

    # make a boomerang gif
    images = np.stack(images, axis=0)  # (N, H, W, C)
    images = np.concatenate((images, np.flip(images, axis=0)), axis=0)

    fps = P / 4.0
    if use_mp4_format:
        print('Saving MP4 with %.1f fps...' % fps)
        path = f'figures/dm_em_sc_si_{suffix}.mp4' if suffix is not None else get_next_path('figures/dm_em_sc_si.mp4')
        imageio.mimsave(path, images, fps=fps, format='FFMPEG')
        print('MP4 saved to: ', path)
    else:
        print('Saving GIF with %.1f fps...' % fps)
        path = f'figures/dm_em_sc_si_{suffix}.gif' if suffix is not None else get_next_path('figures/dm_em_sc_si.gif')
        imageio.mimsave(path, images, fps=fps, format='GIF', loop=0)
        print('GIF saved to: ', path)
