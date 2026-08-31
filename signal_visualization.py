import io
import os
import tqdm
import PIL
import imageio
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from utils import get_next_path, savefig
from signal_simulation import interpolate_signal
from imaging_algorithms import db_compress, asinh_compress


def plot_energy_scatter(ax, fig, ranges_p, energies_p):
    """Plot range vs energy as a raw scatter of points."""
    ax.scatter(ranges_p, energies_p, s=1)
    ax.set_title('Scatter')
    ax.set_xlabel('Range')
    ax.set_ylabel('Energy')


def plot_energy_hexbin(ax, fig, ranges_p, energies_p,
                       gridsize=80, cmap='viridis', count_min=1, count_max=None):
    """Plot range vs energy as a hexbin, coloring each cell by point count.

    Makes dense regions readable by encoding how many points fall in each
    hexagonal bin (which a raw scatter can't show once points overplot).

    Pass count_min/count_max to fix the color scale across frames so the
    colorbar stays constant in an animation.
    """
    hb = ax.hexbin(ranges_p, energies_p, gridsize=gridsize, cmap=cmap,
                   mincnt=1, vmin=count_min, vmax=count_max)
    fig.colorbar(hb, ax=ax, label='count')
    ax.set_title('Hexbin (point count)')
    ax.set_xlabel('Range')
    ax.set_ylabel('Energy')


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
            hb = tmp_ax.hexbin(ranges_p, energies_p, gridsize=80, mincnt=1)
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

        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35)

        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])

        ax00.imshow(depth_map,  cmap='gray')
        ax00.set_title('Depth Map')
        ax00.axis('off')

        ax01.imshow(energy_map, cmap='gray')
        ax01.set_title('Energy Map')
        ax01.axis('off')

        plot_energy_view(ax10, fig, ranges_p, energies_p, **view_kwargs)

        ax11.plot(sz, sig)
        ax11.set_title('Signal')
        ax11.set_xlabel('Range')
        ax11.set_ylabel('Amplitude')
        ax11.set_ylim(sig_min, sig_max)

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


def signal_column_image(signals, sample_z, ping_offsets=None, suffix=None,
                        compression='db', db_floor=-60.0, asinh_k_ratio=0.1, depression_deg=None):
    '''
    Save the interpolated signals as an image, one ping per column, left to right.

    This is the raw (range, along-track) picture at full range/ping resolution, before
    side_scan_sonar_image resamples it onto its own (slant range, cross range) pixel grid. So a
    feature in that final image can be traced back to a ping or blamed on the resampling.

    inputs:
        signals (T,P,Z): interpolated signal of each ping, real or complex
        sample_z (T,P,Z) or (T,Z): one-way range of each sample; every ping shares one range
            window, so ping 0's samples label the whole axis
        ping_offsets (P,): along-track offset of each ping; column index is used when None
        suffix (str): name for the saved file(s), numbered when None
        compression (str): how the still is displayed, matching render_side_scan_image's knob of
            the same name. 'db' shows dB relative to the brightest sample, floored at db_floor --
            the returns span ~100 dB, so a linear stretch is all near range. 'linear' is a plain
            stretch. 'asinh' arcsinh-compresses (asinh_compress) referenced to this still's own
            99.9th percentile sample, staying linear near zero and logarithmic past
            asinh_k_ratio * that reference, so it shows texture like dB does but without a hard
            floor
        db_floor (float): black point of the dB display, ignored unless compression == 'db'
        asinh_k_ratio (float): asinh softening scale as a fraction of this still's own reference
            level, ignored unless compression == 'asinh'
        depression_deg (float): boresight depression below horizontal. Given it, the range axis
            is projected to ground range and the two axes are drawn at one shared scale, so a
            metre down the image is a metre across it. Without it the range axis stays slant
            and the image is stretched to fill the figure

    outputs:
        paths (list[str]): the files written, one per track
    '''
    signals = torch.abs(signals)  # (T,P,Z)
    T, P, Z = signals.shape

    if sample_z.dim() == 3:
        sample_z = sample_z[:, 0, :]  # (T,Z) every ping shares the one range window

    labelled_track = ping_offsets is not None
    if ping_offsets is None:
        ping_offsets = torch.arange(P, device=signals.device)

    x = ping_offsets.detach().cpu().numpy()  # (P,)
    paths = []
    for t in range(T):
        # (P,Z) -> (Z,P), so the ping axis runs left to right and range runs down the rows
        columns = signals[t].detach().cpu().numpy().T  # (Z,P)
        z = sample_z[t].detach().cpu().numpy()         # (Z,)

        peak = columns.max()
        if compression == 'db' and peak > 0:
            columns = db_compress(columns, peak, db_floor)
            color_label, vmin, vmax = 'Amplitude (dB re peak)', db_floor, 0.0
        elif compression == 'asinh' and peak > 0:
            ref = float(np.percentile(columns, 99.9))
            if ref > 0:
                columns = asinh_compress(columns, asinh_k_ratio * ref, ref)
                color_label, vmin, vmax = 'Amplitude (asinh compressed)', 0.0, 1.0
            else:  # nearly all-dark still: 99.9th percentile can round to 0 even with peak > 0
                color_label, vmin, vmax = 'Amplitude', None, None
        else:
            color_label, vmin, vmax = 'Amplitude', None, None

        # slant range is not a ground distance: the altitude is fixed and the seafloor flat, so
        # ground range g = sqrt(R^2 - h^2) and dg/dR = 1/cos(depression). Linearizing that about
        # the window center keeps the rows evenly spaced, which imshow's extent needs, and puts
        # the window's region_radius half width at region_radius/cos(depression) on the ground.
        ground_scale = depression_deg is not None and labelled_track
        if ground_scale:
            cos_el   = float(np.cos(np.pi / 180 * depression_deg))
            z_center = (z[0] + z[-1]) / 2                # the window is centered on the target
            y        = (z - z_center) / cos_el           # (Z,) ground range offset from target
            y_label, aspect = 'Ground range offset from target', 'equal'
        else:
            y, y_label, aspect = z, 'One way range', 'auto'

        # half a step of margin, so imshow centers the edge pixels on their samples
        dx = (x[1] - x[0]) / 2 if P > 1 else 0.5
        dy = (y[1] - y[0]) / 2 if Z > 1 else 0.5

        fig, ax = plt.subplots(figsize=(10, 8))
        # origin='lower' puts near range at the bottom, matching side_scan_sonar_image's output.
        # aspect='equal' is what makes the two axes share a scale once y is a ground distance
        im = ax.imshow(columns, cmap='gray', origin='lower', aspect=aspect,
                       vmin=vmin, vmax=vmax,
                       extent=[x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy])
        # extra pad leaves the slant range axis room to sit between the image and the bar
        fig.colorbar(im, ax=ax, label=color_label, pad=0.12 if ground_scale else 0.05)
        ax.set_title('Interpolated signals, one column per ping')
        ax.set_xlabel('Along track offset' if labelled_track else 'Ping')
        ax.set_ylabel(y_label)
        if ground_scale:
            # keep the slant range readable, since that is the axis the samples actually live on.
            # default args bind this track's window, so a later track cannot rebind the closure
            slant = ax.secondary_yaxis(
                'right',
                functions=(lambda g, c = cos_el, z0 = z_center: z0 + g * c,
                           lambda r, c = cos_el, z0 = z_center: (r - z0) / c))
            slant.set_ylabel('One way range')

        track_suffix = '' if T == 1 else '_track%02d' % t
        path = ('figures/signal_columns_%s%s.png' % (suffix, track_suffix)
                if suffix is not None else get_next_path('figures/signal_columns.png'))
        savefig(path)
        print('Signal column image saved to: ', path)
        paths.append(path)

    return paths


def analyze_window_functions(
        window_funcs=('sinc', 'gaussian', 'lfm', 'barker13'),
        bw_list=(20.0,),
        fs_list=(40.0, 80.0),
        region_radius=1.0,
        db_floor=-60.0,
        device='cpu',
        save_dir='figures',
):
    """
    Characterize the range impulse response (point-spread function) of each
    interpolate_signal window.  A single unit scatter at z=0 is fed to
    interpolate_signal with sensor_distance=0, so the recovered signal is exactly
    the effective window w(z) sampled at the radar sampling frequency:
        s(z_o) = sum_r E_r w(-z_o - z_r/2) = w(-z_o)   (E=1, z=0).
    For each (bw, fs) pair this plots the impulse response in the time/range domain
    (linear and dB, to expose mainlobe width and range sidelobes) and its spectrum
    in the frequency domain (dB, to expose the spectral support and any aliasing
    when fs < bw).  Figures are written to save_dir for use in the paper.

    Inputs:
        window_funcs (iterable of str): window functions to compare
        bw_list (iterable of float): spatial bandwidths to sweep
        fs_list (iterable of float): spatial sampling frequencies to sweep
        region_radius (float): radius of the sampled range region
        db_floor (float): lower dB limit for the log-scale plots
        device (str): torch device to run on
        save_dir (str): directory to write the figures to
    """
    os.makedirs(save_dir, exist_ok=True)

    # human-readable legend labels for each window function
    window_labels = {
        'sinc': 'Sinc Interpolation',
        'gaussian': 'Gaussian Pulse',
        'lfm': 'LFM Chirp',
        'barker13': 'Barker 13',
    }

    # one unit scatter at the origin: shape (1, R=1)
    scatter_z = torch.zeros(1, 1, device=device)
    scatter_e = torch.ones(1, 1, device=device, dtype=torch.complex64)
    sensor_distance = torch.zeros(1, device=device)

    def to_db(mag):  # normalized magnitude -> dB, floored for plotting
        return db_compress(mag, float(mag.max()), db_floor)

    for bw in bw_list:
        for fs in fs_list:
            fig, (ax_lin, ax_freq) = plt.subplots(1, 2, figsize=(12, 5))

            for wf in window_funcs:
                signal, sample_z = interpolate_signal(
                    scatter_z, scatter_e, region_radius, sensor_distance,
                    spatial_bw=bw, spatial_fs=fs, window_func=wf,
                )
                s = signal[0].cpu().numpy()           # (Z,) complex impulse response
                z = sample_z[0].cpu().numpy()         # (Z,) range axis (sensor_distance=0)
                mag = np.abs(s)
                label = window_labels.get(wf, wf)

                # time/range domain
                ax_lin.plot(z, mag / (mag.max() + 1e-30), label=label, ms=3)

                # frequency domain: spectrum of the sampled impulse response
                Z = len(s)
                spec = np.fft.fftshift(np.fft.fft(s))
                freq = np.fft.fftshift(np.fft.fftfreq(Z, d=1.0 / fs))  # [-fs/2, fs/2)
                ax_freq.plot(freq, to_db(np.abs(spec)), label=label, ms=3)

            ax_lin.set_title('Range impulse response (linear)')
            ax_lin.set_xlabel('range z'); ax_lin.set_ylabel('|s| (normalized)')
            ax_lin.grid(True, alpha=0.3); ax_lin.legend()

            # mark the bandwidth edges +-bw/2 to show spectral support
            ax_freq.axvline(+bw / 2, color='k', ls='--', lw=0.8, alpha=0.5, label='+-bw/2')
            ax_freq.axvline(-bw / 2, color='k', ls='--', lw=0.8, alpha=0.5)
            ax_freq.set_title('Spectrum (dB),  dashed = +-bw/2')
            ax_freq.set_xlabel('spatial frequency'); ax_freq.set_ylabel('|S| (dB)')
            ax_freq.set_ylim(db_floor, 3); ax_freq.grid(True, alpha=0.3); ax_freq.legend()
            ax_freq.set_xlim(-bw , +bw)

            path = os.path.join(save_dir, 'window_analysis_bw%g_fs%g.png' % (bw, fs))
            savefig(path)
            print('saved %s' % path)


if __name__ == '__main__':
    # characterize the interpolate_signal window functions for the paper
    analyze_window_functions(bw_list=(10.0,), fs_list=(2000.0,))