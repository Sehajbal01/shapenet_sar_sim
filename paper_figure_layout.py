'''
One stitched-sweep layout, shared by all three paper figure suites: render_images'
multi_param_experiment (SAR), paper_figures_range_angle's multi_param_range_angle_experiment, and
sonar_paper_figures' multi_param_sonar_experiment. Each of the three grew its own copy of the same
matplotlib code, so the layout now lives here and they all call stitch_panels.

Every panel carries its own horizontal colorbar directly underneath it and its own short
description directly above it, with white space between panels. Per-panel colorbars are the point:
a sweep can move the absolute level by orders of magnitude, so each panel is stretched over its own
range for contrast while its colorbar ticks report the level in units that stay comparable across
the figure.

Axes are placed by hand in inches rather than through subplots/gridspec, so a panel box is exactly
panel_width x panel_height and its colorbar is exactly as wide as the image above it.
'''
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker


# every other measurement is quoted against the panel box, in inches
_TITLE_HEIGHT       = 0.34  # room above a panel for its description
_CBAR_HEIGHT        = 0.11  # thickness of the horizontal colorbar
_CBAR_GAP           = 0.10  # white space between a panel and its colorbar
_CBAR_LABEL_HEIGHT  = 0.44  # room under the colorbar for its tick labels and its label
_AXIS_HEIGHT        = 0.42  # room under a panel for x tick labels and x label, when show_axes
_AXIS_WIDTH         = 0.46  # room left of the first panel for y tick labels and y label
_MARGIN             = 0.06


def _per_panel(value, n, name):
    '''One entry per panel: a scalar, string or None is repeated, a sequence is length checked.'''
    if value is None or isinstance(value, str) or np.isscalar(value):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError('%s has %d entries but there are %d panels' % (name, len(value), n))
    return value


def _tick_formatter(fmt):
    '''Colorbar tick labels from either a %-format string or a value -> string callable.'''
    if callable(fmt):
        return ticker.FuncFormatter(lambda v, pos: fmt(v))
    return ticker.FormatStrFormatter(fmt)


def stitch_panels(
        panels, titles, path,
        cmap='gray',
        vmin=None, vmax=None, min_span=None,
        cbar_label=None,
        cbar_tick_fmt='%.3g',
        cbar_nticks=4,
        extents=None,
        xlabel=None, ylabel=None,
        show_axes=False,
        panel_width=2.2, panel_height=2.2,
        gap=0.55,
        title_fontsize=9,
        dpi=200,
):
    '''
    Stitch one sweep's panels into a single row, each with its own colorbar and description.

    inputs:
        panels (list of (H,W)): panel images, already in display units (raw amplitude, dB, or a
            compressed scale) -- this function only lays them out, it does not compress them
        titles (list of str): the short description drawn above each panel, one per panel
        path (str): where the figure is written
        cmap (str): colormap for every panel
        vmin, vmax (float | list | None): color limits. A scalar applies to every panel, a list
            gives one per panel, and None autoscales that end to each panel's own data -- which is
            the usual choice, since the per-panel colorbar is what reports the level
        min_span (float | None): smallest color range a panel is allowed, in the panels' own units.
            A panel that is entirely at the dB floor would otherwise autoscale to a single value
            and print the same number at every colorbar tick
        cbar_label (str | list | None): label under each colorbar. A list gives one per panel, so a
            sweep can name the value it varied (a panel's own peak, its own asinh k, ...)
        cbar_tick_fmt (str | callable | list): %-format string or value -> str callable for the
            colorbar tick labels, e.g. '%.0f dB'. A list gives one per panel
        cbar_nticks (int): approximate number of colorbar ticks; horizontal bars are narrow, so
            this stays small
        extents (list | None): per-panel imshow extent, for panels plotted against real axes
        xlabel, ylabel (str | None): axis labels, drawn only when show_axes
        show_axes (bool): draw tick marks and axis labels. False turns the axes off entirely, for
            panels whose pixel indices mean nothing
        panel_width, panel_height (float): the panel box, in inches
        gap (float): white space between panels, in inches
        title_fontsize (float): point size of the per-panel description
        dpi (int): resolution of the saved figure
    outputs:
        path (str): the figure written
    '''
    n = len(panels)
    if n == 0:
        raise ValueError('nothing to stitch')
    titles = _per_panel(titles, n, 'titles')
    vmins = _per_panel(vmin, n, 'vmin')
    vmaxs = _per_panel(vmax, n, 'vmax')
    labels = _per_panel(cbar_label, n, 'cbar_label')
    fmts = [cbar_tick_fmt] * n if (callable(cbar_tick_fmt) or isinstance(cbar_tick_fmt, str)) \
        else _per_panel(cbar_tick_fmt, n, 'cbar_tick_fmt')
    extents = [None] * n if extents is None else list(extents)

    axis_h = _AXIS_HEIGHT if show_axes else 0.0
    axis_w = _AXIS_WIDTH if show_axes else 0.0

    fig_w = 2 * _MARGIN + axis_w + n * panel_width + (n - 1) * gap
    fig_h = (2 * _MARGIN + _TITLE_HEIGHT + panel_height + axis_h
             + _CBAR_GAP + _CBAR_HEIGHT + _CBAR_LABEL_HEIGHT)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # bottom edges, as figure fractions: colorbar sits under the panel, with room for the panel's
    # own tick labels between them when the axes are shown
    cbar_bottom = (_MARGIN + _CBAR_LABEL_HEIGHT) / fig_h
    panel_bottom = (_MARGIN + _CBAR_LABEL_HEIGHT + _CBAR_HEIGHT + _CBAR_GAP + axis_h) / fig_h

    for i, panel in enumerate(panels):
        panel = np.asarray(panel, dtype=np.float32)
        left = (_MARGIN + axis_w + i * (panel_width + gap)) / fig_w

        ax = fig.add_axes([left, panel_bottom, panel_width / fig_w, panel_height / fig_h])

        lo = float(np.nanmin(panel)) if vmins[i] is None else float(vmins[i])
        hi = float(np.nanmax(panel)) if vmaxs[i] is None else float(vmaxs[i])
        if min_span is not None:
            hi = max(hi, lo + min_span)
        if not hi > lo:  # a flat panel still needs a colorbar with two distinct ends
            hi = lo + max(abs(lo), 1.0) * 1e-3

        im = ax.imshow(panel, cmap=cmap, vmin=lo, vmax=hi, extent=extents[i],
                       aspect='auto', origin='upper')
        ax.set_title(titles[i], fontsize=title_fontsize, pad=4)

        if show_axes:
            ax.tick_params(labelsize=6)
            if xlabel is not None:
                ax.set_xlabel(xlabel, fontsize=7, labelpad=1)
            if i == 0 and ylabel is not None:
                ax.set_ylabel(ylabel, fontsize=7, labelpad=1)
            if i > 0:
                ax.set_yticklabels([])
        else:
            ax.axis('off')

        cax = fig.add_axes([left, cbar_bottom, panel_width / fig_w, _CBAR_HEIGHT / fig_h])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.locator = ticker.MaxNLocator(nbins=cbar_nticks)
        cbar.formatter = _tick_formatter(fmts[i])
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=6, pad=1)
        if labels[i] is not None:
            cbar.set_label(labels[i], fontsize=7, labelpad=1)

    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('Saved stitched image to: %s' % path)
    return path
