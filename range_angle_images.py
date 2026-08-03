'''
Range angle imaging, the sonar-style counterpart to the SAR images made in render_images.py.

A SAR image comes from many pulses along an aperture. A range angle image comes from a *single*
pulse: the sensor sits at one place, fans rays out over its field of view, and sorts the returns
into a (range, angle) grid. Range is measured directly from the round-trip delay, and angle comes
from the beam pattern -- the sensor cannot resolve a single scatter's bearing, so each scatter
smears across the angle axis according to how the beam is shaped.

Pipeline:
    1. accumulate_scatters_perspective ray-traces the scene from the sensor point, returning the
       range, energy, and azimuth angle of every scatter. Energies carry the 1/r^2 spreading
       loss of a point source, so near returns outshine far ones within a single image.
    2. Ranges are binned into rows, near range at the bottom of the image, far range at the top.
    3. Each scatter is spread over the angle axis by a Gaussian of the beam width, and every
       scatter in a row is summed into that row's pixels.

Entry points:
    render_range_angle_image(obj_path, pose_path, ...)  -- .obj + pose file -> image on disk
    sar_render_range_angle_image(file_name, poses, ...) -- mesh path + pose tensor -> image tensor
'''

import numpy as np
import torch
from matplotlib import pyplot as plt

from utils import extract_pose_info, get_next_path, savefig
from signal_simulation import load_mesh
from accumulate_scatters import accumulate_scatters_perspective, centered_linspace


# a Gaussian's full width at half maximum is 2*sqrt(2*ln2) standard deviations
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def beam_spread_weights(angle_difference_deg, beam_width_deg):
    '''
    Gaussian beam pattern: how much of a scatter's energy lands at a pixel whose look angle is
    angle_difference_deg away from the scatter.

    beam_width_deg is the full width at half maximum of the pattern, so a scatter sitting half a
    beam width off the pixel angle contributes half its energy there. The pattern peaks at 1
    rather than integrating to 1: a scatter dead on boresight of a pixel deposits its full energy.

    inputs:
        angle_difference_deg (...): pixel look angle minus scatter azimuth, in degrees
        beam_width_deg (float): FWHM of the beam, in degrees
    outputs:
        weights (...): beam pattern value in (0, 1]
    '''
    sigma_deg = beam_width_deg / FWHM_PER_SIGMA
    return torch.exp(-0.5 * (angle_difference_deg / sigma_deg) ** 2)


def bin_range_angle(scatter_range, scatter_energy, scatter_azimuth_deg,
                    range_near, range_far, n_range_bins,
                    angle_bins_deg, beam_width_deg,
                    batch_size=2**16):
    '''
    Sort one pulse worth of scatters into a range angle image.

    Rows are range bins, uniformly spaced over [range_near, range_far); scatters outside that
    window are dropped. Columns are the look angles in angle_bins_deg. A pixel is the sum, over
    every scatter in its range bin, of that scatter's energy weighted by the beam pattern
    evaluated at (pixel angle - scatter azimuth).

    Complex energies (wavelength given) are summed coherently, which is why the accumulation is
    a sum rather than a sum of magnitudes.

    inputs:
        scatter_range (R,): one-way range of each scatter, i.e. sensor to scatter distance
        scatter_energy (R,): energy of each scatter, real or complex
        scatter_azimuth_deg (R,): azimuth of each scatter off boresight, in degrees
        range_near/range_far (float): range extent of the image
        n_range_bins (int): number of rows
        angle_bins_deg (A,): look angle of each column, in degrees
        beam_width_deg (float): FWHM of the beam pattern, in degrees
        batch_size (int): scatters per chunk; the beam weights form an (R, A) matrix, so this
            bounds memory rather than changing the result
    outputs:
        image (n_range_bins, A): near range in the bottom row, far range in the top row
    '''
    device = scatter_range.device
    n_angle_bins = angle_bins_deg.shape[0]
    image = torch.zeros(n_range_bins, n_angle_bins, device=device, dtype=scatter_energy.dtype)

    # rows are uniform in range: bin b covers [near + b*dr, near + (b+1)*dr)
    dr = (range_far - range_near) / n_range_bins
    row = torch.floor((scatter_range - range_near) / dr).long()  # (R,)
    in_window = (row >= 0) & (row < n_range_bins)
    row = row[in_window]
    energy = scatter_energy[in_window]
    azimuth = scatter_azimuth_deg[in_window]

    for start in range(0, row.shape[0], batch_size):
        stop = start + batch_size
        # (chunk, A) beam pattern of every scatter against every look angle
        weights = beam_spread_weights(
            angle_bins_deg.reshape(1, n_angle_bins) - azimuth[start:stop].reshape(-1, 1),
            beam_width_deg,
        )
        image.index_add_(0, row[start:stop], energy[start:stop].reshape(-1, 1) * weights)

    # rows were filled near-to-far; the image wants near range at the bottom
    return image.flip(0)


def sar_render_range_angle_image(
        file_name, poses,

        # beam / field of view
        fov_width_deg = 30.0,
        fov_height_deg = 30.0,
        beam_width_deg = 5.0,
        n_ray_width = 512,
        n_ray_height = 512,

        # image size stuff
        n_range_bins = 128,
        n_angle_bins = 128,
        angle_span_deg = None,
        range_near = None,
        range_far = None,
        region_radius = 1.7,

        # scene / physics
        wavelength = None,
        use_sig_magnitude = True,
        num_bounce = 2,
        second_bounce_batch_size = 2**9,
        mesh_scale = None,
        make_ground = True,
        level_with_ground = True,
        object_x_flip = False,
        object_rotate_xyz = (0.0, 0.0, 0.0),

        # material properties
        obj_raids =    (1.0, 1.0, 100.0, 0.1, 0.9),
        ground_raids = (1.0, 1.0,   1.0, 0.9, 0.1),

        verbose = False,
    ):
    '''
    Render a range angle image of a mesh for each of a set of sensor poses.

    Each pose is one pulse from one sensor position -- there is no aperture and no trajectory,
    unlike sar_render_image in render_images.py.

    inputs:
        file_name (str): path to the .obj to image
        poses (T,4,4): sensor poses in the srn_cars convention; only the position is used, since
            the sensor always looks at the scene origin
        fov_width_deg/fov_height_deg (float): angular extent of the ray fan
        beam_width_deg (float): FWHM of the Gaussian beam pattern that spreads each scatter over
            the angle axis
        n_ray_width/n_ray_height (int): rays across the fan; more rays means less speckle from
            the finite ray sampling
        n_range_bins/n_angle_bins (int): output image size (rows, columns)
        angle_span_deg (float): angular extent of the image; defaults to the fan's fov_width_deg
        range_near/range_far (float): one-way range extent of the image. Defaults bracket the
            scene origin at +/- region_radius along the line of sight
        region_radius (float): radius of the scene region, used for the default range window
        wavelength (float): radar wavelength; when given, scatter energies are complex and the
            angle spreading sums them coherently
        use_sig_magnitude (bool): take the magnitude of the accumulated image. Required for a
            complex image to be displayable; leave True unless you want the raw complex values
        num_bounce (int): ray bounces to simulate
        region/material/mesh arguments: as in render_images.sar_render_image

    outputs:
        images (T, n_range_bins, n_angle_bins): near range at the bottom row, left of boresight
            in the leftmost column
        range_bins (n_range_bins,): one-way range of each row, top row first
        angle_bins_deg (n_angle_bins,): look angle of each column, in degrees
    '''
    device = poses.device

    if angle_span_deg is None:
        angle_span_deg = fov_width_deg

    # load the mesh and hardcode the material properties
    mesh, normals, material_properties = load_mesh( file_name,
                                                    device=device,
                                                    make_ground=make_ground,
                                                    scale=mesh_scale,
                                                    obj_raids = obj_raids,
                                                    ground_raids = ground_raids,
                                                    level_with_ground = level_with_ground,
                                                    x_flip = object_x_flip,
                                                    rotate_xyz = object_rotate_xyz,
                                                )

    # one pulse per pose: (T,1,3) sensor positions
    cam_center = extract_pose_info(poses)[0]      # (T,3)
    trajectory = cam_center.unsqueeze(1)          # (T,1,3)

    # default range window brackets the scene origin along the line of sight
    sensor_distance = torch.linalg.norm(cam_center, dim=-1).mean().item()
    if range_near is None:
        range_near = max(0.0, sensor_distance - region_radius)
    if range_far is None:
        range_far = sensor_distance + region_radius
    assert range_far > range_near, 'range_far (%g) must exceed range_near (%g)' % (range_far, range_near)

    if verbose:
        print('Accumulating scatters...')
    torch.cuda.empty_cache()
    all_ranges, all_energies, all_azimuths, _ = accumulate_scatters_perspective(
        mesh, normals, material_properties, trajectory,
        wavelength     = wavelength,
        fov_width_deg  = fov_width_deg,
        fov_height_deg = fov_height_deg,
        n_ray_width    = n_ray_width,
        n_ray_height   = n_ray_height,
        num_bounce     = num_bounce,
        second_bounce_batch_size = second_bounce_batch_size,
    )
    if verbose:
        print('done.')

    # look angle of each column, and the range of each row once flipped near-to-bottom
    angle_bins_deg = centered_linspace(angle_span_deg, n_angle_bins, device)  # (A,)
    row_edges = torch.linspace(range_near, range_far, n_range_bins + 1, device=device)
    range_bins = ((row_edges[:-1] + row_edges[1:]) / 2).flip(0)  # (n_range_bins,) top row first

    if verbose:
        print('Binning range angle image...')
    images = []
    for t in range(trajectory.shape[0]):
        image = bin_range_angle(
            all_ranges[t][0] / 2,   # (R,) round trip range -> one-way range to the scatter
            all_energies[t][0],     # (R,)
            all_azimuths[t][0],     # (R,)
            range_near, range_far, n_range_bins,
            angle_bins_deg, beam_width_deg,
        )
        images.append(image)
    images = torch.stack(images)  # (T, n_range_bins, A)
    if verbose:
        print('done.')

    if use_sig_magnitude:
        images = images.abs()

    return images, range_bins, angle_bins_deg


def plot_range_angle_image(image, range_bins, angle_bins_deg, path, title=None, db=True, db_floor=-40.0):
    '''
    Save a range angle image with labelled axes.

    inputs:
        image (n_range_bins, A): image as returned by sar_render_range_angle_image
        range_bins (n_range_bins,): row ranges, top row first
        angle_bins_deg (A,): column look angles in degrees
        path (str): where to write the figure
        db (bool): plot in dB relative to the brightest pixel, floored at db_floor
    '''
    image = image.detach().cpu().numpy()
    range_bins = range_bins.detach().cpu().numpy()
    angle_bins_deg = angle_bins_deg.detach().cpu().numpy()

    if db:
        peak = image.max()
        image = 20 * np.log10(np.clip(image / peak, 10 ** (db_floor / 20), None)) if peak > 0 else np.zeros_like(image)
        vmin, vmax = db_floor, 0.0
        label = 'dB'
    else:
        vmin, vmax = 0.0, None
        label = 'amplitude'

    # extent puts near range at the bottom, matching the row order of the image
    extent = [angle_bins_deg[0], angle_bins_deg[-1], range_bins[-1], range_bins[0]]
    plt.figure(figsize=(6, 5))
    plt.imshow(image, cmap='inferno', vmin=vmin, vmax=vmax, extent=extent, aspect='auto', origin='upper')
    plt.colorbar(label=label)
    plt.xlabel('Azimuth angle (deg)')
    plt.ylabel('Range')
    if title is not None:
        plt.title(title)
    savefig(path)
    return path


def render_range_angle_image(obj_path, pose_path, suffix=None, plot=True, device='cuda', **kwargs):
    '''
    Render a range angle image from an .obj file and a pose file, and save it to figures/.

    inputs:
        obj_path (str): path to the .obj mesh
        pose_path (str): path to a pose file holding a 4x4 srn_cars pose matrix
        suffix (str): name for the saved files; a numbered path is used when None
        plot (bool): also save a labelled figure next to the raw .npy
        kwargs: forwarded to sar_render_range_angle_image
    outputs:
        image (n_range_bins, n_angle_bins): the range angle image
        range_bins (n_range_bins,): row ranges, top row first
        angle_bins_deg (n_angle_bins,): column look angles in degrees
    '''
    pose = np.loadtxt(pose_path).reshape(1, 4, 4).astype(np.float32)
    poses = torch.tensor(pose, device=device)  # (1,4,4)

    pose_info = extract_pose_info(poses)
    az, el = pose_info[6].item(), pose_info[5].item()
    print('Pose azimuth (deg):   ', az)
    print('Pose elevation (deg): ', el)

    images, range_bins, angle_bins_deg = sar_render_range_angle_image(obj_path, poses, **kwargs)
    image = images[0]  # (n_range_bins, n_angle_bins)

    if suffix is None:
        npy_path = get_next_path('figures/range_angle_image.npy')
    else:
        npy_path = 'figures/range_angle_image_%s.npy' % suffix
    np.save(npy_path, image.detach().cpu().numpy())
    print('Saved range angle image to: ', npy_path)

    if plot:
        plot_range_angle_image(
            image, range_bins, angle_bins_deg,
            npy_path[:-4] + '.png',
            title='Range angle image\nAz: %.1f, El: %.1f' % (az, el),
        )
        print('Saved range angle plot to:  ', npy_path[:-4] + '.png')

    return image, range_bins, angle_bins_deg
