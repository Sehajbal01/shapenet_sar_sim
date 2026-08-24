import os
import time

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the second to
# initialize aborts with "OMP: Error #15". Allow the duplicate, as paper_figures.py does.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import PIL
from PIL import ImageDraw
import torch

from utils import extract_pose_info
from range_angle_images import beam_spread_weights
from signal_simulation import interpolate_signal, load_mesh
from accumulate_scatters import accumulate_scatters_side_scan, centered_linspace
from imaging_algorithms import side_scan_ground_plane_image
from signal_visualization import signal_gif, signal_column_image
from display_compression import asinh_compress, to_asinh, compute_dataset_reference


def side_scan_sonar_image(
    mean_sensor_position,
    track_length,
    num_pings,
    elevation_fov_deg,
    azimuth_beam_width_deg,
    object_mesh,
    face_normals,
    material_properties,
    num_ray_width, # (azimuth direction)
    num_ray_height, # elevation direction
    region_radius,

    image_width = 64,
    image_height = 64,
    image_plane_width = 1,
    image_plane_height = 1,

    wavelength = None,
    num_bounce = 1,
    second_bounce_batch_size = 2**9,
    spherical_spread = True,
    water_absorption = 0.0,
    tvg_exponent = 4.0,
    spatial_bw = 32,
    spatial_fs = 64,
    window_func = 'sinc',
    use_sig_magnitude = True,
    debug_gif = False,
    debug_columns = False,
    debug_gif_suffix = None,

    # display -- only used by the debug_columns still; the returned images are raw amplitude
    compression = 'db',
    db_floor = -60.0,
    asinh_k_ratio = 0.1,

        ):

    # figure out each sensor position. The track is a straight line through
    # mean_sensor_position, running along the sensor's right vector, so the object stays off
    # to one side of the platform the whole way -- that sidelong look is what makes this a
    # side scan.
    device = mean_sensor_position.device
    line_of_sight   = torch.nn.functional.normalize(-mean_sensor_position, dim=-1)  # (3,) sensor -> origin
    world_up        = torch.tensor([0.0, 0.0, 1.0], device=device)                  # +z
    track_direction = torch.nn.functional.normalize(
        torch.linalg.cross(line_of_sight, world_up), dim=-1)                        # (3,) sensor's right


    ground_forward = torch.nn.functional.normalize(
        -torch.linalg.cross(track_direction, world_up), dim=-1)                     # (3,)



    # calculate trajectory of the sensor
    ping_offsets = centered_linspace(track_length, num_pings, device)               # (P,)
    trajectory   = (mean_sensor_position.reshape(1, 3)
                    + ping_offsets.reshape(num_pings, 1) * track_direction.reshape(1, 3))  # (P,3)

    # figure out the camera matrix for each sensor position. Broadside: every ping shares the
    # single orientation built at the track center, so all the boresights are parallel and only
    # the translation column changes down the track. That is what separates this from the
    # spotlight geometry of range_angle_images, where each pose steers onto the origin and the
    # object therefore never leaves azimuth 0; here the object drifts across the beam as the
    # platform passes, and the azimuth weighting below has something to bite on.
    # Columns are (right, up, forward, center), the srn_cars layout of generate_pose_mat.
    up_vector = torch.linalg.cross(track_direction, line_of_sight)                  # (3,)
    mean_pose = torch.zeros(4, 4, device=device)                                    # (4,4)
    mean_pose[:3, 0] = track_direction
    mean_pose[:3, 1] = up_vector
    mean_pose[:3, 2] = line_of_sight
    mean_pose[:3, 3] = mean_sensor_position
    mean_pose[ 3, 3] = 1.0
    poses = mean_pose.reshape(1, 4, 4).repeat(num_pings, 1, 1)                      # (P,4,4)
    poses[:, :3, 3] = trajectory                                                    # only the center moves

    # the ray fan only needs to cover the angles where the beam pattern still has weight. The
    # Gaussian is down to ~0.002 at 1.5 beam widths off boresight, so 3 beam widths total spends
    # rays where the energy actually is instead of on the tails.
    fan_azimuth_fov_deg   = azimuth_beam_width_deg * 3
    fan_elevation_fov_deg = elevation_fov_deg

    # use accumulate_scatters to ray trace on the object for each camera matrix
    scatter_ranges, scatter_energies, scatter_azimuths, debugging_maps = accumulate_scatters_side_scan(
        object_mesh, face_normals, material_properties,
        poses.unsqueeze(0),                      # (1,P,4,4), one scene
        wavelength     = wavelength,
        fov_width_deg  = fan_azimuth_fov_deg,
        fov_height_deg = fan_elevation_fov_deg,
        n_ray_width    = num_ray_width,
        n_ray_height   = num_ray_height,
        num_bounce     = num_bounce,
        second_bounce_batch_size = second_bounce_batch_size,
        spherical_spread = spherical_spread,
        water_absorption = water_absorption,
        debug_gif      = debug_gif,
    )  # list[T][P] of (R',) each

    # weigh received scatters according to azimuth beam width with gaussian
    scatter_energies = [
        [energy * beam_spread_weights(azimuth, azimuth_beam_width_deg)
         for energy, azimuth in zip(energies_t, azimuths_t)]
        for energies_t, azimuths_t in zip(scatter_energies, scatter_azimuths)
    ]  # list[T][P] of (R',)

    # interpolate signal, on one range window shared by every ping so the columns line up.
    # Centered on the target, so the object sits mid-window. The window can start inside
    # swath_near, which is correct: the object stands off the seafloor and returns before it.
    target_range  = torch.linalg.norm(mean_sensor_position)
    window_center = target_range.reshape(1)  # (1,)
    signals = []
    sample_z = []
    for ranges_t, energies_t in zip(scatter_ranges, scatter_energies):
        signals_t = []
        sample_z_t = []
        for scatter_range, scatter_energy in zip(ranges_t, energies_t):
            signal_p, sample_z_p = interpolate_signal(
                scatter_range.unsqueeze(0) / 2,   # (1,R') round trip -> one-way range
                scatter_energy.unsqueeze(0),      # (1,R')
                region_radius,
                window_center,
                spatial_bw = spatial_bw, spatial_fs = spatial_fs,
                window_func = window_func,
            )
            signals_t.append(signal_p.squeeze(0))      # (Z,)
            sample_z_t.append(sample_z_p.squeeze(0))   # (Z,)
        signals.append(torch.stack(signals_t))         # (P,Z)
        sample_z.append(torch.stack(sample_z_t))       # (P,Z)
    signals  = torch.stack(signals)   # (T,P,Z)
    sample_z = torch.stack(sample_z)  # (T,P,Z)

    # time varying gain: a receiver ramp of R^n against the seafloor's fall with range. Absolute
    # rather than referenced to a range, so it rescales the image as well as tilting it. n=0 turns it off.
    if tvg_exponent:
        print('tvg_exponent: tvg_exponent')
        signals = signals * sample_z ** tvg_exponent  # (T,P,Z)

    # debug outputs, independently switched: debug_columns is one still (fast), debug_gif is a
    # per-ping movie (slow) -- split so the still can be checked without waiting on the movie.
    if debug_gif or debug_columns:
        T = signals.shape[0]
        base_suffix = 'side_scan' if debug_gif_suffix is None else 'side_scan_%s' % debug_gif_suffix

    if debug_columns:
        # the same signals as one still, pings as columns, before the ground plane projection.
        # the depression angle is what lets it put its range axis on the ground, to the same
        # scale as the along-track axis
        # how far the boresight sits above horizontal, which is what scales the still's range axis
        elevation_angle_deg = float(torch.asin(-line_of_sight[2].clamp(-1.0, 1.0))) * 180 / np.pi
        signal_column_image(signals, sample_z, ping_offsets, suffix = base_suffix,
                            depression_deg = elevation_angle_deg,
                            compression = compression, db_floor = db_floor,
                            asinh_k_ratio = asinh_k_ratio)

    # per-ping debug movie: depth map, energy map, range-vs-energy scatter, and the interpolated
    # signal. signal_gif only ever looks at track 0, so feed it one track at a time rather than
    # indexing T away here.
    if debug_gif:
        for t in range(T):
            track_suffix = '' if T == 1 else '_track%02d' % t
            maps_t = {(0, p): debugging_maps[(t, p)] for p in range(num_pings)}
            signal_gif(signals[t:t+1], sample_z[t:t+1], maps_t,
                       [scatter_ranges[t]], [scatter_energies[t]], region_radius,
                       suffix = base_suffix + track_suffix)

    if use_sig_magnitude:
        signals = signals.abs()  # project the envelope; a complex image is not displayable

    # project the pings onto the ground plane, since range and along-track are not ground coordinates
    images, row_coords, col_coords = side_scan_ground_plane_image(
        signals,
        sample_z[:, 0, :],   # (T,Z) every ping shares the one range window
        ping_offsets,
        mean_pose,
        image_width        = image_width,
        image_height       = image_height,
        image_plane_width  = image_plane_width,
        image_plane_height = image_plane_height,
    )  # (T,H,W), (H,), (W,)

    return images, row_coords, col_coords
    #      (T,H,W), (H,) ground forward offset of each row, (W,) ground right offset of each column


def render_side_scan_image(
        obj_id = None,
        pose_num = None,
        suffix = None,
        device = 'cuda',

        override_obj_path = None,
        sensor_distance = None,

        # track geometry
        track_length = 2.0,
        num_pings = 128,
        elevation_fov_deg = 60.0,
        azimuth_beam_width_deg = 1.0,
        num_ray_width = 64,
        num_ray_height = 512,
        region_radius = 0.75,

        # ground plane image geometry
        image_width = 64,
        image_height = 64,
        image_plane_width = 1.5,
        image_plane_height = 1.5,

        # signal / physics
        wavelength = None,
        num_bounce = 1,
        second_bounce_batch_size = 2**9,
        spherical_spread = True,
        water_absorption = 0.0,
        tvg_exponent = 4.0,
        spatial_bw = 64,
        spatial_fs = 64,
        window_func = 'sinc',
        use_sig_magnitude = True,

        # debug
        debug_gif = False,
        debug_columns = False,

        # display
        compression = 'db',
        db_floor = -60.0,
        asinh_k_ratio = 0.1,

        # mesh
        mesh_scale = None,
        make_ground = True,
        level_with_ground = True,
        object_x_flip = False,
        object_rotate_xyz = (90.0, 0.0, 0.0),

        # material properties
        obj_raids =    (1.0, 1.0, 100.0, 0.1, 0.9),
        ground_raids = (1.0, 1.0,   1.0, 0.9, 0.1),
    ):
    '''
    Render a side scan sonar image of one srn_cars object and save it beside the RGB view that
    shares its pose, the same way render_images.render_random_image saves SAR next to RGB.

    The pose file only supplies the sensor *position*: side_scan_sonar_image flies a straight
    track through it and stares off to one side, so the RGB camera and the sonar platform share
    a vantage point but not a look direction. The pings are projected onto a patch of seafloor
    centered on the origin: columns run along the track, rows run out in ground range with near
    range at the bottom.

    inputs:
        obj_id (str): srn_cars object id; a random one is drawn when None
        pose_num (str): pose/rgb file stem for that object; a random one is drawn when None
        suffix (str): name for the saved files, defaults to '<pose_num>_<obj_id>'
        override_obj_path (str): render this .obj instead of the selected object's mesh
        sensor_distance (float): overrides the pose's sensor range from the origin, keeping its
            azimuth and elevation. The sensor position is normalized then scaled to this distance;
            None keeps the pose file's own distance
        track_length (float): along-track extent the platform flies, centered on the pose
            position. Keep it at least image_plane_width: a track shorter than the patch is wide
            leaves the outer image columns with no ping abeam of them, and they come out zero
        num_pings (int): pings along the track, i.e. columns of the image
        elevation_fov_deg (float): vertical extent of the ray fan, about the boresight
        azimuth_beam_width_deg (float): FWHM of the along-track beam, which sets the along-track
            resolution; the ray fan spans 3x this
        region_radius (float): half the range extent imaged, centered on the sensor->origin
            distance so the target sits mid-window. Too small and the patch's near and far
            edges fall outside the window and come out zero
        num_ray_width/num_ray_height (int): rays per ping across the fan
        image_width/image_height (int): pixels across and down the ground patch
        image_plane_width/image_plane_height (float): extent of the ground patch in world units,
            along the track and out in ground range. The patch stays centered on the origin, so
            it needs room for the object's own span plus the shadow it throws down range
        spherical_spread (bool): True applies energy /= 4*pi * range**2 over the round trip;
            False turns the spreading loss off, which is useful for telling how much of the
            near-range dominance is spreading and how much is geometry
        water_absorption (float): two-way absorption in nepers per unit length, 0 = off
        tvg_exponent (float): time varying gain, a receiver ramp of (R/R_ref)^n applied per
            range sample against the seafloor's fall with range, referenced to the window
            center so the target's own level is unchanged. 0 = off
        compression (str): how the composite panel is displayed. 'db' (default here) shows dB
            relative to the brightest pixel, floored at db_floor, as plot_range_angle_image does
            -- a linear stretch is all seafloor and specular glint, since the returns span ~100
            dB. 'linear' is that plain min-max stretch. 'asinh' arcsinh-compresses referenced to
            this image's own 99.9th percentile amplitude (see asinh_compress) -- stays linear
            near zero and logarithmic past asinh_k_ratio * that reference, so seafloor texture
            survives without dB's hard floor
        db_floor (float): black point of the dB display, ignored unless compression == 'db'
        asinh_k_ratio (float): asinh softening scale as a fraction of this image's own reference
            level (k = asinh_k_ratio * ref), ignored unless compression == 'asinh'
        debug_gif (bool): also write a per-ping movie of the depth map, energy map,
            range-vs-energy scatter, and interpolated signal, as sar_render_image does
        debug_columns (bool): also write a still of the interpolated signals with one column
            per ping. Independent of debug_gif -- much faster, since it skips the per-ping movie
        remaining arguments: as in side_scan_sonar_image / render_images.render_random_image

    outputs:
        images (T,H,W): the side scan image(s), near range at the bottom row
        row_coords (H,): ground forward offset of each row, far range first
        col_coords (W,): ground right offset of each column
    '''

    # cluster dirs, same as render_images.render_random_image
    dataset_dir = '/workspace/data/srncars/cars_train/'
    models_dir = '/workspace/data/srncars/02958343'

    if obj_id is None:
        obj_id = np.random.choice(os.listdir(dataset_dir), 1)[0]
    print('Selected object ID: ', obj_id)

    if pose_num is None:
        all_pose_nums = os.listdir(os.path.join(dataset_dir, obj_id, 'pose'))
        pose_num = np.random.choice(all_pose_nums, 1)[0].split('.')[0]
    print('Selected pose number: ', pose_num)

    if suffix is None:
        suffix = '%s_%s' % (pose_num, obj_id)

    # load image, pose, and mesh
    rgb_path  = os.path.join(dataset_dir, obj_id, 'rgb', '%s.png' % pose_num)
    pose_path = os.path.join(dataset_dir, obj_id, 'pose', '%s.txt' % pose_num)
    mesh_path = os.path.join(models_dir, obj_id, 'models', 'model_normalized.obj')
    if override_obj_path is not None:
        print('Overriding object path to %s.' % override_obj_path)
        mesh_path = override_obj_path
    rgb  = np.array(PIL.Image.open(rgb_path))[..., :3]  # (H,W,3)
    pose = np.loadtxt(pose_path).reshape(1, 4, 4).astype(np.float32)
    poses = torch.tensor(pose, device=device)  # (1,4,4)

    pose_info = extract_pose_info(poses)
    az, el = pose_info[6].item(), pose_info[5].item()
    print('Center azimuth (deg):   ', az)
    print('Center elevation (deg): ', el)

    # the track runs along cross(line of sight, +z), which collapses for a nadir look, and the
    # swath geometry needs the platform above the seafloor rather than below it
    assert el < 85.0, 'pose elevation %.1f deg is too close to vertical for a side scan track' % el
    assert el > 0.0,  'pose elevation %.1f deg puts the sensor below the seafloor' % el

    mean_sensor_position = pose_info[0].reshape(3)  # (3,) camera center of the rgb view
    if sensor_distance is not None:
        # normalize then rescale so azimuth/elevation (a ratio of components) survive the change
        mean_sensor_position = torch.nn.functional.normalize(mean_sensor_position, dim=-1) * sensor_distance

    mesh, normals, material_properties = load_mesh( mesh_path,
                                                    device=device,
                                                    make_ground=make_ground,
                                                    scale=mesh_scale,
                                                    obj_raids = obj_raids,
                                                    ground_raids = ground_raids,
                                                    level_with_ground = level_with_ground,
                                                    x_flip = object_x_flip,
                                                    rotate_xyz = object_rotate_xyz,
                                                )

    torch.cuda.empty_cache()
    images, row_coords, col_coords = side_scan_sonar_image(
        mean_sensor_position,
        track_length,
        num_pings,
        elevation_fov_deg,
        azimuth_beam_width_deg,
        mesh, normals, material_properties,
        num_ray_width,
        num_ray_height,
        region_radius,
        image_width = image_width,
        image_height = image_height,
        image_plane_width = image_plane_width,
        image_plane_height = image_plane_height,
        wavelength = wavelength,
        num_bounce = num_bounce,
        second_bounce_batch_size = second_bounce_batch_size,
        spherical_spread = spherical_spread,
        water_absorption = water_absorption,
        tvg_exponent = tvg_exponent,
        spatial_bw = spatial_bw,
        spatial_fs = spatial_fs,
        window_func = window_func,
        use_sig_magnitude = use_sig_magnitude,
        debug_gif = debug_gif,
        debug_columns = debug_columns,
        debug_gif_suffix = suffix,
        compression = compression,
        db_floor = db_floor,
        asinh_k_ratio = asinh_k_ratio,
    )  # (T,H,W), (H,), (W,)

    # one figure per track, so a multi-track run writes one file each instead of dropping all
    # but the first
    T = images.shape[0]
    for t in range(T):
        track_suffix = '' if T == 1 else '_track%02d' % t

        # save raw amplitude so downstream stitching can use a shared color scale
        sonar_amp = images[t].detach().cpu().numpy()  # (H,W)

        # normalize to a displayable 8-bit gray, then widen to 3 channels to sit next to the rgb
        peak = sonar_amp.max()
        ref = float(np.percentile(sonar_amp, 99.9)) if peak > 0 else 0.0
        if compression == 'db' and peak > 0:
            sonar = 20 * np.log10(np.clip(sonar_amp / peak, 10 ** (db_floor / 20), None))
            sonar = ((sonar - db_floor) / -db_floor * 255.0).astype(np.uint8)
        elif compression == 'asinh' and ref > 0:
            # ref is this one image's own 99.9th percentile; compute_dataset_reference exists for
            # a real cross-dataset reference once there's a dataset of saved amplitudes to use
            sonar = to_asinh(sonar_amp, asinh_k_ratio * ref, ref)
        else:
            span = max(float(peak - sonar_amp.min()), 1e-12)  # an all-dark image would divide by 0
            sonar = ((sonar_amp - sonar_amp.min()) / span * 255.0).astype(np.uint8)
        sonar = np.tile(sonar[..., None], (1, 1, 3))  # (H,W,3)
        sonar = cv2.resize(sonar, (rgb.shape[1], rgb.shape[0]))  # (H,W,3)
        image = np.concatenate((rgb, sonar), axis=1)

        # write azimuth and elevation at the top left of the image
        image = PIL.Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), 'Az: %.1f, El: %.1f' % (az, el), fill=(0, 0, 0))

        path = 'figures/side_scan_rgb_image_%s%s.png' % (suffix, track_suffix)
        npy_path = 'figures/side_scan_amp_%s%s.npy' % (suffix, track_suffix)
        image.save(path)
        np.save(npy_path, sonar_amp)
        print('Saved side scan and RGB image to: ', path)

    return images, row_coords, col_coords


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    render_side_scan_image()

