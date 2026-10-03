'''
Forward looking sonar imaging: one perspective ray trace, many beam-steered pings.

The sensor sits at one position with its boresight on the scene origin and fans rays over
ray_fov_az x ray_fov_el. That fan is ray traced once. Each ping then steers the two-way beam to its
own azimuth, spaced evenly over image_azimuth_range: every scatter's tx and rx azimuth are taken
relative to the ping's azimuth and weighted by sidescansonar.apply_beam_pattern, and the weighted
scatters go through interpolate_signal as in every other imaging mode. Each ping's signal is one
column of the image, near range in the bottom row.

The image's azimuth span should sit inside the ray fan: a ping at the edge of the image still needs
rays out to a beam width or so past it, or its column loses part of its beam.

Entry points:
    forward_looking_sonar_image(sensor_positions, ...)         -- mesh tensors -> image tensor
    render_forward_looking_sonar_image(obj_id, pose_num, ...)  -- srn_cars object -> image on disk
'''
import os

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the second to
# initialize aborts with "OMP: Error #15". Allow the duplicate, as paper_figures.py does.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import PIL
from PIL import ImageDraw
import torch

from config import SHAPENET_CARS_DIR, srn_split_dir
from utils import extract_pose_info
from signal_simulation import interpolate_signal, load_mesh
from accumulate_scatters import accumulate_scatters_perspective, centered_linspace
from sidescansonar import apply_beam_pattern
from paper_figure_layout import panel_display


def forward_looking_sonar_image(
    sensor_positions,
    image_azimuth_range,
    num_pings,
    num_range_values,
    ray_fov_az,
    ray_fov_el,
    azimuth_beam_width_deg,
    object_mesh,
    face_normals,
    material_properties,
    num_ray_width, # azimuth direction
    num_ray_height, # elevation direction

    range_near = None,
    range_far = None,
    region_radius = 1.7,

    wavelength = None,
    num_bounce = 1,
    second_bounce_batch_size = 2**9,
    spatial_bw = 32,
    spatial_fs = 64,
    waveform = 'sinc',
    use_sig_magnitude = True,

    # a previously built octree for this mesh, built inside the accumulator when None. Only
    # the mesh decides it, so a caller rendering many poses of one object builds it once
    octree = None,

        ):
    '''
    Forward looking sonar image(s) of a mesh, one per sensor position.

    inputs:
        sensor_positions (T,3): where the sensor sits; its boresight always points at the origin
        image_azimuth_range (float): total azimuth span of the image in degrees, centered on
            boresight. Keep it inside ray_fov_az, with room for the beam at the edges
        num_pings (int): pings across that span, i.e. columns of the image, left to right
        num_range_values (int): rows of the image, far range in the top row
        ray_fov_az/ray_fov_el (float): azimuth and elevation extent of the ray fan, in degrees
        azimuth_beam_width_deg (float): two-way FWHM of the beam each ping steers, in degrees
        num_ray_width/num_ray_height (int): rays across the fan in azimuth and elevation
        range_near/range_far (float): one-way range of the bottom and top rows. None brackets the
            scene origin at +/- region_radius along the line of sight, so moving the sensor moves
            the window with it
        region_radius (float): half the default range window, ignored when both ends are given
        wavelength (float): when given, scatter energies are complex and sum coherently
        remaining arguments: as in sidescansonar.side_scan_sonar_image
    outputs:
        images (T,H,W): the sonar image(s), H = num_range_values, W = num_pings
        row_ranges (H,): one-way range of each row, far range first
        ping_azimuths (W,): steering azimuth of each column in degrees, left to right
    '''
    device = sensor_positions.device
    T = sensor_positions.shape[0]

    # ray trace once: a single perspective view per sensor position, boresight on the origin
    scatter_ranges, scatter_energies, scatter_azimuths, scatter_arrival_azimuths, _ = accumulate_scatters_perspective(
        object_mesh, face_normals, material_properties,
        sensor_positions.reshape(T, 1, 3),       # (T,1,3), one view per sensor position
        wavelength     = wavelength,
        fov_width_deg  = ray_fov_az,
        fov_height_deg = ray_fov_el,
        n_ray_width    = num_ray_width,
        n_ray_height   = num_ray_height,
        num_bounce     = num_bounce,
        second_bounce_batch_size = second_bounce_batch_size,
        octree         = octree,
    )  # list[T][1] of (R',) each

    # default range window brackets the scene origin along the line of sight
    sensor_distance = torch.linalg.norm(sensor_positions, dim=-1).mean().item()
    if range_near is None:
        range_near = max(0.0, sensor_distance - region_radius)
    if range_far is None:
        range_far = sensor_distance + region_radius
    assert range_far > range_near, 'range_far (%g) must exceed range_near (%g)' % (range_far, range_near)

    # one signal window shared by every ping, covering the rows plus a sample of margin
    window_center = torch.tensor([(range_near + range_far) / 2], device=device)  # (1,)
    window_radius = (range_far - range_near) / 2 + 1 / spatial_fs

    # each ping steers the beam to its own azimuth, left to right across the image
    ping_azimuths = centered_linspace(image_azimuth_range, num_pings, device)  # (P,)

    signals = []
    for t in range(T):
        scatter_z = scatter_ranges[t][0] / 2           # (R',) round trip -> one-way range
        energies  = scatter_energies[t][0]             # (R',)
        tx        = scatter_azimuths[t][0]             # (R',)
        rx        = scatter_arrival_azimuths[t][0]     # (R',)
        signals_t = []
        for ping_azimuth in ping_azimuths:
            # ping adjusted tx and rx azimuths through the side scan's two-way beam
            energy_p = apply_beam_pattern(energies, tx - ping_azimuth, rx - ping_azimuth,
                                          azimuth_beam_width_deg)  # (R',)

            # scatters the beam weights to exactly zero add nothing; dropping them bounds memory at high fs
            lit = energy_p != 0  # (R',)
            signal_p, sample_z = interpolate_signal(
                scatter_z[lit].unsqueeze(0),   # (1,R'')
                energy_p[lit].unsqueeze(0),    # (1,R'')
                window_radius,
                window_center,
                spatial_bw = spatial_bw, spatial_fs = spatial_fs,
                waveform = waveform,
            )
            signals_t.append(signal_p.squeeze(0))  # (Z,)
        signals.append(torch.stack(signals_t))     # (P,Z)
    signals  = torch.stack(signals)  # (T,P,Z)
    sample_z = sample_z.squeeze(0)   # (Z,) every ping shares the one window

    if use_sig_magnitude:
        signals = signals.abs()  # project the envelope; a complex image is not displayable

    # resample each ping's signal onto the rows, far range first; the columns are the pings themselves
    row_ranges = torch.linspace(range_far, range_near, num_range_values, device=device, dtype=sample_z.dtype)  # (H,)
    y_norm = (2 * (row_ranges - sample_z[0]) / (sample_z[-1] - sample_z[0]) - 1
              ).reshape(1, num_range_values, 1).expand(T, num_range_values, num_pings)  # (T,H,W)
    x_norm = torch.linspace(-1.0, 1.0, num_pings, device=device, dtype=sample_z.dtype
              ).reshape(1, 1, num_pings).expand(T, num_range_values, num_pings)         # (T,H,W)
    sample_grid = torch.stack((x_norm, y_norm), dim=-1)  # (T,H,W,2)

    grid = signals.transpose(1, 2).unsqueeze(1)  # (T,1,Z,P)
    if torch.is_complex(grid):
        images = torch.complex(
            torch.nn.functional.grid_sample(grid.real, sample_grid, align_corners=True, padding_mode='zeros'),
            torch.nn.functional.grid_sample(grid.imag, sample_grid, align_corners=True, padding_mode='zeros'),
        ).squeeze(1)  # (T,H,W)
    else:
        images = torch.nn.functional.grid_sample(
            grid, sample_grid, align_corners=True, padding_mode='zeros'
        ).squeeze(1)  # (T,H,W)

    return images, row_ranges, ping_azimuths
    #      (T,H,W), (H,) one-way range of each row (far to near), (W,) azimuth of each column (left to right)


def render_forward_looking_sonar_image(
        obj_id = None,
        pose_num = None,
        suffix = None,
        device = 'cuda',

        override_obj_path = None,
        sensor_distance = None,
        elevation_angle_deg = None,

        # image geometry
        image_azimuth_range = 50.0,
        num_pings = 128,
        num_range_values = 128,
        range_near = None,
        range_far = None,
        region_radius = 1.7,

        # ray fan and beam
        ray_fov_az = 60.0,
        ray_fov_el = 100.0,
        azimuth_beam_width_deg = 1.0,
        num_ray_width = 1024,
        num_ray_height = 1024,

        # signal / physics
        wavelength = None,
        num_bounce = 1,
        second_bounce_batch_size = 2**9,
        spatial_bw = 32,
        spatial_fs = 64,
        waveform = 'sinc',
        use_sig_magnitude = True,

        # display
        compression = 'db',
        db_floor = -40.0,
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
    Render a forward looking sonar image of one srn_cars object and save it beside the RGB view
    that shares its pose, as sidescansonar.render_side_scan_image does.

    The sonar sits at the pose's camera center and, like the camera, looks at the scene origin, so
    the two share a vantage point and a look direction. Columns run left to right in azimuth and
    rows run out in range, near range at the bottom.

    inputs:
        obj_id (str): srn_cars object id; a random one is drawn when None
        pose_num (str): pose/rgb file stem for that object; a random one is drawn when None
        suffix (str): name for the saved files, defaults to '<pose_num>_<obj_id>'
        override_obj_path (str): render this .obj instead of the selected object's mesh
        sensor_distance (float): overrides the pose's sensor range from the origin, keeping its
            azimuth and elevation, to move the sensor back. Null range_near/range_far along with
            it so the range window follows, and narrow the fovs to the smaller angle the car
            subtends. None keeps the pose file's own distance
        elevation_angle_deg (float): overrides the pose's elevation, keeping its azimuth and
            distance, so a sweep can reach elevations no pose file has: 0 puts the sensor on the
            seafloor, 90 straight overhead. The rgb beside the sonar stays the pose file's view.
            None keeps the pose file's own elevation
        compression (str): how the saved png is displayed, 'db' | 'linear' | 'asinh', as
            paper_figure_layout.panel_display does it for the stitched figures
        db_floor/asinh_k_ratio (float): that display's dB floor and asinh softening ratio
        remaining arguments: as in forward_looking_sonar_image / render_side_scan_image

    outputs:
        images (T,H,W): the forward looking sonar image(s), near range at the bottom row
        row_ranges (H,): one-way range of each row, far range first
        ping_azimuths (W,): steering azimuth of each column in degrees, left to right
    '''

    # dataset locations from config.json, same as render_side_scan_image
    dataset_dir = srn_split_dir('cars_train')
    models_dir = SHAPENET_CARS_DIR

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
    if elevation_angle_deg is not None:
        el = float(elevation_angle_deg)
    print('Center azimuth (deg):   ', az)
    print('Center elevation (deg): ', el)

    # a sensor below the seafloor would image the ground plane from underneath
    assert el >= 0.0, 'elevation %.1f deg puts the sensor below the seafloor' % el

    sensor_position = pose_info[0].reshape(1, 3)  # (1,3) camera center of the rgb view
    if elevation_angle_deg is not None:
        # the pose's azimuth and distance at this elevation, in float64 so cos(90 deg) keeps the azimuth's sign
        x, y, z = sensor_position[0].tolist()
        az_rad, el_rad = np.arctan2(y, x), np.radians(el)
        sensor_position = torch.tensor(np.sqrt(x * x + y * y + z * z) * np.array(
            [[np.cos(el_rad) * np.cos(az_rad), np.cos(el_rad) * np.sin(az_rad), np.sin(el_rad)]]),
            dtype=torch.float32, device=device)  # (1,3)
    if sensor_distance is not None:
        # normalize then rescale so azimuth/elevation (a ratio of components) survive the change
        sensor_position = torch.nn.functional.normalize(sensor_position, dim=-1) * sensor_distance

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
    images, row_ranges, ping_azimuths = forward_looking_sonar_image(
        sensor_position,
        image_azimuth_range,
        num_pings,
        num_range_values,
        ray_fov_az,
        ray_fov_el,
        azimuth_beam_width_deg,
        mesh, normals, material_properties,
        num_ray_width,
        num_ray_height,
        range_near = range_near,
        range_far = range_far,
        region_radius = region_radius,
        wavelength = wavelength,
        num_bounce = num_bounce,
        second_bounce_batch_size = second_bounce_batch_size,
        spatial_bw = spatial_bw,
        spatial_fs = spatial_fs,
        waveform = waveform,
        use_sig_magnitude = use_sig_magnitude,
    )  # (T,H,W), (H,), (W,)

    # one set of files per sensor position, so a multi-position run keeps every one
    T = images.shape[0]
    for t in range(T):
        view_suffix = '' if T == 1 else '_view%02d' % t

        # raw amplitude and both axes, so the stitched figures share one display and label their axes
        fls_amp = images[t].detach().cpu().numpy()  # (H,W)
        npz_path = 'figures/fls_amp_%s%s.npz' % (suffix, view_suffix)
        np.savez(npz_path,
                 image=fls_amp,
                 row_ranges=row_ranges.detach().cpu().numpy(),
                 ping_azimuths=ping_azimuths.detach().cpu().numpy())

        # the stitched figures' display, stretched to 8-bit gray and widened to sit beside the rgb
        panel, vmin, vmax, _, _ = panel_display(np.abs(fls_amp), compression=compression,
                                                db_floor=db_floor, asinh_k_ratio=asinh_k_ratio)
        sonar = (np.clip((panel - vmin) / (vmax - vmin), 0.0, 1.0) * 255.0).astype(np.uint8)
        sonar = np.tile(sonar[..., None], (1, 1, 3))  # (H,W,3)
        sonar = cv2.resize(sonar, (rgb.shape[1], rgb.shape[0]))  # (H,W,3)
        image = np.concatenate((rgb, sonar), axis=1)

        # write azimuth and elevation at the top left of the image
        image = PIL.Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), 'Az: %.1f, El: %.1f' % (az, el), fill=(0, 0, 0))

        path = 'figures/fls_rgb_image_%s%s.png' % (suffix, view_suffix)
        image.save(path)
        print('Saved forward looking sonar and RGB image to: ', path)

    return images, row_ranges, ping_azimuths


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    render_forward_looking_sonar_image()
