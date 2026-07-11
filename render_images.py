import sys
import tqdm
import PIL
from PIL import ImageDraw
import imageio
import cv2
import os
from utils import get_next_path, generate_pose_mat, savefig, extract_pose_info, plot_image
import torch
import numpy as np
from matplotlib import pyplot as plt

from signal_simulation import interpolate_signal, apply_snr, load_mesh, generate_trajectory, accumulate_scatters

from imaging_algorithms import projected_CBP, strip_map_imaging

from signal_visualization import signal_gif



def sar_render_image(   file_name, num_pulses, poses, az_spread,
                        spatial_bw = 64,
                        spatial_fs = 64,
                        debug_gif = False,
                        debug_gif_suffix = None,
                        snr_db = None,
                        wavelength = None,
                        use_sig_magnitude = True,
                        verbose = False,
                        imaging_algorithm = 'cbp',
                        cbp_batch_size = None,
                        trajectory_type = 'circular',
                        trajectory_noise_var = 0,
                        mesh_scale = None,
                        num_bounce = 2,
                        object_x_flip = False,
                        object_rotate_xyz = (0.0, 0.0, 0.0),

                        override_obj_path = None,

                        # image size stuff
                        image_width = 64,
                        image_height = 64,
                        image_plane_width = 1,
                        image_plane_height = 1,
                        grid_width = 1,
                        grid_height = 1,
                        n_ray_width = 1,
                        n_ray_height = 1,
                        # range_near = 1,
                        # range_far = 1,
                        region_radius = 1.7,

                        # material properties
                        obj_raids =    (1.0, 1.0, 100.0, 0.1, 0.9),
                        ground_raids = (1.0, 1.0,   1.0, 0.9, 0.1),
    ):

    # set device
    device = poses.device

    # load the mesh and hardcode the material properties
    mesh, normals, material_properties = load_mesh( file_name,
                                                    device=device,
                                                    make_ground=True,
                                                    scale=mesh_scale,
                                                    obj_raids = obj_raids,
                                                    ground_raids = ground_raids,
                                                    x_flip = object_x_flip,
                                                    rotate_xyz = object_rotate_xyz,
                                                )

    # generate the sensor trajectory for each pose
    # (T,P,3)        (T,P,3)              (T,P)
    true_trajectory, perceived_trajectory, cam_azimuth_deg = generate_trajectory(   poses,
                                                        trajectory_type=trajectory_type,
                                                        n_pulses=num_pulses,
                                                        azimuth_spread_deg=az_spread,
                                                        trajectory_noise_var = trajectory_noise_var
                                                    )

    # SAR raytracing / rasterization
    if verbose:
        print('Accumulating scatters...')

    # (T,P,R)   (T,P,R)
    torch.cuda.empty_cache()
    all_ranges, all_energies, debugging_maps = accumulate_scatters(
        mesh, normals, material_properties, true_trajectory,
        wavelength     = wavelength,
        debug_gif      = debug_gif,

        # image size stuff
        grid_width     = grid_width,
        grid_height    = grid_height,
        n_ray_width    = n_ray_width,
        n_ray_height   = n_ray_height,

        num_bounce = num_bounce,
        second_bounce_batch_size = 2**10,
    )
    if verbose:
        print('done.')

    # Generate signal
    # (T,P,Z) (T,P,Z)
    if verbose:
        print('Interpolating signal...')
    T, P = true_trajectory.shape[:2]
    signals_list  = []
    sample_z_list = []
    for t in range(T):
        signals_list.append([])
        sample_z_list.append([])
        for p in range(P):
            sig_tp, sz_tp = interpolate_signal(
                all_ranges[t][p].unsqueeze(0) / 2,   # (1, R') spatial range
                all_energies[t][p].unsqueeze(0),      # (1, R')
                region_radius,
                torch.linalg.norm(true_trajectory[t, p], dim=-1).reshape(1),  # (1,)
                spatial_bw = spatial_bw, spatial_fs = spatial_fs,
                batch_size = None,
            )
            signals_list[t].append(sig_tp.squeeze(0))    # (Z,)
            sample_z_list[t].append(sz_tp.squeeze(0))    # (Z,)
    signals  = torch.stack([torch.stack(row) for row in signals_list])   # (T,P,Z)
    sample_z = torch.stack([torch.stack(row) for row in sample_z_list])  # (T,P,Z)
    if verbose:
        print('done.')

    # apply SNR
    if snr_db is not None:
        T,P,Z = signals.shape
        signals = apply_snr(signals.reshape(T,P*Z), snr_db).reshape(T,P,Z)

    # convert to signal magnitude if desired
    complex_signals = signals
    if use_sig_magnitude:
        signals = signals.abs()

    # Compute sar image
    if verbose:
        print('Computing SAR image...')
    if imaging_algorithm == 'cbp':
        sar_image = projected_CBP(
            signals,
            sample_z,
            perceived_trajectory,
            spatial_fs,
            image_plane_rotation_deg = cam_azimuth_deg+90,
            image_width = image_width,
            image_height = image_height,
            image_plane_width = image_plane_width,
            image_plane_height = image_plane_height,
            batch_size = cbp_batch_size,
            coherent_integration = not use_sig_magnitude,
            wavelength = wavelength,
        )
    elif imaging_algorithm == 'stripmap':
        sar_image = strip_map_imaging(
            complex_signals,
            wavelength,
            perceived_trajectory,
            sample_z,
            spatial_fs,
            planar_wave = False,
            image_plane_rotation_deg = cam_azimuth_deg+90,
            image_width = image_width,
            image_height = image_height,
            image_plane_width = image_plane_width,
            image_plane_height = image_plane_height,
        )
    else:
        raise ValueError('Invalid imaging algorithm \'%s\', expected \'cbp\' or \'stripmap\''%imaging_algorithm)
    if verbose:
        print('done.')

    # # save the sar image with colorbar for qualitative analysis
    # plot_image(sar_image, title="SAR", cmap='inferno', db=True)
    # savefig(get_next_path("figures/colorbar_sar_image.png"))

    # make a gif if desired
    if debug_gif:
        signal_gif(signals, sample_z, debugging_maps, all_ranges, all_energies, region_radius, suffix=debug_gif_suffix)

    return sar_image




def render_random_image(
        debug_gif = False, 
        num_pulse = 120,
        azimuth_spread = 180,
        spatial_fs = 64, 
        spatial_bw = 64,
        snr_db = None,
        wavelength = None,
        use_sig_magnitude=True,
        suffix = None,
        imaging_algorithm = 'cbp',
        cbp_batch_size = None,
        trajectory_type = 'circular',
        trajectory_noise_var = 0,
        num_bounce = 2,
        object_x_flip = False,
        object_rotate_xyz = (0.0, 0.0, 0.0),

        override_obj_path = None,

        image_plane_width = 1,
        image_plane_height = 1,
        image_width = 64,
        image_height = 64,
        grid_width = 1,
        grid_height = 1,
        n_ray_width = 1,
        n_ray_height = 1,
        # range_near = 1,
        # range_far = 1,
        region_radius = 1.7,

        # material properties
        obj_raids =    (1.0, 1.0, 100.0, 0.1, 0.9),
        ground_raids = (1.0, 1.0,   1.0, 0.9, 0.1),

        log_scale = False,
    ):
    """
    Renders a random image from the ShapeNet dataset using SAR simulation.
    """

    # cluster dirs
    dataset_dir = '/workspace/data/srncars/cars_train/'
    models_dir = '/workspace/data/srncars/02958343'

    # # lab pc dirs
    # dataset_dir = '/home/berian/Documents/shapenet/cars_train/'
    # models_dir  = '/home/berian/Documents/shapenet/object-models/02958343/'

    all_obj_id = os.listdir(dataset_dir)  # list all object IDs in the dataset
    obj_id     = np.random.choice(all_obj_id, 1)[0]  # randomly select an object ID from the dataset
    print('Selected object ID: ', obj_id)

    all_pose_paths = os.path.join(dataset_dir,obj_id,'pose')
    all_pose_nums  = os.listdir(all_pose_paths)
    pose_num       = np.random.choice(all_pose_nums, 1)[0].split('.')[0]
    print('Selected pose number: ', pose_num)

    if suffix is None:
        suffix = '%s_%s'%(pose_num, obj_id)

    # load image, pose, and mesh
    # rgb_path  = '/workspace/data/srncars/cars_train/%s/rgb/%s.png' % (obj_id, pose_num)
    # pose_path = '/workspace/data/srncars/cars_train/%s/pose/%s.txt' % (obj_id, pose_num)
    # mesh_path = '/workspace/data/srncars/02958343/%s/models/model_normalized.obj' % obj_id
    rgb_path  = os.path.join(dataset_dir, obj_id, 'rgb', '%s.png'%pose_num)
    pose_path = os.path.join(dataset_dir, obj_id, 'pose', '%s.txt'%pose_num)
    mesh_path = os.path.join(models_dir, obj_id, 'models', 'model_normalized.obj')
    rgb  = np.array(PIL.Image.open(rgb_path))[...,:3] # (H, W, 3)
    pose = np.loadtxt(pose_path).reshape(1,4,4).astype(np.float32)  # (4, 4)

    # print the center azimuth and elevation for the selected pose
    pose_info = extract_pose_info(torch.tensor(pose))
    center_az, center_el = pose_info[6].item(), pose_info[5].item()
    print('Center azimuth (deg):   ', center_az)
    print('Center elevation (deg): ', center_el)

    # get azimuth, elevation, and distance from the pose
    target_poses = torch.tensor(pose, device='cuda') # (1, 4, 4)
    # target_poses = generate_pose_mat(0,90,1.3, device='cuda').reshape(1,4,4)  # (1, 4, 4)
    
    # render the SAR images for each pose
    sar = sar_render_image( mesh_path, # fname
                            num_pulse, # num_pulses
                            target_poses, # poses
                            azimuth_spread, # azimuth spread

                            spatial_bw = spatial_bw,
                            spatial_fs = spatial_fs,
                            snr_db = snr_db,
                            wavelength=wavelength,
                            use_sig_magnitude=use_sig_magnitude,

                            override_obj_path = override_obj_path,

                            debug_gif=debug_gif, # debug gif
                            debug_gif_suffix = suffix,

                            imaging_algorithm=imaging_algorithm,
                            cbp_batch_size = cbp_batch_size,
                            trajectory_type=trajectory_type,
                            trajectory_noise_var = trajectory_noise_var,
                            num_bounce = num_bounce,
                            object_x_flip = object_x_flip,
                            object_rotate_xyz = object_rotate_xyz,

                            # image size stuff
                            image_width = image_width,
                            image_height = image_height,
                            image_plane_width = image_plane_width,
                            image_plane_height = image_plane_height,
                            grid_width = grid_width,
                            grid_height = grid_height,
                            n_ray_width = n_ray_width,
                            n_ray_height = n_ray_height,
                            # range_near = range_near,
                            # range_far = range_far,
                            region_radius = region_radius,

                            obj_raids = obj_raids,
                            ground_raids = ground_raids,
    ) # (1,H,W)

    # save raw SAR amplitude so downstream stitching can use a shared color scale
    sar_amp = sar.squeeze(0).cpu().numpy()  # (H,W) raw amplitude
    if log_scale:
        sar_amp = np.log1p(sar_amp)

    # plot the SAR image next to the RGB image
    sar = torch.tile(sar, (3,1,1)).permute(1,2,0)  # (H,W,3)
    if log_scale:
        sar = torch.log1p(sar)
    sar = (sar - sar.min()) / (sar.max() - sar.min()) * 255.0  # normalize to [0, 255]
    sar = sar.cpu().numpy().astype(np.uint8)  # convert to uint8
    sar = cv2.resize(sar, (rgb.shape[1], rgb.shape[0]))  # (H,W,3)
    image = np.concatenate((rgb, sar), axis=1)  # concatenate RGB and SAR images

    # write azimuth and elevation at thee top left of the image
    image = PIL.Image.fromarray(image)  # convert to PIL image
    pose_info = extract_pose_info(torch.tensor(pose))  # extract pose info
    az, el = pose_info[6].item(), pose_info[5].item()
    draw_str = 'Az: %.1f, El: %.1f' % (az, el)
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), draw_str, fill=(0, 0, 0))

    if suffix is None:
        path = get_next_path('figures/sar_rgb_image.png')
        npy_path = path[:-4] + '.npy'
    else:
        path = 'figures/sar_rgb_image_%s.png'%(suffix)
        npy_path = 'figures/sar_amp_%s.npy'%(suffix)
    image.save(path)  # save the image
    np.save(npy_path, sar_amp)
    print('Saved SAR and RGB image to: ', path)




def _prepare_stitched_plot_arrays(sar_arrays, plot_db_scale=False, db_floor=-60.0):
    """Convert raw SAR amplitudes into the plotting data for stitched figures."""
    if not plot_db_scale:
        return sar_arrays, dict(vmin=0.0, vmax=float(max(a.max() for a in sar_arrays)))

    reference = float(max(a.max() for a in sar_arrays))
    if reference <= 0.0:
        plot_arrays = [np.zeros_like(a, dtype=np.float32) for a in sar_arrays]
        return plot_arrays, dict(vmin=db_floor, vmax=0.0)

    plot_arrays = []
    for arr in sar_arrays:
        amplitude = np.asarray(arr, dtype=np.float32)
        db_values = 20.0 * np.log10(np.clip(amplitude / reference, 1e-12, None))
        plot_arrays.append(db_values)
    return plot_arrays, dict(vmin=db_floor, vmax=0.0)


def multi_param_experiment(param_dict, default_kwargs, experiment_name="experiment", seed=8134, custom_title_strings = None, plot_db_scale=False, db_floor=-60.0):
    """
    A modular function to run experiments by varying multiple parameters together
    
    Args:
        param_dict (dict): Dictionary where each key is a parameter name and value is a list/array of values.
                          All lists/arrays must have the same length.
        default_kwargs (dict): Default arguments for render_random_image
        experiment_name (str): Name of the experiment for saving files
        seed (int): Random seed for reproducibility
        plot_db_scale (bool): If True, render the stitched plots in dB relative to a shared max.
    """
    # Verify all parameter arrays have the same length
    lengths = [len(vals) for vals in param_dict.values()]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError("All parameter arrays must have the same length")
    n_experiments = lengths[0]
    
    # Create parameter names string for labeling
    param_names = "_".join(param_dict.keys())

    # remove all files in figures with the experiment name
    for f in os.listdir('figures'):
        if experiment_name in f and (f.endswith('.png') or f.endswith('.npy')):
            os.remove(os.path.join('figures', f))

    # create strings to title each experiment
    if custom_title_strings is None:
        experiment_strings = []
        for i in range(n_experiments):
            param_str_parts = []
            for param_name, param_vals in param_dict.items():
                try:
                    try:
                        val = float(param_vals[i])
                        if val < 0.1:
                            param_str_parts.append("%s%.2e" % (param_name, val))
                        else:
                            param_str_parts.append("%s%.2f" % (param_name, val))
                    except(ValueError):
                        param_str_parts.append(f"{param_name}{param_vals[i]}")

                except(TypeError):
                    param_str_parts.append(f"{param_name}{param_vals[i]}")
            experiment_strings.append('_'.join(param_str_parts))
    else:
        experiment_strings = custom_title_strings

    # generate the images for each parameter combination
    for i in range(n_experiments):
        # set the random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # update the kwargs with the current parameter values
        kwargs = default_kwargs.copy()
        for param_name, param_vals in param_dict.items():
            kwargs[param_name] = param_vals[i]

        # Add a numeric ID to ensure correct sorting
        kwargs['suffix'] = f"{experiment_name}_{i:03d}_{experiment_strings[i]}"
        
        # render the image with the current parameters
        render_random_image(**kwargs)

    # find all raw SAR amplitude arrays saved for this experiment
    npy_files = [f for f in os.listdir('figures') if f'sar_amp_{experiment_name}' in f and f.endswith('.npy')]

    # Extract the figure ID (the 3-digit number after experiment_name_)
    npy_ids = [int(f.split(experiment_name + '_')[1][:3]) for f in npy_files]

    # Sort files by the figure ID
    sorted_npy = [f for _, f in sorted(zip(npy_ids, npy_files))]

    # Load raw SAR amplitude arrays; use a shared color scale so brightness is comparable
    sar_arrays = [np.load(os.path.join('figures', f)) for f in sorted_npy]
    plot_arrays, plot_range = _prepare_stitched_plot_arrays(
        sar_arrays,
        plot_db_scale=plot_db_scale,
        db_floor=db_floor,
    )
    vmin = plot_range['vmin']
    vmax = plot_range['vmax']

    # Layout: if even and > 4, use 2 rows; else 1 row
    n_image = len(sar_arrays)
    if n_image % 2 == 0 and n_image > 4:
        n_rows, n_cols = 2, n_image // 2
    else:
        n_rows, n_cols = 1, n_image

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.2 * n_rows + 0.5),
        squeeze=False,
    )
    for i, (ax, arr, title) in enumerate(zip(axes.flat, plot_arrays, experiment_strings)):
        im = ax.imshow(arr, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    # hide any unused axes (defensive; shouldn't happen given layout math)
    for ax in axes.flat[n_image:]:
        ax.axis('off')

    fig.subplots_adjust(right=0.9, wspace=0.05, hspace=0.15)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    if plot_db_scale:
        db_ticks = [0, -3, -6, -10, -20, -30]
        if db_floor < -30:
            db_ticks = [db_floor] + [t for t in db_ticks if t > db_floor]
        cbar.set_ticks(db_ticks)
        cbar.set_ticklabels([f'{db} dB' for db in db_ticks])
        cbar.set_label('SAR amplitude (dB re shared max)')
    else:
        # Image data is linear amplitude, so show a simple linear colorbar.
        cbar.set_label('SAR amplitude (linear)')

    path = f'figures/sar_stitched_{experiment_name}.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved stitched image to: %s' % path)

