import sys
import tqdm
import PIL
from PIL import ImageDraw
import imageio
import cv2
import os
from utils import get_next_path, generate_pose_mat, savefig, extract_pose_info
import torch
import numpy as np
from matplotlib import pyplot as plt

from signal_simulation import interpolate_signal, apply_snr, load_mesh

from signal_simulation import accumulate_scatters as accumulate_scatters_rasterization
from ray_tracer_signal_simulation import accumulate_scatters as accumulate_scatters_raytracing
from ray_tracer_signal_simulation import load_mesh_raytracing

from imaging_algorithms import projected_CBP, strip_map_imaging



def sar_render_image(   file_name, num_pulses, poses, az_spread,
                        spatial_bw = 64,
                        spatial_fs = 64,
                        debug_gif = False,
                        debug_gif_suffix = None,
                        snr_db = None,
                        wavelength = None,
                        use_sig_magnitude = True,
                        verbose = False,
                        render_method = 'rasterization',
                        imaging_algorithm = 'cbp',
                        
                        # image size stuff
                        image_width = 64,
                        image_height = 64,
                        image_plane_width = 1,
                        image_plane_height = 1,
                        grid_width = 1,
                        grid_height = 1,
                        n_ray_width = 1,
                        n_ray_height = 1,
                        range_near = 1,
                        range_far = 1,
    ):

    # set device
    device = 'cuda'

    # select the function for accumulating scatters
    if render_method == 'rasterization':
        accumulate_scatters_fn = accumulate_scatters_rasterization
    elif render_method == 'raytracing':
        accumulate_scatters_fn = accumulate_scatters_raytracing
    else:
        raise ValueError('Invalid render method \'%s\', expected \'rasterization\' or \'raytracing\''%render_method)

    # load the mesh and hardcode the material properties
    if render_method == 'rasterization':
        mesh, normals, material_properties = load_mesh(file_name, device=device,)
    elif render_method == 'raytracing':
        mesh = load_mesh_raytracing(file_name, device=device,)
        normals, material_properties = None, None  # embedded into each triangle of the mesh object
    else:
        raise ValueError('Invalid render method \'%s\', expected \'rasterization\' or \'raytracing\''%render_method)

    # SAR raytracing / rasterization
    if verbose:
        print('Accumulating scatters...')

    # (T,P,R)   (T,P,R)       (T,P)    (T,P)      (T,P)     (T,)         (T,)
    all_ranges, all_energies, azimuth, elevation, distance, cam_azimuth, cam_distance = accumulate_scatters_fn(
        poses.to(device),
        mesh, normals, material_properties,

        azimuth_spread = az_spread,
        n_pulses       = num_pulses,
        wavelength     = wavelength,
        debug_gif      = debug_gif,

        # image size stuff
        grid_width     = grid_width,
        grid_height    = grid_height,
        n_ray_width    = n_ray_width,
        n_ray_height   = n_ray_height,
    )
    if verbose:
        print('done.')

    # Generate signal
    # (T,P,Z) (Z,)
    if verbose:
        print('Interpolating signal...')
    signals, sample_z = interpolate_signal(all_ranges/2, all_energies, # divide ranges by 2 to convert to spatial range
            range_near = range_near, range_far = range_far,
            spatial_bw = spatial_bw, spatial_fs = spatial_fs,
            batch_size = None, debug = debug_gif,
    )
    if verbose:
        print('done.')

    # apply SNR
    if snr_db is not None:
        T,P,Z = signals.shape
        signals = apply_snr(signals.reshape(T,P*Z), snr_db).reshape(T,P,Z)

    # compute forward vectors from azimuth and elevation angles
    forward_vectors = -torch.stack([
        torch.cos(3.14159/180*azimuth) * torch.cos(3.14159/180*elevation),
        torch.sin(3.14159/180*azimuth) * torch.cos(3.14159/180*elevation),
        torch.sin(3.14159/180*elevation),
    ], dim=-1)  # (T, P, 3)

    # convert to signal magnitude if desired
    if use_sig_magnitude:
        complex_signals = signals
        signals = signals.abs()

    # Compute sar image
    if verbose:
        print('Computing SAR image...')
    if imaging_algorithm == 'cbp':
        sar_image = projected_CBP(
            signals, 
            sample_z, 
            forward_vectors, 
            cam_azimuth, 
            cam_distance, 
            spatial_fs,
            image_width = image_width,
            image_height = image_height,
            image_plane_width = image_plane_width,
            image_plane_height = image_plane_height,
        )
    elif imaging_algorithm == 'stripmap':
        trajectory = cam_distance.reshape(-1,1,1) * (-forward_vectors)  # (T,P,3)
        sar_image = strip_map_imaging(
            complex_signals,
            wavelength,
            trajectory,
            sample_z,
            spatial_fs,
            planar_wave = False,
            image_plane_rotation_deg = cam_azimuth+90,
            image_width = image_width,
            image_height = image_height,
            image_plane_width = image_plane_width,
            image_plane_height = image_plane_height,
        )
    else:
        raise ValueError('Invalid imaging algorithm \'%s\', expected \'cbp\' or \'stripmap\''%imaging_algorithm)
    if verbose:
        print('done.')

    # make a gif if desired
    if debug_gif:
        signal_gif(signals, all_ranges, all_energies, sample_z, range_near, range_far, suffix =debug_gif_suffix)

    return sar_image




def signal_gif(signals, all_ranges, all_energies, sample_z, z_near, z_far, suffix=None):


    # convert to amplitude
    signals      = torch.abs(signals     )
    all_energies = torch.abs(all_energies)

    # plot the signal and scatters for every pulse
    sig_max = signals.max().item()
    sig_min = signals.min().item()
    energy_max = all_energies.max().item()
    energy_min = all_energies.min().item()
    for p in tqdm.tqdm(range(signals.shape[1]), desc='Plotting scatters and signals'):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(all_ranges[0,p].cpu().numpy(),all_energies[0,p].cpu().numpy())
        plt.title('Scatters')
        plt.xlabel('Range')
        plt.ylabel('Energy')
        plt.xlim(z_near, z_far)
        plt.ylim(energy_min, energy_max)
        plt.subplot(1, 2, 2)
        plt.plot(sample_z.cpu().numpy(), signals[0,p].cpu().numpy())
        plt.title('Signal')
        plt.xlabel('Range')
        plt.ylabel('Amplitude')
        plt.xlim(z_near, z_far)
        plt.ylim(sig_min, sig_max)


        path = get_next_path('figures/tmp/scatters_signal.png')
        savefig(path)

    # make a gif of the depth map, energy map, scatter plot, and signal plot
    # [ [depth, energy],
    #   [scatter, signal] ]
    images = []
    for p in tqdm.tqdm(range(signals.shape[1]), desc='Creating GIF'):

        # load the depth energy image file
        depth_energy_path = f'figures/tmp/depth_energy_{p:02d}.png'
        if not os.path.exists(depth_energy_path):
            print(f'Warning: {depth_energy_path} does not exist. Skipping.')
            continue
        depth_energy_im = PIL.Image.open(depth_energy_path)
        depth_energy_im = np.array(depth_energy_im)
        depth_energy_im = np.tile(depth_energy_im[..., np.newaxis], (1, 1, 3))

        # delete the image
        os.remove(depth_energy_path)

        # load the scatter signal image file
        scatter_signal_path = f'figures/tmp/scatters_signal_{p:02d}.png'
        if not os.path.exists(scatter_signal_path):
            print(f'Warning: {scatter_signal_path} does not exist. Skipping.')
            continue
        scatter_signal_im = PIL.Image.open(scatter_signal_path)
        scatter_signal_im = np.array(scatter_signal_im)[..., :3]

        # delete the image
        os.remove(scatter_signal_path)

        # resize the depth energy image to match the scatter signal image
        new_rows = scatter_signal_im.shape[1] // 2
        depth_energy_im = cv2.resize(depth_energy_im, (scatter_signal_im.shape[1], new_rows))

        # concatenate accross the row dimension
        combined_im = np.concatenate((depth_energy_im, scatter_signal_im), axis=0)

        # save image to list
        images.append(combined_im)

    # make a boomerang gif
    images = np.stack(images, axis=0) # (N, H, W, C)
    time_reversed = np.flip(images, axis=0)  # reverse the time dimension
    images = np.concatenate((images, time_reversed), axis=0)  # concatenate original and reversed

    # create gif from the images
    fps = signals.shape[1]/4.0
    print('Saving GIF with %.1f fps...'% fps)
    if suffix is not None:
        path = f'figures/dm_em_sc_si_{suffix}.gif'
    else:
        path = get_next_path('figures/dm_em_sc_si.gif')
    imageio.mimsave(path, images, fps=fps, format='GIF', loop=0)
    print('GIF saved to: ', path)


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
        render_method = 'rasterization',
        imaging_algorithm = 'cbp',

        image_plane_width = 1,
        image_plane_height = 1,
        image_width = 64,
        image_height = 64, 
        grid_width = 1,
        grid_height = 1,
        n_ray_width = 1,
        n_ray_height = 1, 
        range_near = 1,
        range_far = 1,
    ):
    """
    Renders a random image from the ShapeNet dataset using SAR simulation.
    """

    all_obj_id = os.listdir('/workspace/data/srncars/cars_train/')  # list all object IDs in the dataset
    obj_id     = np.random.choice(all_obj_id, 1)[0]  # randomly select an object ID from the dataset
    print('Selected object ID: ', obj_id)

    all_pose_paths = '/workspace/data/srncars/cars_train/%s/pose/'%obj_id
    all_pose_nums  = os.listdir(all_pose_paths)
    pose_num       = np.random.choice(all_pose_nums, 1)[0].split('.')[0]
    print('Selected pose number: ', pose_num)

    if suffix is None:
        suffix = '%s_%s'%(pose_num, obj_id)

    # load image, pose, and mesh
    rgb_path  = '/workspace/data/srncars/cars_train/%s/rgb/%s.png' % (obj_id, pose_num)
    pose_path = '/workspace/data/srncars/cars_train/%s/pose/%s.txt' % (obj_id, pose_num)
    mesh_path = '/workspace/data/srncars/02958343/%s/models/model_normalized.obj' % obj_id
    rgb  = np.array(PIL.Image.open(rgb_path))[...,:3] # (H, W, 3)
    pose = np.loadtxt(pose_path).reshape(1,4,4).astype(np.float32)  # (4, 4)
    
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

                            debug_gif=debug_gif, # debug gif
                            debug_gif_suffix = suffix,

                            render_method=render_method,

                            imaging_algorithm=imaging_algorithm,

                            # image size stuff
                            image_width = image_width,
                            image_height = image_height,
                            image_plane_width = image_plane_width,
                            image_plane_height = image_plane_height,
                            grid_width = grid_width,
                            grid_height = grid_height,
                            n_ray_width = n_ray_width,
                            n_ray_height = n_ray_height,
                            range_near = range_near,
                            range_far = range_far,
    ) # (1,H,W)

    # plot the SAR image next to the RGB image
    sar = torch.tile(sar, (3,1,1)).permute(1,2,0)  # (H,W,3)
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
    else:
        path = 'figures/sar_rgb_image_%s.png'%(suffix)
    image.save(path)  # save the image
    print('Saved SAR and RGB image to: ', path)




def multi_param_experiment(param_dict, default_kwargs, experiment_name="experiment", seed=8134, custom_title_strings = None):
    """
    A modular function to run experiments by varying multiple parameters together
    
    Args:
        param_dict (dict): Dictionary where each key is a parameter name and value is a list/array of values.
                          All lists/arrays must have the same length.
        default_kwargs (dict): Default arguments for render_random_image
        experiment_name (str): Name of the experiment for saving files
        seed (int): Random seed for reproducibility
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
        if experiment_name in f:
            os.remove(os.path.join('figures', f))

    # create strings to title each experiment
    if custom_title_strings is None:
        experiment_strings = []
        for i in range(n_experiments):
            param_str_parts = []
            for param_name, param_vals in param_dict.items():
                try:
                    # param_str_parts.append(f"{param_name}{int(param_vals[i])}")
                    try:
                        param_str_parts.append("%s%.2f" % (param_name, float(param_vals[i])))
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

    # find all files in figures that have the experiment name
    figure_files = [f for f in os.listdir('figures') if f'sar_rgb_image_{experiment_name}' in f]
    
    # Extract the figure ID (the 3-digit number after experiment_name_)
    figure_ids = [int(f.split(experiment_name + '_')[1][:3]) for f in figure_files]
    
    # Sort files by the figure ID
    sorted_files = [f for _, f in sorted(zip(figure_ids, figure_files))]

    # Load images, crop left 128 columns, and collect
    cropped_images = []
    for i, fname in enumerate(sorted_files):
        img = PIL.Image.open(os.path.join('figures', fname))
        cropped = img.crop((128, 0, img.width, img.height))
        draw = ImageDraw.Draw(cropped)
        
        draw.text((10, 10), experiment_strings[i], fill=(255, 255, 255))
        cropped_images.append(np.array(cropped))

    # Stitch horizontally
    # if divisible by 2, split into 2 rows
    n_image = len(cropped_images)
    if n_image%2 == 0 and n_image > 4:
        stitched = np.vstack([
            np.hstack(cropped_images[:n_image//2 ]),
            np.hstack(cropped_images[ n_image//2:]),
        ])
    else:
        stitched = np.hstack(cropped_images)
    stitched_img = PIL.Image.fromarray(stitched)
    path = f'figures/sar_{experiment_name}_stitched.png'
    stitched_img.save(path)
    print('Saved stitched image to: %s' % path)


if __name__ == '__main__':
    
    # # Bandwidth experiment
    # bw_vals = torch.tensor(np.linspace(128, 256, 10,endpoint=True))
    # # bw_vals = 2**torch.arange(3, 11)
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'azimuth_spread': 100,

    # }
    # multi_param_experiment({'bw': bw_vals, 'fs': bw_vals}, default_kwargs, "bw_experiment")

    # # ray density experiment
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'azimuth_spread': 100,
    #     'spatial_bw': 170.7,
    #     'spatial_fs': 170.7,
    # }
    # vary_kwargs = {
    #     'n_rays_per_side': np.linspace(133,140, 8,endpoint=True).astype(np.int32).tolist()
    # }
    # multi_param_experiment(vary_kwargs, default_kwargs, "ray_density_experiment")

    # # Azimuth spread experiment
    # n_vals = 5  # number of parameter combinations
    # spread_vals = torch.linspace(60, 180, n_vals)
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'fs': 64,
    #     'bw': 64,
    # }
    # param_dict = {
    #     'azimuth_spread': spread_vals
    # }
    # multi_param_experiment(param_dict, default_kwargs, "spread_experiment")

    # # SNR experiment
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'azimuth_spread': 100,
    #     'spatial_bw': 128,
    #     'spatial_fs': 128,
    #     'n_rays_per_side': 200,
    #     'wavelength': 0.3,
    #     'use_sig_magnitude': True,
    #     'snr_db': 30
    # }
    # vary_kwargs = {
    #     'grid_size': np.linspace(0,30, 9,endpoint=True).tolist()+[None]
    # }
    # multi_param_experiment(vary_kwargs, default_kwargs, "snr_experiment")

    # # wavelength experiment WITH magnitude of signal used for CBP
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'azimuth_spread': 100,
    #     'spatial_bw': 128,
    #     'spatial_fs': 128,
    #     'n_rays_per_side': 500,
    #     'snr_db': None,
    #     'use_sig_magnitude': True,
    # }
    # vary_kwargs = {
    #     'wavelength': (10**np.linspace(np.log10(0.01), np.log10(2), 8, endpoint=True)).tolist()
    #     # 'wavelength': np.linspace(0.01, 0.1, 10, endpoint=True).tolist()+[None]
    # }
    # print('vary_kwargs:', vary_kwargs)
    # multi_param_experiment(vary_kwargs, default_kwargs, "mag_wavelength_experiment")

    # # wavelength experiment WITHOUT magnitude of signal used for CBP
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'azimuth_spread': 100,
    #     'spatial_bw': 128,
    #     'spatial_fs': 128,
    #     'n_rays_per_side': 500,
    #     'snr_db': None,
    #     'use_sig_magnitude': False,
    # }
    # vary_kwargs = {
    #     'wavelength': (10**np.linspace(np.log10(0.01), np.log10(2), 8, endpoint=True)).tolist()
    #     # 'wavelength': np.linspace(0.01, 0.1, 10, endpoint=True).tolist()+[None]
    # }
    # print('vary_kwargs:', vary_kwargs)
    #

    # # variable ray grid size experiment
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'azimuth_spread': 100,
    #     'spatial_bw': 128,
    #     'spatial_fs': 128,
    #     'wavelength': 0.3,
    #     'use_sig_magnitude': True,
    #     'snr_db': 30,

    #     'image_width'        : 128,
    #     'image_height'       : 128,
    #     'image_plane_width'  : 1,
    #     'image_plane_height' : 1,
    #     'grid_width'         : 2,
    #     'grid_height'        : 2,
    #     'n_ray_width'        : 256,
    #     'n_ray_height'       : 256,
    #     'range_near'         : 0.5,
    #     'range_far'          : 2.1,
    # }
    # # a sensible example
    # vary_kwargs = {
    #     'grid_width'         : np.linspace(0.5, 1.2, 6, endpoint=True),
    #     'grid_height'        : np.linspace(0.5, 1.2, 6, endpoint=True),
    # }
    # multi_param_experiment(vary_kwargs, default_kwargs, "var_image_size_stuff")

    # # render method
    # default_kwargs = {
    #     'debug_gif': True,
    #     'num_pulse': 32,
    #     'azimuth_spread': 100,
    #     'spatial_bw': 128,
    #     'spatial_fs': 128,
    #     'wavelength': 0.3,
    #     'use_sig_magnitude': True,
    #     'snr_db': 30,

    #     'image_width'        : 128,
    #     'image_height'       : 128,
    #     'image_plane_width'  : 1,
    #     'image_plane_height' : 1,
    #     'grid_width'         : 2,
    #     'grid_height'        : 2,
    #     'n_ray_width'        : 256,
    #     'n_ray_height'       : 256,
    #     'range_near'         : 0.5,
    #     'range_far'          : 2.1,
    #     'grid_width'         : 1.2,
    #     'grid_height'        : 1.2,
    # }
    # # a sensible example
    # vary_kwargs = {
    #     'render_method': ['rasterization', 'raytracing'],
    # }
    # multi_param_experiment(vary_kwargs, default_kwargs, "render_method_experiment")

    ####################################### PAPER RESULTS #######################################
    ####################################### PAPER RESULTS #######################################
    ####################################### PAPER RESULTS #######################################
    ####################################### PAPER RESULTS #######################################
    ####################################### PAPER RESULTS #######################################
    ####################################### PAPER RESULTS #######################################

    # # azimuth spread
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     # 'azimuth_spread': 90,
    #     'spatial_bw': 90,
    #     'spatial_fs': 90,
    #     'wavelength': 0.5,
    #     'use_sig_magnitude': True,
    #     'snr_db': 50,

    #     'image_width'        : 128,
    #     'image_height'       : 128,
    #     'image_plane_width'  : 1,
    #     'image_plane_height' : 1,
    #     'grid_width'         : 2,
    #     'grid_height'        : 2,
    #     'n_ray_width'        : 256,
    #     'n_ray_height'       : 256,
    #     'range_near'         : 0.5,
    #     'range_far'          : 2.1,
    #     'grid_width'         : 1.2,
    #     'grid_height'        : 1.2,

    #     # 'render_method': 'rasterization',
    #     'render_method': 'raytracing',
    # }
    # vary_kwargs = {
    #     'azimuth_spread': np.linspace(0,360,8).tolist(),
    # }
    # multi_param_experiment(vary_kwargs, default_kwargs, "az_spread")

    # # num pulses
    # default_kwargs = {
    #     'debug_gif': False,
    #     # 'num_pulse': 32,
    #     'azimuth_spread': 90,
    #     'spatial_bw': 90,
    #     'spatial_fs': 90,
    #     'wavelength': 0.5,
    #     'use_sig_magnitude': True,
    #     'snr_db': 50,

    #     'image_width'        : 128,
    #     'image_height'       : 128,
    #     'image_plane_width'  : 1,
    #     'image_plane_height' : 1,
    #     'grid_width'         : 2,
    #     'grid_height'        : 2,
    #     'n_ray_width'        : 256,
    #     'n_ray_height'       : 256,
    #     'range_near'         : 0.5,
    #     'range_far'          : 2.1,
    #     'grid_width'         : 1.2,
    #     'grid_height'        : 1.2,

    #     # 'render_method': 'rasterization',
    #     'render_method': 'raytracing',
    # }
    # vary_kwargs = {
    #     'num_pulse': np.linspace(2,32,8).astype(np.int32).tolist(),
    # }
    # multi_param_experiment(vary_kwargs, default_kwargs, "num_pulse")

    # # bw/fs
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'azimuth_spread': 90,
    #     # 'spatial_bw': 90,
    #     # 'spatial_fs': 90,
    #     'wavelength': 0.5,
    #     'use_sig_magnitude': True,
    #     'snr_db': 50,

    #     'image_width'        : 128,
    #     'image_height'       : 128,
    #     'image_plane_width'  : 1,
    #     'image_plane_height' : 1,
    #     'grid_width'         : 2,
    #     'grid_height'        : 2,
    #     'n_ray_width'        : 256,
    #     'n_ray_height'       : 256,
    #     'range_near'         : 0.5,
    #     'range_far'          : 2.1,
    #     'grid_width'         : 1.2,
    #     'grid_height'        : 1.2,

    #     # 'render_method': 'rasterization',
    #     'render_method': 'raytracing',
    # }
    # vary_kwargs = {
    #     'spatial_bw': [4,8,16,32,64,128,256,512],
    #     'spatial_fs': [4,8,16,32,64,128,256,512],
    # }
    # custom_title_strings = [ 'Fs: 4', 'Fs: 8', 'Fs: 16', 'Fs: 32', 'Fs: 64', 'Fs: 128', 'Fs: 256', 'Fs: 512']
    # multi_param_experiment(vary_kwargs, default_kwargs, "fsbw",custom_title_strings = custom_title_strings)

    # # Noise
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'azimuth_spread': 90,
    #     'spatial_bw': 90,
    #     'spatial_fs': 90,
    #     'wavelength': 0.5,
    #     'use_sig_magnitude': True,
    #     # 'snr_db': 50,

    #     'image_width'        : 128,
    #     'image_height'       : 128,
    #     'image_plane_width'  : 1,
    #     'image_plane_height' : 1,
    #     'grid_width'         : 2,
    #     'grid_height'        : 2,
    #     'n_ray_width'        : 256,
    #     'n_ray_height'       : 256,
    #     'range_near'         : 0.5,
    #     'range_far'          : 2.1,
    #     'grid_width'         : 1.2,
    #     'grid_height'        : 1.2,

    #     # 'render_method': 'rasterization',
    #     'render_method': 'raytracing',
    # }
    # snr_db = np.linspace(0,22,8).tolist()
    # vary_kwargs = {
    #     'snr_db': snr_db
    # }
    # custom_title_strings = ['SNR dB: %.1f'%s for s in snr_db]
    # multi_param_experiment(vary_kwargs, default_kwargs, "snrdb", custom_title_strings=custom_title_strings)

    # # render method
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'azimuth_spread': 90,
    #     'spatial_bw': 90,
    #     'spatial_fs': 90,
    #     'wavelength': 0.5,
    #     'use_sig_magnitude': True,
    #     'snr_db': 50,

    #     'image_width'        : 128,
    #     'image_height'       : 128,
    #     'image_plane_width'  : 1,
    #     'image_plane_height' : 1,
    #     'grid_width'         : 2,
    #     'grid_height'        : 2,
    #     'n_ray_width'        : 256,
    #     'n_ray_height'       : 256,
    #     'range_near'         : 0.5,
    #     'range_far'          : 2.1,
    #     'grid_width'         : 1.2,
    #     'grid_height'        : 1.2,

    #     # 'render_method': 'rasterization',
    #     # 'render_method': 'raytracing',
    # }
    # vary_kwargs = {
    #     'render_method': ['rasterization','raytracing']
    # }
    # custom_title_strings = ['Rasterization','Ray Tracing']
    # multi_param_experiment(vary_kwargs, default_kwargs, "rendermethod", custom_title_strings=custom_title_strings)

    # # wavelength magnitude
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'azimuth_spread': 90,
    #     'spatial_bw': 90,
    #     'spatial_fs': 90,
    #     # 'wavelength': 0.5,
    #     'use_sig_magnitude': True,
    #     'snr_db': 50,

    #     'image_width'        : 128,
    #     'image_height'       : 128,
    #     'image_plane_width'  : 1,
    #     'image_plane_height' : 1,
    #     'grid_width'         : 2,
    #     'grid_height'        : 2,
    #     'n_ray_width'        : 256,
    #     'n_ray_height'       : 256,
    #     'range_near'         : 0.5,
    #     'range_far'          : 2.1,
    #     'grid_width'         : 1.2,
    #     'grid_height'        : 1.2,

    #     # 'render_method': 'rasterization',
    #     'render_method': 'raytracing',
    # }
    # vary_kwargs = {
    #     'wavelength': [0.01, 0.02, 0.05, 0.1]
    # }
    # custom_title_strings = ['wavelength: 0.01','wavelength: 0.02','wavelength: 0.05','wavelength: 0.10',]
    # multi_param_experiment(vary_kwargs, default_kwargs, "wavelengthmagnitude", custom_title_strings=custom_title_strings)

    # # wavelength complex
    # default_kwargs = {
    #     'debug_gif': False,
    #     'num_pulse': 32,
    #     'azimuth_spread': 90,
    #     'spatial_bw': 90,
    #     'spatial_fs': 90,
    #     # 'wavelength': 0.5,
    #     'use_sig_magnitude': False,
    #     'snr_db': 50,

    #     'image_width'        : 128,
    #     'image_height'       : 128,
    #     'image_plane_width'  : 1,
    #     'image_plane_height' : 1,
    #     'grid_width'         : 2,
    #     'grid_height'        : 2,
    #     'n_ray_width'        : 256,
    #     'n_ray_height'       : 256,
    #     'range_near'         : 0.5,
    #     'range_far'          : 2.1,
    #     'grid_width'         : 1.2,
    #     'grid_height'        : 1.2,

    #     # 'render_method': 'rasterization',
    #     'render_method': 'raytracing',
    # }
    # vary_kwargs = {
    #     'wavelength': [0.01, 0.02, 0.05, 0.1]
    # }
    # custom_title_strings = ['wavelength: 0.01','wavelength: 0.02','wavelength: 0.05','wavelength: 0.10',]
    # multi_param_experiment(vary_kwargs, default_kwargs, "wavelengthcomplex", custom_title_strings=custom_title_strings)

    # idk
    default_kwargs = {
        'debug_gif': False,
        # 'debug_gif': True,
        'num_pulse': 32,
        'azimuth_spread': 360,
        'spatial_bw': 90,
        'spatial_fs': 90,
        # 'wavelength': 0.5,
        'use_sig_magnitude': True,
        'snr_db': 50,

        'image_width'        : 128,
        'image_height'       : 128,
        'image_plane_width'  : 1,
        'image_plane_height' : 1,
        'grid_width'         : 2,
        'grid_height'        : 2,
        'n_ray_width'        : 256,
        'n_ray_height'       : 256,
        'range_near'         : 0.5,
        'range_far'          : 2.1,
        'grid_width'         : 1.2,
        'grid_height'        : 1.2,

        'render_method': 'rasterization',
        # 'render_method': 'raytracing',

        'imaging_algorithm': 'stripmap',

    }
    vary_kwargs = {
        'wavelength': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    }
    custom_title_strings = None
    multi_param_experiment(vary_kwargs, default_kwargs, "otherplots", custom_title_strings=custom_title_strings)