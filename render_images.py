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
from signal_simulation import accumulate_scatters, interpolate_signal
# from signal_simulation import accumulate_scatters, interpolate_signal



def sar_render_image(   file_name, num_pulses, poses, az_spread,
                        z_near = 0.8,
                        z_far  = 1.8,
                        spatial_bw = 64,
                        spatial_fs = 64,
                        debug_gif = False,
                        debug_gif_suffix = None,
                        image_size = 128,
                        n_rays_per_side = 128,
    ):

    # set device
    device = 'cuda'

    # SAR raytracing 
    print('Accumulating scatters...')
    # (T,P,R)   (T,P,R)       (T,P)    (T,P)      (T,P)     (T,)         (T,)
    all_ranges, all_energies, azimuth, elevation, distance, cam_azimuth, cam_distance = accumulate_scatters(
        poses.to(device), z_near, z_far, file_name,
        azimuth_spread=az_spread,
        n_pulses=num_pulses,
        n_rays_per_side=n_rays_per_side,
        debug_gif=debug_gif,
    )
    print('done.')

    # Generate signal
    # (T,P,Z) (Z,)
    print('Interpolating signal...')
    signals, sample_z = interpolate_signal(all_ranges, all_energies, z_near, z_far,
            spatial_bw = spatial_bw, spatial_fs = spatial_fs,
            batch_size = None,
    )
    print('done.')

    # compute forward vectors from azimuth and elevation angles
    forward_vectors = -torch.stack([
        torch.cos(3.14159/180*azimuth) * torch.cos(3.14159/180*elevation),
        torch.sin(3.14159/180*azimuth) * torch.cos(3.14159/180*elevation),
        torch.sin(3.14159/180*elevation),
    ], dim=-1)  # (T, P, 3)

    # Compute sar image
    print('Computing SAR image...')
    sar_image = convolutional_back_projection(signals, sample_z, forward_vectors, cam_azimuth, cam_distance, z_near-z_far, spatial_fs, image_size, image_size)
    print('done.')

    # make a gif if desired
    if debug_gif:
        signal_gif(signals, all_ranges, all_energies, sample_z, z_near, z_far, suffix =debug_gif_suffix)

    return sar_image


def convolutional_back_projection(signal, sample_z, forward_vector, cam_azimuth, cam_distance, side_len, spatial_fs, H, W):
    '''
    Convolutional back projection for SAR imaging.

    inputs:
        signal: (T,P,Z) - the signal to be back projected
        sample_z: (Z,) - the range samples
        forward_vector: (T,P,3) - the forward vector for each ray
        cam_azimuth: (T,) - the azimuth angle of the camera
        cam_distance: (T,) - the distance of the camera from the origin
        side_len: float - the length of the side of the image plane
        spatial_fs: float - the spatial frequency sampling rate
        H: int - the height of the target image
        W: int - the width of the target image

    outputs:
        image: (T,H,W) - the computed image
    '''
    # Get shapes
    T,P,Z = signal.shape
    I = H * W
    device = signal.device

    # filter with |r| in frequency domain (equation 2.30)
    sample_r = sample_z - cam_distance.reshape(T,1)  # (T,Z)
    signal_freq = torch.fft.fftshift(torch.fft.fft(signal, dim=-1), dim=-1)  # (T,P,Z)
    filtered_signal_freq = signal_freq * torch.abs(sample_r.reshape(T,1,Z)) # (T,P,Z)
    filtered_signal = torch.fft.ifft(torch.fft.ifftshift(filtered_signal_freq, dim=-1), dim=-1)  # (T,P,Z)

    # create grid of target image cooordinates on the ground plane
    h_coord,w_coord = torch.meshgrid(   torch.linspace(-side_len/2, side_len/2, H, device=device, dtype=sample_z.dtype),
                                        torch.linspace(-side_len/2, side_len/2, W, device=device, dtype=sample_z.dtype),
                                        indexing='ij')  # (H,W)
    h_coord = h_coord.float()  # (H,W)
    w_coord = w_coord.float()  # (H,W)

    # rotate the image plane according to the azimuth angle of the target pose
    # we want the SAR image to look as if we took the current camera position and moved up in elevation
    # so the top pixel should be in the up vector direction projected onto the ground plane
    coord_grid = torch.stack((-h_coord, -w_coord), dim=-1) # (H,W,2)
    cam_azimuth_rad = cam_azimuth * (np.pi / 180.0)  # convert to radians
    rotation_matrix = torch.stack([
        torch.cos(cam_azimuth_rad), -torch.sin(cam_azimuth_rad), torch.sin(cam_azimuth_rad), torch.cos(cam_azimuth_rad)
    ], dim=-1) # (T,4)
    coord_grid = rotation_matrix.reshape(T,1,2,2) @ coord_grid.reshape(1,I,2,1)  # (T,I,2,1)

    # interpolate pixel coordinated projected onto the filtered signal
    r_coord = torch.sum(forward_vector[...,:2].reshape(T,P,1,2) * coord_grid.reshape(T,1,I,2), dim=-1)  # (T,P,I,2)
    interpolated_r_points = torch.sum( filtered_signal.reshape(T,P,1,Z) * \
                                    torch.sinc( spatial_fs * (r_coord.reshape(T,P,I,1) - sample_r.reshape(1,1,1,Z)) ), # (T,P,I,Z)
                                    dim=-1
                                    ) # (T,P,I)
    
    # integrate over theta (eqation 2.31)
    image = torch.sum(interpolated_r_points, dim=1) / (4*np.pi**2)  # (T,I)

    # reshape and convert to real-valued images
    image = image.reshape(T,H,W)
    return torch.sqrt(image.real**2 + image.imag**2)  # (T,H,W)


def signal_gif(signals, all_ranges, all_energies, sample_z, z_near, z_far, suffix=None):


    # convert to amplitude
    try:
        signals = torch.sqrt(signals.real**2 + signals.imag**2)
    except(RuntimeError):
        pass

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


def render_random_image(debug_gif=False, num_pulse=120, azimuth_spread = 180, fsbw = 64, suffix = None):
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

                            z_near = 0.8,
                            z_far  = 1.8,
                            spatial_bw = fsbw,
                            spatial_fs = fsbw,
                            image_size = 64,
                            n_rays_per_side = 128,

                            debug_gif=debug_gif, # debug gif
                            debug_gif_suffix = suffix,
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


def az_spread_experiment():

    seed = 8134

    # remove all files in figures with *spread*
    for f in os.listdir('figures'):
        if 'spread' in f:
            os.remove(os.path.join('figures', f))

    # generate the images for each azspread
    for azimuth_spread in torch.arange(0, 361, 30).numpy():

        # set the random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        render_random_image(debug_gif=False, num_pulse=50, azimuth_spread=azimuth_spread, suffix='spread%d'%azimuth_spread)

    # find all files in figures that have 'spread' in the name and remove the left 128 columns, and stich them into 1 wide image
    figure_files = [f for f in os.listdir('figures') if 'spread' in f]
    figure_num = [int(f.split('spread')[-1].split('.')[0]) for f in figure_files]
    # Sort files by spread number
    sorted_files = [f for _, f in sorted(zip(figure_num, figure_files))]

    # Load images, crop left 128 columns, and collect
    cropped_images = []
    for fname in sorted_files:
        img = PIL.Image.open(os.path.join('figures', fname))
        cropped = img.crop((128, 0, img.width, img.height))
        cropped_images.append(np.array(cropped))

    # Stitch horizontally
    stitched = np.hstack(cropped_images)
    stitched_img = PIL.Image.fromarray(stitched)
    stitched_img.save('figures/sar_spread_stitched.png')


def pulse_experiment():

    seed = 8134

    # remove all files in figures with *spread*
    for f in os.listdir('figures'):
        if 'pulses' in f:
            os.remove(os.path.join('figures', f))

    n_pulse_vals = torch.arange(2, 101, 5).numpy()

    # generate the images for each azspread
    for n_pulse in n_pulse_vals:

        # set the random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        render_random_image(debug_gif=False, num_pulse=n_pulse, azimuth_spread=100, suffix='pulses%d'%n_pulse)

    # find all files in figures that have 'pulses' in the name and remove the left 128 columns, and stich them into 1 wide image
    figure_files = [f for f in os.listdir('figures') if 'pulses' in f]
    figure_num = [int(f.split('pulses')[-1].split('.')[0]) for f in figure_files]
    # Sort files by pulses number
    sorted_files = [f for _, f in sorted(zip(figure_num, figure_files))]

    # Load images, crop left 128 columns, and collect
    cropped_images = []
    for i,fname in enumerate(sorted_files):
        img = PIL.Image.open(os.path.join('figures', fname))
        cropped = img.crop((128, 0, img.width, img.height))
        draw = ImageDraw.Draw(cropped)
        draw.text((10, 10), 'Pulses: %d'%n_pulse_vals[i], fill=(255, 255, 255))
        cropped_images.append(np.array(cropped))

    # Stitch horizontally
    stitched = np.hstack(cropped_images)
    stitched_img = PIL.Image.fromarray(stitched)
    stitched_img.save('figures/sar_pulses_stitched.png')


def bw_experiment():

    seed = 8134

    # remove all files in figures with *spread*
    for f in os.listdir('figures'):
        if 'bw' in f:
            os.remove(os.path.join('figures', f))

    bw_vals = 2**np.arange(13)

    # generate the images for each azspread
    for bw in bw_vals:

        # set the random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        render_random_image(debug_gif=False, num_pulse=32, azimuth_spread=100, fsbw=bw, suffix='bw%d'%bw)

    # find all files in figures that have 'bw' in the name and remove the left 128 columns, and stich them into 1 wide image
    figure_files = [f for f in os.listdir('figures') if 'bw' in f]
    figure_num = [int(f.split('bw')[-1].split('.')[0]) for f in figure_files]
    # Sort files by bw number
    sorted_files = [f for _, f in sorted(zip(figure_num, figure_files))]

    # Load images, crop left 128 columns, and collect
    cropped_images = []
    for i,fname in enumerate(sorted_files):
        img = PIL.Image.open(os.path.join('figures', fname))
        cropped = img.crop((128, 0, img.width, img.height))
        draw = ImageDraw.Draw(cropped)
        draw.text((10, 10), 'BW: %d'%bw_vals[i], fill=(255, 255, 255))
        cropped_images.append(np.array(cropped))

    # Stitch horizontally
    stitched = np.hstack(cropped_images)
    stitched_img = PIL.Image.fromarray(stitched)
    stitched_img.save('figures/sar_bw_stitched.png')



if __name__ == '__main__':
    # # select random seed
    # seed = np.random.randint(0, 10000)
    # seed = 8134
    # print('Random seed: ', seed)

    # # for i in range(10):
    bw_experiment()


