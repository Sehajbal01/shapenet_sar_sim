import tqdm
import PIL
import imageio
import cv2
import os
from utils import get_next_path
import torch
import numpy as np
from matplotlib import pyplot as plt
from signal_simulation import accumulate_scatters, interpolate_signal
from utils import camera_to_world_matrix



def sar_render_image( file_name, num_pulses, az_angle, ele_angle, az_spread,
                      z_near = 0.8,
                      z_far  = 1.8,
                      spatial_bw = 64,
                      spatial_fs = 64,
    ):

    # set device
    device = 'cuda'

    # get target pose
    target_pose = camera_to_world_matrix(az_angle, ele_angle, (z_near + z_far) / 2, device=device, debug=False)
    target_poses = target_pose.reshape(1,4,4)

    # SAR raycasting 
    # (T,P,R)   (T,P,R)       (T,P)    (T,P)      (T,P)     (T,P,3)
    all_ranges, all_energies, azimuth, elevation, distance, forward_vectors = accumulate_scatters(
        target_poses, z_near, z_far, file_name,
        azimuth_spread=az_spread,
        n_pulses=num_pulses,
        n_rays_per_side=128,
        debug_gif=True,
    )

    # Generate signal
    # (T,P,Z) (Z,)
    signals, sample_z = interpolate_signal(all_ranges, all_energies, z_near, z_far,
            spatial_bw = spatial_bw, spatial_fs = spatial_fs,
            batch_size = None,
    )

    signal_gif(signals, all_ranges, all_energies, sample_z, z_near, z_far)


def signal_gif(signals, all_ranges, all_energies, sample_z, z_near, z_far):

    # plot the signal and scatters for every pulse
    for p in tqdm.tqdm(range(signals.shape[1]), desc='Plotting scatters and signals'):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(all_ranges[0,p].cpu().numpy(),all_energies[0,p].cpu().numpy())
        plt.title('Scatters')
        plt.xlabel('Range')
        plt.ylabel('Energy')
        plt.xlim(z_near, z_far)
        plt.subplot(1, 2, 2)
        plt.plot(sample_z.cpu().numpy(), signals[0,p].cpu().numpy())
        plt.title('Signal')
        plt.xlabel('Range')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        path = get_next_path('figures/tmp/scatters_signal.png')
        plt.savefig(path)
        plt.close()

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

        # load the scatter signal image file
        scatter_signal_path = f'figures/tmp/scatters_signal_{p:02d}.png'
        if not os.path.exists(scatter_signal_path):
            print(f'Warning: {scatter_signal_path} does not exist. Skipping.')
            continue
        scatter_signal_im = PIL.Image.open(scatter_signal_path)
        scatter_signal_im = np.array(scatter_signal_im)[..., :3]

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
    imageio.mimsave('figures/dm_em_sc_si.gif', images, fps=fps, format='GIF', loop=0)

if __name__ == '__main__':
    
    sar_render_image( '/workspace/data/srncars/02958343/7dac31838b627748eb631ba05bd8dfe/models/model_normalized.obj', # fname
                      50, # num_pulses
                      90, # azimuth angle
                      45, # elevation angle
                      30 # azimuth spread
    )
