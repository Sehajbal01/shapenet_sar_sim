import torch
import numpy as np
from matplotlib import pyplot as plt
from signal_simulation import accumulate_scatters, interpolate_signal
from pytorch3d.renderer import look_at_view_transform


def camera_to_world_matrix(azimuth, elevation, distance, device='cuda', debug=False):
    rotation, translation = look_at_view_transform(distance, elevation, azimuth,device=device)
    pose = torch.eye(4, device=device)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation.squeeze()
    return pose

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

    # plot the signal and scatters for the first pulse
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(all_ranges[0,0].cpu().numpy(),all_energies[0,0].cpu().numpy())
    plt.title('Scatters')
    plt.xlabel('Range')
    plt.ylabel('Energy')
    plt.xlim(z_near, z_far)
    plt.subplot(1, 2, 2)
    plt.plot(sample_z.cpu().numpy(), signals[0,0].cpu().numpy())
    plt.title('Signal')
    plt.xlabel('Range')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig('figures/scatters_and_signal.png')
    plt.close()


if __name__ == '__main__':
    
    sar_render_image( '/workspace/data/srncars/02958343/7dac31838b627748eb631ba05bd8dfe/models/model_normalized.obj', # fname
                      50, # num_pulses
                      90, # azimuth angle
                      45, # elevation angle
                      360 # azimuth spread
    )
