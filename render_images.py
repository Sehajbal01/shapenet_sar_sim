import torch
import numpy as np
from matplotlib import pyplot as plt
from accumulate_scatters import accumulate_scatters


def camera_to_world_matrix(azimuth_deg, elevation_deg, distance, device='cpu', debug=False):
    """
    Computes a 4x4 camera-to-world matrix from spherical coordinates.
    The format of the pose matrix is:
    [ [ right, down, forward, camera_center ],
      [ 0    ,    0,       0,             1 ] ]
    where right, down, forward are (3,1) vectors of the camera axes in world coordinates.

    Args:
        azimuth_deg (float): Azimuth angle in degrees (from X-axis in XY plane)
        elevation_deg (float): Elevation angle in degrees (from XY plane toward Z)
        distance (float): Distance from the origin
        device (str or torch.device): Device to compute on

    Returns:
        torch.Tensor: (4, 4) camera-to-world pose matrix
    """
    # convert to radians
    a = azimuth_deg   * np.pi / 180.0
    e = elevation_deg * np.pi / 180.0

    # calculate camera center
    c = distance * np.array([
        np.cos(a) * np.cos(e),
        np.sin(a) * np.cos(e),
        np.sin(e)
    ], dtype=np.float32)

    # calculate camera axes u,v,w = right, down, forward
    w = -c
    v = np.array([
        np.cos(a) * np.cos(e-np.pi/2),
        np.sin(a) * np.cos(e-np.pi/2),
        np.sin(e-np.pi/2)
    ], dtype=np.float32)
    u = np.cross(v, w)

    # normalize the axes
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = w / np.linalg.norm(w)

    # assemble the pose matrix
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = u
    pose[:3, 1] = v
    pose[:3, 2] = w
    pose[:3, 3] = c

    if debug:
        print("a,e,d: ", azimuth_deg, elevation_deg, distance)
        print("pose: \n", pose)
        print("temp_v dot w: ", np.dot(temp_v, w))
        print("magnitude temp_v: ", np.linalg.norm(temp_v))
        print("temp_v: \n", temp_v)
        print("v: \n", v)
        print()

    # apply the JhihYang Transform [+1 -1 -1 +1]
    coord_transform = np.diag([1, -1, -1, 1]).astype(np.float32)  # (4, 4)
    pose = pose @ coord_transform  # (4, 4)

    return torch.tensor(pose, device=device).reshape(4, 4)  # (4, 4)


def sar_render_image( file_name, num_pulses, az_angle, ele_angle, az_spread,
                      z_near = 0.8,
                      z_far  = 1.8,
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

    # plot the scatters
    plt.scatter(all_ranges[0,0].cpu().numpy(),all_energies[0,0].cpu().numpy())
    plt.savefig('figures/scatter_plot.png')
    plt.close()

    # Generate signal
    # z_vals, signal = simulate_echo_signal(all_ranges, all_energies, z_near, z_far, radar_fs)

    # plot the signal


if __name__ == '__main__':
    
    sar_render_image( '/workspace/data/srncars/02958343/7dac31838b627748eb631ba05bd8dfe/models/model_normalized.obj', # fname
                      50, # num_pulses
                      90, # azimuth angle
                      45, # elevation angle
                      360 # azimuth spread
    )
