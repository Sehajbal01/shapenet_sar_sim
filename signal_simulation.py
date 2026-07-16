from utils import savefig, correct_material_properties, dot_product, directional_scatter_polynomial_alpha5, rotation_matrix_xyz
import matplotlib.pyplot as plt
import tqdm
import os
import imageio
import sys
import torch
from PIL import Image
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings, 
    MeshRasterizer,  
)
import math
import time
from utils import get_next_path, extract_pose_info, spherical_to_cartesian, generate_pose_mat, cartesian_to_spherical
from ray_tracer_v2 import ray_trace, build_octree
from accumulate_scatters import accumulate_scatters



def generate_trajectory(pose, trajectory_type='linear', n_pulses=100, azimuth_spread_deg=None, trajectory_noise_var=0, debug=False):
    '''
    Generate a trajectory for multiple poses based on a single pose. The pose is the middle of the trajectory.
    '''
    assert(azimuth_spread_deg >= 0  ), 'the azimuth cannot be negative'

    device = pose.device
    T = pose.shape[0]  # no. of camera views
    P = n_pulses               # no. of pulses per view

    # Pull out camera positions info # TODO: this is probably not consistent with the pytorch3d coordinate system
    cam_center, cam_right, cam_up, cam_forward, cam_distance, cam_elevation_deg, cam_azimuth_deg = extract_pose_info(pose)
    #  (T,3)       (T,3)      (T,3)    (T,3)        (T,)           (T,)              (T,)
   

    if trajectory_type == 'circular':
        # Spread the pulses across a small range of azimuth angles
        azimuth_offsets = torch.linspace(-azimuth_spread_deg / 2, azimuth_spread_deg / 2, P, device=device) # (P,)
        azimuth_deg = cam_azimuth_deg.reshape(T, 1) + azimuth_offsets.reshape(1, P) # (T,P)
        elevation_deg = cam_elevation_deg.reshape(T, 1).repeat(1, P) # (T,P)
        distance = cam_distance.reshape(T, 1).repeat(1, P) # (T,P)
        trajectory = spherical_to_cartesian(azimuth_deg, elevation_deg, distance) # (T,P,3)

    elif trajectory_type == 'linear':

        assert(azimuth_spread_deg < 180), 'with a linear trajectory, the azimuth cannot be greater than 180 degrees'
        mag_ground_pos = cam_center[:,:2].norm(dim=-1)  # (T,) magnitude of the ground position
        dist_from_center = mag_ground_pos * np.tan(azimuth_spread_deg/2 * np.pi/180)  # (T,) distance from the center position for the first and last pulse
        start_pos = cam_center - dist_from_center.reshape(T,1) * cam_right # (T,3) starting position for the first pulse

        trajectory = torch.linspace(0,1, P, device=device).reshape(1, P, 1) * \
                     cam_right.reshape(T, 1, 3) * \
                     (2*dist_from_center).reshape(T, 1, 1) + \
                     start_pos.reshape(T, 1, 3)
        # (T,P,3) trajectory of the camera for each pulse

    else:
        raise ValueError('trajectory_type should be either circular or linear, but got %s' % trajectory_type)

    if debug:
        # plot the trajectory for debugging
        plt.figure(figsize=(6,6))
        min_z = trajectory[:,:,2].min().item()
        max_z = trajectory[:,:,2].max().item()
        plt.scatter(trajectory[0,:,0].cpu().numpy(), trajectory[0,:,1].cpu().numpy())
        plt.title('Camera Trajectory\ntrajectory trajectory_type: %s\nazimuth spread: %d degrees\nz-range: %.2f - %.2f'%(trajectory_type, azimuth_spread_deg, min_z, max_z))
        plt.xlabel('X')
        plt.ylabel('Y')
        path = get_next_path('figures/camera_trajectory.png')
        savefig(path)

    # apply trajectory gaussian noise if desired
    if trajectory_noise_var > 0:
        noise = torch.randn_like(trajectory) * np.sqrt(trajectory_noise_var)
        true_trajectory = trajectory + noise
    else:
        true_trajectory = trajectory
    perceived_trajectory = trajectory
        
    return true_trajectory, perceived_trajectory, cam_azimuth_deg


# Barker-13 code: the longest known Barker sequence, whose autocorrelation has
# uniform sidelobes of magnitude 1 against a peak of 13 (-22.3 dB).
BARKER13 = (+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1)


def _matched_filter_window(x, dz, device, dtype):
    """
    Build the effective matched-filter window w(z) = h(z) * x(z) for a transmit
    waveform x(z) sampled on a uniform grid of spacing dz, where h(z) = conj(x(-z))
    is the matched filter.  w(z) is just the autocorrelation of x and plays the same
    role as the ideal sinc window in interpolate_signal:
        s_mf(z_o) = w(z) * g(z) |_{z=z_o} = sum_r E_r w(-z_o - z_r/2).
    It is normalized to unit peak magnitude so it is directly comparable to the sinc
    window (unit peak, no range sidelobes).

    Inputs:
        x (np.ndarray): complex baseband transmit waveform x(z), centered on the grid
        dz (float): grid spacing in range units
        device: torch device of the scatter tensors
        dtype: real torch dtype of the scatter tensors
    Returns:
        window (callable): window(x) -> complex tensor, same shape as x; linearly
            interpolates w(z) at the continuous range offsets x, 0 outside support
    """
    # matched-filter output w[k] = sum_j conj(x[j]) x[j+k] (autocorrelation of x);
    # np.correlate conjugates its second argument, peak lands at the center.
    w = np.correlate(x, x, mode='full')          # length 2*len(x)-1 (odd)
    w = w / np.max(np.abs(w))                     # unit peak magnitude

    M = len(w)
    z0 = -(M - 1) / 2.0 * dz                      # range of the first grid sample
    cdtype = torch.complex64 if dtype in (torch.float32, torch.float16) else torch.complex128
    w_grid = torch.tensor(w, device=device, dtype=cdtype)  # (M,)

    def window(x):
        # linear interpolation of w(z) at the continuous offsets x, 0 outside support
        idxf = (x - z0) / dz
        i0 = torch.floor(idxf)
        frac = (idxf - i0).to(cdtype)
        i0 = i0.long()
        valid = (i0 >= 0) & (i0 < M - 1)
        i0c = i0.clamp(0, M - 2)
        w0 = w_grid[i0c]
        w1 = w_grid[i0c + 1]
        return (w0 + (w1 - w0) * frac) * valid.to(cdtype)

    return window


def make_lfm_window(spatial_bw, device, dtype, oversample=32, time_bandwidth=100.0):
    """
    Precompute the matched-filter window for an LFM (linear frequency modulation)
    chirp transmit waveform x(z) = exp(j*pi*K*z^2) of bandwidth B = spatial_bw.
    The matched-filter output is approximately a sinc of mainlobe width 1/B (matching
    the sinc window's first null) with the classic ~ -13 dB range sidelobes.

    Inputs:
        spatial_bw (float): chirp bandwidth B; sets range resolution (~1/B)
        device: torch device of the scatter tensors
        dtype: real torch dtype of the scatter tensors
        oversample (int): grid samples per resolution cell (1/B)
        time_bandwidth (float): time-bandwidth product B*T of the chirp
    Returns:
        window (callable): window(x) -> complex tensor
    """
    B = float(spatial_bw)
    dz = 1.0 / (B * oversample)              # fine grid spacing in range units
    T = time_bandwidth / B                   # pulse duration (range units)
    K = B / T                                # chirp rate, so that B = K*T
    n = int(round(T / dz)) + 1
    z = (np.arange(n) - (n - 1) / 2.0) * dz  # centered support, length n
    x = np.exp(1j * np.pi * K * z ** 2)      # inst. freq K*z spans [-B/2, B/2]
    return _matched_filter_window(x, dz, device, dtype)


def make_barker13_window(spatial_bw, device, dtype, oversample=32):
    """
    Precompute the matched-filter window for a Barker-13 phase-coded transmit
    waveform, with each of the 13 chips a rectangular sub-pulse of width 1/B
    (bandwidth ~ B = spatial_bw).  The matched-filter output has a mainlobe of width
    ~1/B and uniform range sidelobes at -22.3 dB (1/13 of the peak).

    Inputs:
        spatial_bw (float): waveform bandwidth B; sets range resolution (~1/B)
        device: torch device of the scatter tensors
        dtype: real torch dtype of the scatter tensors
        oversample (int): grid samples per chip / resolution cell (1/B)
    Returns:
        window (callable): window(x) -> complex tensor
    """
    B = float(spatial_bw)
    dz = 1.0 / (B * oversample)              # fine grid spacing in range units
    x = np.repeat(np.array(BARKER13, dtype=np.float64), oversample).astype(np.complex128)
    return _matched_filter_window(x, dz, device, dtype)


def make_transmit_waveform(window_func, spatial_bw, oversample=32, time_bandwidth=100.0):
    """
    Build the complex baseband transmit waveform x(z) for a given window, sampled on
    a common fine range grid.  This exposes the pulse whose matched-filter
    autocorrelation is used as the window in interpolate_signal, so callers can
    characterize the transmitted pulse alongside its compressed matched-filter output.

    The 'lfm' and 'barker13' branches mirror the pulses built inside make_lfm_window
    and make_barker13_window.  'sinc' and 'gaussian' have no separate transmit pulse
    in the simulator (their windows are defined directly in the range domain), so the
    window itself is returned as the waveform.  All four are placed on the same grid
    z in [-T/2, T/2], dz = 1/(B*oversample), T = time_bandwidth/B, so their spectra
    are directly comparable.

    Inputs:
        window_func (str): 'sinc', 'gaussian', 'lfm' or 'barker13'
        spatial_bw (float): waveform bandwidth B
        oversample (int): grid samples per resolution cell (1/B)
        time_bandwidth (float): time-bandwidth product B*T; sets the grid span T
    Returns:
        x (np.ndarray): complex transmit waveform on the grid, unit peak magnitude
        z (np.ndarray): range grid the waveform is sampled on (centered on 0)
        dz (float): grid spacing (so the sampling rate is 1/dz)
    """
    B = float(spatial_bw)
    dz = 1.0 / (B * oversample)              # fine grid spacing in range units
    T = time_bandwidth / B                   # common grid span (= LFM pulse duration)
    n = int(round(T / dz)) + 1
    z = (np.arange(n) - (n - 1) / 2.0) * dz  # centered support, length n

    if window_func == "sinc":
        x = np.sinc(B * z).astype(np.complex128)
    elif window_func == "gaussian":
        sigma_x = math.sqrt(math.log(2.0)) / (math.pi * B)
        x = np.exp(-0.5 * (z / sigma_x) ** 2).astype(np.complex128)
    elif window_func == "lfm":
        K = B / T                            # chirp rate, B = K*T (see make_lfm_window)
        x = np.exp(1j * np.pi * K * z ** 2)  # inst. freq K*z spans [-B/2, B/2]
    elif window_func == "barker13":
        chips = np.repeat(np.array(BARKER13, dtype=np.float64), oversample)  # 13/B wide
        x = np.zeros(n, dtype=np.complex128)
        start = (n - len(chips)) // 2        # center the 13-chip code in the grid
        x[start:start + len(chips)] = chips
    else:
        raise ValueError("window_func should be 'sinc', 'gaussian', 'lfm' or 'barker13', but got %s" % window_func)

    x = x / (np.max(np.abs(x)) + 1e-30)      # unit peak magnitude
    return x, z, dz


def interpolate_signal(scatter_z, scatter_e,# range_near, range_far,
        region_radius, sensor_distance,
        spatial_bw = 20, spatial_fs = 20,
        batch_size = None, window_func = 'sinc',
):
    """
    Simulates the received signal for the SAR algorithm given energy-range scatter.
    z is range. Z is the number of samples in the output signal.

    Inputs:
        scatter_z (...,R): the range of each scatter point
        scatter_e (...,R): the energy of each scatter point
        # range_near (float): z near to use when rendering
        # range_far (float): z far to use when rendering
        region_radius (float): the radius of the region to consider for output signal samples
        sensor_distance (...,): the distance from the sensor to the origin for each pulse
        spatial_bw (float): spatial bandwidth of the radar
        spatial_fs (float): spatial sampling frequency of the radar
        batch_size (int): number of signals to process in a batch, None means no batching
        window_func (str): window function to use ('sinc', 'gaussian', 'lfm' or
            'barker13'); 'lfm' and 'barker13' are pulse-compression matched-filter
            windows (autocorrelation of the transmit waveform)


    Returns:
        signal (tensor): simulated received signal .shape=(..., Z)
    """

    device = scatter_z.device

    # reshaping
    R = scatter_z.shape[-1]  # number of scatter points
    shape_prefix = scatter_z.shape[:-1]  # shape before the last dimension
    N = np.prod(shape_prefix)
    assert(len(scatter_z.shape) > 1), "scatter_z should have at least 2 dimensions, but got %d" % len(scatter_z.shape)
    assert(scatter_z.shape == scatter_e.shape), "scatter_z and scatter_e should have the same shape, but got %s and %s" % (scatter_z.shape, scatter_e.shape)

    # make the window function
    if window_func == "sinc":
        window = lambda x: torch.sinc(spatial_bw * x)
    elif window_func == "gaussian":
        # Gaussian pulse whose half-power (-3 dB) two-sided bandwidth equals
        # spatial_bw, matching the sinc window's occupied band [-bw/2, bw/2].
        # Frequency-domain std: |G(f)|^2 = 1/2 at f = bw/2  =>  sigma_f =
        # (bw/2)/sqrt(ln2), so the spatial-domain std is
        # sigma_x = 1/(2*pi*sigma_f) = sqrt(ln2)/(pi*bw).
        sigma_x = math.sqrt(math.log(2.0)) / (math.pi * spatial_bw)
        window = lambda x: torch.exp(-0.5 * (x / sigma_x) ** 2)
    elif window_func == "lfm":
        window = make_lfm_window(spatial_bw, device, scatter_z.dtype)
    elif window_func == "barker13":
        window = make_barker13_window(spatial_bw, device, scatter_z.dtype)
    else:
        raise ValueError("window_func should be 'sinc', 'gaussian', 'lfm' or 'barker13', but got %s" % window_func)


    # calculate the center of each spatial sample
    # first_z = int(math.ceil(range_near*spatial_fs))
    # last_z =  int(math.floor(range_far*spatial_fs))
    # sample_z = torch.arange(first_z, last_z+1, device=device, dtype=scatter_z.dtype)/spatial_fs # (Z,)
    # Z = len(sample_z)
    # the new way
    Z = int(2*region_radius*spatial_fs) + 1
    sample_z = (torch.linspace(0,1, Z, device=device, dtype=scatter_z.dtype) - 0.5 ) * (Z-1)/spatial_fs # (Z,)
    sample_z = sensor_distance.reshape(N,1) + sample_z # (N, 1) + (Z,) -> (N, Z)

    # calculate the received signal for each pulse
    # signal = sum_over_R_scatters( scatter_e * torch.sinc( spatial_bw * (scatter_z - sample_z) ) )
    # When we do the broadcasting we want the shape to be (N,R,Z) before the sum, then sum over the R dimension
    if batch_size is None:
        signal = torch.sum(
            scatter_e.reshape(N,R,1) * \
            window(scatter_z.reshape(N,R,1) - sample_z.reshape(N,1,Z)),
            dim=1
        ) # (N, Z)

    # calculate the received signal in batches to save memory
    else:
        signal = []

        reshaped_scatter_e = scatter_e.reshape(N,R,1)
        reshaped_scatter_z = scatter_z.reshape(N,R,1)
        reshaped_sample_z  = sample_z.reshape(N,1,Z)

        start = 0
        while start < N:
            end = min(start + batch_size, N)
            signal.append(
                torch.sum( reshaped_scatter_e[start:end] * \
                           window(reshaped_scatter_z[start:end] - reshaped_sample_z),
                           dim=2
            ))  # (N', Z)
            start = end

        signal = torch.cat(signal, dim=1) # (N, Z)

    # return stuff
    signal = signal.reshape(*shape_prefix, Z)  # (..., Z)
    sample_z = sample_z.reshape(*shape_prefix, Z)  # (..., Z)
    ray_normalized_signal = signal/R # normalize by the number of rays
    return ray_normalized_signal, sample_z



def resample_signal(self, signal, out_samples, dim = -1):
    """
    Resample the signal to the specified number of output samples.
    Uses linear interpolation to resample the signal.
    
    Args:
        signal (tensor): input signal to resample .shape=(..., in_samples, ...)
        out_samples (int): number of output samples to resample to
        dim (int): dimension along which to resample the signal
    
    Returns:
        resampled_signal (tensor): resampled signal .shape=(..., out_samples, ...)
    """
    # get shape information
    in_samples = signal.shape[dim]
    if in_samples == out_samples: # no need to resample when the number of samples is the same
        return signal
    assert(out_samples > 1), "out_samples should be greater than 1, but got %d" % out_samples
    in_sample_z = torch.arange(in_samples, device=signal.device, dtype=signal.dtype)  # (in_samples,)
    out_sample_z = torch.linspace(0, in_samples-1, out_samples, device=signal.device, dtype=signal.dtype)  # (out_samples,)
    shape_prefix = list(signal.shape[:dim])  # shape before the resampling dimension
    shape_suffix = list(signal.shape[dim+1:])  # shape after the resampling dimension
    n_prefix = np.prod(shape_prefix) if shape_prefix else 1  # number of elements before the resampling dimension
    n_suffix = np.prod(shape_suffix) if shape_suffix else 1  # number of elements after the resampling dimension
    # resample the signal using sinc interpolation
    # equation: resampled_signal = sum_over_inputs(torch.sinc(in_sample_z - out_sample_z) * signal)
    resampled_signal = torch.sum(
        #          (1, in_samples)                     (out_samples, 1)
        torch.sinc(in_sample_z.reshape(1,-1) - out_sample_z.reshape(-1, 1)).reshape(1,out_samples, in_samples, 1) * \
        signal.reshape(n_prefix, 1, in_samples, n_suffix),
        dim=2
    ) # (n_prefix, out_samples, n_suffix)
    
    # reshape and return
    resampled_signal = resampled_signal.reshape(shape_prefix + [out_samples] + shape_suffix)  # (shape_prefix, out_samples, shape_suffix)
    return resampled_signal  # (shape_prefix, out_samples, shape_suffix)




def apply_snr(signal, snr_db, dim=-1):
    """
    Apply a signal-to-noise ratio (SNR) to the input signal.

    Args:
        signal (torch.Tensor): The input signal tensor.
        snr_db (float): The desired SNR in decibels.
        dim (int): The dimension along which to compute the SNR.

    Returns:
        torch.Tensor: The signal with the applied SNR.
    """

    # N = Ni + Nr # real and imaginary parts are iid
    # SNR = Ps/Pn = E[|S|^2]/E[|N|^2]
    # E[|N|^2] = E[|S|^2]/SNR

    # E[|N|^2] = E[Nr^2 + Ni^2] = 2 * E[Nr^2]
    # E[|N|^2]/2 = E[Nr^2] = sigma^2
    # sigma = sqrt( E[|N|^2]/2 ) = sqrt( E[|S|^2]/(2*SNR) )

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Compute the signal power
    signal_power = torch.mean(signal.abs() ** 2, dim=dim, keepdim=True)
    noise_std_dev = torch.sqrt(signal_power / (2 * snr_linear)).cpu().numpy()

    # Generate noise
    noise = torch.tensor(
             np.random.normal(scale=noise_std_dev, size=signal.shape) + \
        1j * np.random.normal(scale=noise_std_dev, size=signal.shape)
    ).to(signal.device, dtype=signal.dtype)

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal


def load_mesh(  file_name,
                obj_raids = (1.0, 1.0, 0.9, 0.1, 1.0),
                make_ground = True,
                ground_below = True,
                ground_raids = (0.1, 0.1, 0.1, 0.9, 1.0),  # set to None to skip ground
                ground_dim = 2,
                level_with_ground = True,
                x_flip = False,
                rotate_xyz = (0.0, 0.0, 0.0),
                device = 'cuda',
                scale = None,
        ):
    '''
    Load a mesh from an obj file and compute face normals.
    Inputs:
        file_name: str - path to the obj file
        make_ground: bool - whether to add a ground plane
        obj_raids: tuple - roughness, specular, ambient for the object material
        ground_raids: tuple - roughness, specular, ambient for the ground material, set to None to skip ground
        ground_dim: int - axis index for the vertical dimension (default 2 for z-up)
        level_with_ground: bool - if True, translate the object so its bottom sits at 0 along ground_dim before adding the ground
        x_flip: bool - if True, mirror the object along the x axis before leveling
        rotate_xyz: tuple/list/tensor of 3 floats - rotation angles in degrees about x, y, z axes applied before leveling
        device: str - device to load the mesh onto
    Outputs:
        mesh: Meshes - the loaded mesh with face normals
        face_normals: (F, 3) - the face normals
        raids: (F, 5) - the material properties for each face in Reflectivity, Scatter, Absorption
    '''
    # load verts and faces
    mesh = load_objs_as_meshes([file_name], device=device)
    verts = mesh.verts_packed()  # (V, 3)
    faces = mesh.faces_packed()  # (F, 3)

    # optional scaling
    if scale is not None:
        verts = verts * scale

    # optional x-axis flip
    if x_flip:
        verts = verts * torch.tensor([-1.0, 1.0, 1.0], device=device, dtype=verts.dtype)

    # optional rotation about x, y, z axes (applied before leveling)
    rx, ry, rz = rotate_xyz
    if rx != 0.0 or ry != 0.0 or rz != 0.0:
        R = rotation_matrix_xyz(rx, ry, rz, device=device).to(verts.dtype)
        verts = verts @ R.T

    # optionally translate the object so its bottom sits at 0 along ground_dim
    if level_with_ground:
        dim_min = verts[:, ground_dim].min()
        verts[:, ground_dim] -= dim_min

    # set material properties for each face
    raids = torch.tensor(obj_raids, device=device, dtype=torch.float32).reshape(1, 5).repeat(faces.shape[0], 1)  # (F, 5)

    # add a ground if desired to the mesh
    if make_ground and ground_raids is None:
        print('WARNING: load_mesh: ground_raids is None, skipping ground addition')
        make_ground = False
    if make_ground:
        ground_size = 200
        ground_verts,ground_faces = make_big_ground( ground_size, ground_dim, ground_level = 0, max_triangle_len = ground_size/100.0, device = device )
        num_verts_before = verts.shape[0]
        verts = torch.cat([verts, ground_verts], dim=0)
        faces = torch.cat([faces, ground_faces + num_verts_before], dim=0)

        # set ground material properties
        ground_properties = torch.tensor(ground_raids, device=device, dtype=torch.float32).reshape(1, 5).repeat(ground_faces.shape[0], 1)  # (F_g, 5)
        raids = torch.cat([raids, ground_properties], dim=0)  # (F, 5)

    # calculate the normals
    face_verts = verts[faces]  # (F, 3, 3)
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2] # (F, 3), (F, 3), (F, 3)
    edge_1 = v1 - v0 # (F, 3)
    edge_2 = v2 - v0 # (F, 3)
    face_normals = torch.cross(edge_1, edge_2, dim=1)  # (F, 3)
    face_normals = torch.nn.functional.normalize(face_normals, dim=1)  # (F, 3)

    # repack the mesh with the new verts and faces
    mesh = Meshes(verts=[verts], faces=[faces])

    # add edges to the local variables of mesh for convenience in ray tracing
    mesh.edge_1 = edge_1
    mesh.edge_2 = edge_2

    # ensure material properties are valid
    raids = correct_material_properties(raids)

    # report mesh size for debugging/profiling
    try:
        print(f"load_mesh: verts={verts.shape[0].item() if hasattr(verts.shape[0],'item') else verts.shape[0]}  faces={faces.shape[0].item() if hasattr(faces.shape[0],'item') else faces.shape[0]}")
    except Exception:
        # best-effort; don't crash callers if printing fails
        print(f"load_mesh: verts={int(verts.shape[0])} faces={int(faces.shape[0])}")

    return mesh, face_normals, raids
    #      obj   (F,3)         (F,5)


def make_big_ground( size, ground_dim, ground_level = 0.0, max_triangle_len = 0.1, device = 'cpu' ):
    """
    Make a big ground plane with the specified size and ground level.
    
    Inputs:
        size (int): size of the ground plane
        ground_dim (int): dimension of the ground plane
        ground_level (float): level of the ground plane
        max_triangle_len (float): maximum length of a side in the triangle mesh
        device (str): device to use for the mesh
    Returns:
        mesh (Mesh): the generated ground mesh
    """
    N = int(math.floor(size/max_triangle_len))+1
    assert(N > 1), "N should be greater than 1, but got %d" % N

    # calculate verticies
    grid = torch.meshgrid(   torch.linspace(-size/2, size/2, N, device=device, dtype=torch.float32),
                             torch.linspace(-size/2, size/2, N, device=device, dtype=torch.float32),
                             indexing='ij')  # (N,N), (N,N)
    verts = torch.stack(grid, dim=-1)  # (N,N,2)
    verts = torch.cat((verts, torch.full((N,N,1), ground_level, device=device, dtype=torch.float32)), dim=-1)  # (V,3)
    verts = verts.reshape(-1, 3)  # (V,3)

    # calculate faces
    h,w = torch.meshgrid( torch.arange(N-1, device=device, dtype=torch.int64),  # (N-1,)
                          torch.arange(N-1, device=device, dtype=torch.int64), # (N-1,)
    indexing='ij') # (N-1,N-1), (N-1,N-1)

    # square corners: A=(h,w) B=(h+1,w) C=(h,w+1) D=(h+1,w+1)
    squares = torch.stack((h*N+w, (h+1)*N+w, h*N+w+1, (h+1)*N + w+1), dim=-1).reshape(-1,4)  # (F/2,4)
    # both triangles must wind consistently so their normals agree; A,B,C is CCW (+ground_dim),
    # so the second triangle is B,D,C (not B,C,D, which winds the opposite way and flips its normal)
    upper_triangles = squares[..., [0, 1, 2]] # (F/2,3) A,B,C
    lower_triangles = squares[..., [1, 3, 2]] # (F/2,3) B,D,C
    faces = torch.cat((upper_triangles, lower_triangles), dim=1)  # (F,3)
    faces = torch.reshape(faces, (-1, 3))  # (F,3)

    # make sure faces are within the range of vertices
    assert( torch.all(faces < verts.shape[0]) ), "faces are out of bounds"  # ensure faces are within the range of vertices

    # set the ground dim, currently it's 2
    verts = torch.roll(verts, shifts=ground_dim+1, dims=1)  # (V,3)

    # return the mesh
    return verts,faces



