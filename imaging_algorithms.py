import numpy as np
import torch
import sys
import warnings
from utils import dot_product

def projected_CBP(
    signal,
    sample_z,
    trajectory,
    spatial_fs,
    image_plane_rotation_deg = 0,
    image_width = 64,
    image_height = 64,
    image_plane_width = 1,
    image_plane_height = 1,
    batch_size = None,
    coherent_integration = True,
    wavelength = None,
):
    '''
    does some projection then runs the 2D convolutional back projection algorithm
    
    inputs:
        signal: (T,P,Z) - the signal to be back projected
        sample_z: (T,P,Z,) - the range samples
        trajectory: (T,P,3) - the location of the sensor for each pulse
        image_plane_rotation_deg: (T,) - the rotation angle of the image plane in degrees.
        spatial_fs: float - the spatial frequency sampling rate
        coherent_integration: bool - determines if phase correction is used on the samples.
            requires that the signal be complex values and wavelength is provided.
        wavelength: float or None - wavelength for coherent integration.
    outputs:
        image: (T,H,W) - the computed image
    '''
    # gather shape constants
    T,P,Z = signal.shape

    # phase correction for coherent integration
    if coherent_integration:
        if wavelength is None:
            warnings.warn(
                'coherent_integration is True but wavelength is None; '
                'skipping phase correction.'
            )
        elif not torch.is_complex(signal):
            warnings.warn(
                'coherent_integration is True but signal is not complex; '
                'skipping phase correction.'
            )
        else:
            signal = signal * torch.exp( -4j * np.pi * sample_z / wavelength )

    # calculate sqrt(w_1*2 + w_2*2) because i use it alot in this function
    forward_vector = -trajectory/torch.norm(trajectory,dim=-1,keepdim=True) # (T,P,3)
    ground_vec_mag = torch.sqrt(torch.sum(forward_vector[:,:,:2]**2,dim=-1,keepdim=True)) # (T,P,1)

    # calculate projected r from sample_z
    sample_r = sample_z - torch.linalg.vector_norm(trajectory,dim=-1,keepdim=True) # (T,P,Z)
    projected_r = sample_r / ground_vec_mag # (T,P,Z)

    # calculate the forward vector on the x-y plane
    line_vector = forward_vector[...,:2] / ground_vec_mag # (T,P,2)

    # convert to ground plane fs
    projected_fs = spatial_fs * ground_vec_mag.reshape(T,P) # (T,P,1)

    # run the 2D CBP
    sar_image = CBP_2D(
        signal,
        projected_r,
        line_vector,
        projected_fs,
        image_plane_rotation_deg = image_plane_rotation_deg,
        image_width              = image_width,
        image_height             = image_height,
        image_plane_width        = image_plane_width,
        image_plane_height       = image_plane_height,
        batch_size               = batch_size,
    ) # (T,H,W)
    
    return sar_image


def CBP_2D( pf,
            r,
            line_vector,
            interpolation_fs,
            image_plane_rotation_deg = 0,
            image_width = 64,
            image_height = 64,
            image_plane_width = 1,
            image_plane_height = 1,
            batch_size = None,
    ):
    '''
    Convolutional back projection algorithm in 2D

    inputs:
        pf: (N,P,R) - the projection functions
        r: (N,P,R) - the radial distance of each sample in the projection functions
        line_vector: (N,P,2) - the vector of the origin crossing line that each projection function corresponds to
        interpolation_fs: (N,P) - the spatial frequency sampling rate of the projection functions
        image_plane_rotation_def: (N,) - the rotation angle of the image plane in degrees. 0 degrees means the top left of the image plane is aligned with the +y and -x axes

    outputs:
        image: (N,H,W) - the computed image

    Dimensions:
        N: number of images
        P: number of projection functions
        R: number of radial samples per projection function
        H: image height
        W: image width
        T: number of target image pixels (H*W)
    '''
    # Get shapes
    N,P,R = pf.shape
    H = image_height
    W = image_width
    T = H * W
    device = pf.device

    # filter with |r| in frequency domain (equation 2.30)
    pf_freq = torch.fft.fftshift(torch.fft.fft(pf, dim=-1), dim=-1)  # (N,P,R)
    filtered_pf_freq = pf_freq * torch.abs(r.reshape(N,P,R)) # (N,P,R)
    filtered_pf = torch.fft.ifft(torch.fft.ifftshift(filtered_pf_freq, dim=-1), dim=-1)  # (N,P,R)

    # create grid of target image cooordinates on the ground plane
    x_coord,y_coord = torch.meshgrid(   torch.linspace(-image_plane_width/2, image_plane_width/2, image_width, device=device, dtype=r.dtype),
                                        torch.linspace(image_plane_height/2 , -image_plane_height/2 , image_height , device=device, dtype=r.dtype),
                                        indexing='xy')  # (H,W)
    coord_grid = torch.stack((x_coord, y_coord), dim=-1).float() # (H,W,2)

    # rotate the image plane according to the desired rotation angle
    rotation_rad = image_plane_rotation_deg * (np.pi / 180.0)  # convert to radians
    rotation_matrix = torch.stack([
        torch.cos(rotation_rad), -torch.sin(rotation_rad), torch.sin(rotation_rad), torch.cos(rotation_rad)
    ], dim=-1) # (N,4)
    coord_grid = rotation_matrix.reshape(N,1,2,2) @ coord_grid.reshape(1,T,2,1)  # (N,T,2,1)

    # interpolate pixel coordinated projected onto the filtered signal
    line_vector = torch.nn.functional.normalize(line_vector, dim=-1) # (N,P,2)
    r_coord = torch.sum(line_vector[...,:2].reshape(N,P,1,2) * coord_grid.reshape(N,1,T,2), dim=-1)  # (N,P,T)

    if batch_size is None:
        interpolated_r_points = torch.sum(
            filtered_pf.reshape(N,P,1,R) *
            torch.sinc( interpolation_fs.reshape(N,P,1,1) * (r_coord.reshape(N,P,T,1) - r.reshape(N,P,1,R)) ), # (N,P,T,R)
            dim=-1
        ) # (N,P,T)
    else:
        interpolated_r_points = torch.zeros(N, P, T, dtype=filtered_pf.dtype, device=device)
        for t_start in range(0, T, batch_size):
            t_end = min(t_start + batch_size, T)
            bT = t_end - t_start
            r_coord_batch = r_coord[:, :, t_start:t_end]  # (N,P,bT)
            interpolated_r_points[:, :, t_start:t_end] = torch.sum(
                filtered_pf.reshape(N,P,1,R) *
                torch.sinc( interpolation_fs.reshape(N,P,1,1) * (r_coord_batch.reshape(N,P,bT,1) - r.reshape(N,P,1,R)) ), # (N,P,bT,R)
                dim=-1
            ) # (N,P,bT)
    
    # integrate over theta (eqation 2.31)
    image = torch.sum(interpolated_r_points, dim=1) / (4*np.pi**2)  # (N,T)

    # reshape and convert to real-valued images
    image = image.reshape(N,image_height,image_width) # (N,H,W)
    return torch.sqrt(image.real**2 + image.imag**2)  # (N,H,W)


def side_scan_ground_plane_image(
        signal,
        range_bins,
        ping_offsets,
        sensor_pose,
        image_width = 64,
        image_height = 64,
        image_plane_width = 1,
        image_plane_height = 1,
    ):
    '''
    Project a side scan's pings onto the ground plane.

    Lining the pings up as columns gives an image in (range, along-track), which is not a
    picture of the seafloor: range compresses towards nadir and stretches at the far edge of
    the swath. This resamples them onto a flat patch of seafloor centered on the origin, so a
    pixel is a place rather than a time of arrival.

    Every ping shares one orientation and differs only in where it sits along the track, so one
    change of basis into the sensor frame gives a ground pixel both of its lookups: the cross
    range picks the ping abeam of it, and the range that ping measures picks the sample.

    inputs:
        signal (T,P,Z): received signal of each ping, real or complex
        range_bins (T,Z): one-way range of each sample, one evenly spaced window for all pings
        ping_offsets (P,): along-track offset of each ping, evenly spaced
        sensor_pose (T,4,4) or (4,4): columns (right, up, forward, mean sensor position), the
            orientation every ping shares. right is the track direction and lies flat
        image_width/image_height (int): pixels across and down the ground patch
        image_plane_width/image_plane_height (float): extent of the patch in world units, along
            the ground right and ground forward directions

    outputs:
        image (T,H,W): the ground plane image, near range at the bottom row
        row_coords (H,): ground forward offset of each row, far range first
        col_coords (W,): ground right offset of each column, low ping offset first

    Dimensions:
        T: number of tracks
        P: number of pings per track
        Z: number of range samples per ping
        H: image height
        W: image width
    '''
    # get shapes
    T,P,Z = signal.shape
    H = image_height
    W = image_width
    device = signal.device
    dtype = range_bins.dtype

    assert P > 1, 'ground plane projection interpolates between pings, so it needs at least 2'
    assert Z > 1, 'ground plane projection interpolates between range samples, so it needs at least 2'

    # pose vectors shared by every ping
    sensor_pose = sensor_pose.reshape(T,4,4)
    right       = sensor_pose[:, :3, 0]  # (T,3)
    up          = sensor_pose[:, :3, 1]  # (T,3)
    forward     = sensor_pose[:, :3, 2]  # (T,3)
    mean_sensor_position = sensor_pose[:, :3, 3]  # (T,3)

    # ground axes: right already lies flat, and z cross right completes it pointing away from the track
    world_up       = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=right.dtype)  # (3,)
    ground_right   = torch.nn.functional.normalize(right, dim=-1)  # (T,3)
    ground_forward = torch.nn.functional.normalize(
        -torch.linalg.cross(ground_right, world_up.expand_as(ground_right)), dim=-1)  # (T,3)

    # world position of every pixel, with row 0 at the far edge so near range lands at the bottom
    r = torch.linspace(0, 1, H, device=device, dtype=dtype)  # (H,)
    c = torch.linspace(0, 1, W, device=device, dtype=dtype)  # (W,)
    row_coords = (0.5 - r) * image_plane_height  # (H,)
    col_coords = (c - 0.5) * image_plane_width   # (W,)
    pixel_xyz = (row_coords.reshape(1,H,1,1) * ground_forward.reshape(T,1,1,3) +
                 col_coords.reshape(1,1,W,1) * ground_right.reshape(T,1,1,3))  # (T,H,W,3)

    # into the sensor frame as [cross range, height off boresight, forward range]
    basis = torch.stack([right, up, forward], dim=-1)  # (T,3,3), columns
    relative_position = (pixel_xyz - mean_sensor_position.reshape(T,1,1,3)).reshape(T,H*W,3,1)  # (T,H*W,3,1)
    sensor_xyz = (torch.linalg.inv(basis).reshape(T,1,3,3) @ relative_position).reshape(T,H,W,3)  # (T,H,W,3)
    cross_range = sensor_xyz[..., 0]  # (T,H,W) which ping is abeam of the pixel
    # the abeam ping sits at the pixel's cross range, so its range is the other two components
    slant_range = torch.linalg.norm(sensor_xyz[..., 1:], dim=-1)  # (T,H,W)

    # fractional sample and ping index of each pixel
    range_step = (range_bins[:, 1] - range_bins[:, 0]).reshape(T,1,1)  # (T,1,1)
    ping_step  = ping_offsets[1] - ping_offsets[0]  # ()
    sample_index = (slant_range - range_bins[:, :1].reshape(T,1,1)) / range_step  # (T,H,W)
    ping_index   = (cross_range - ping_offsets[0]) / ping_step  # (T,H,W)

    # bilinear interpolation over the (P,Z) grid, zero outside it
    z0 = torch.floor(sample_index)  # (T,H,W)
    p0 = torch.floor(ping_index)    # (T,H,W)
    z_weight = sample_index - z0  # (T,H,W)
    p_weight = ping_index - p0    # (T,H,W)
    z0 = z0.long()
    p0 = p0.long()
    flat_signal = signal.reshape(T,P*Z)  # (T,P*Z)
    image = torch.zeros(T, H, W, dtype=signal.dtype, device=device)  # (T,H,W)
    for dz in (0, 1):
        for dp in (0, 1):
            z_index = z0 + dz  # (T,H,W)
            p_index = p0 + dp  # (T,H,W)
            in_grid = (z_index >= 0) & (z_index < Z) & (p_index >= 0) & (p_index < P)  # (T,H,W)
            weight  = (z_weight if dz else 1 - z_weight) * (p_weight if dp else 1 - p_weight) * in_grid  # (T,H,W)
            corner  = torch.gather(
                flat_signal, 1,
                (p_index.clamp(0, P-1) * Z + z_index.clamp(0, Z-1)).reshape(T,H*W)
            ).reshape(T,H,W)  # (T,H,W)
            image = image + weight.to(corner.dtype) * corner  # (T,H,W)

    return image, row_coords, col_coords


def strip_map_imaging(  signal,
                        wavelength,
                        trajectory,
                        sample_dist,
                        interpolation_fs,
                        planar_wave = True,
                        attenuation_coeff = 0,
                        image_plane_rotation_deg = 0,
                        image_width = 64,
                        image_height = 64,
                        image_plane_width = 1,
                        image_plane_height = 1,
    ):
    '''
    Strip map imaging algorithm, we only render the ground 
    plane and assume the image plane is about the origin.

    reflectivity at point x is given by 
    avg_over_pulses{ signal(pulse, distance_to_x) * exp(attenuation_coeff * 2 * distance_to_x) * exp(-j*4*pi/wavelength*distance_to_x) }
    we need to interpolate the signal at distance_to_x for each pulse's signal

    inputs:
        signal: (N,P,D) - the signal to be back projected
        wavelength: - the wavelength
        attenuation_coeff: - the attenuation coefficient of the medium
        trajectory: (N,P,3) - the trajectory of the sensor
        sample_dist: (N,P,D) - the distance samples
        interpolation_fs: float - the spatial frequency sampling rate
        image_plane_rotation_def: (N,) - the rotation angle of the image plane in degrees. 0 degrees means the top left of the image plane is aligned with the +y and -x axes

    outputs:
        image: (N,H,W) - the computed image

    Dimensions:
        N: number of images
        P: number of pulses
        D: number of distance samples per pulse
        H: image height
        W: image width
        T: number of target image pixels (H*W)
    '''
    # get shapes
    N,P,D = signal.shape
    H = image_height
    W = image_width
    T = H * W
    device = signal.device

    # create grid of target image cooordinates on the ground plane
    dtype = sample_dist.dtype
    x_coord,y_coord = torch.meshgrid(   torch.linspace(-image_plane_width/2, image_plane_width/2, image_width, device=device, dtype=dtype),
                                        torch.linspace(image_plane_height/2 , -image_plane_height/2 , image_height , device=device, dtype=dtype),
                                        indexing='xy')  # (H,W)
    coord_grid = torch.stack((x_coord, y_coord), dim=-1).float() # (H,W,2)

    # rotate the image plane according to the desired rotation angle
    rotation_rad = image_plane_rotation_deg * (np.pi / 180.0)  # convert to radians
    rotation_matrix = torch.stack([
        torch.cos(rotation_rad), -torch.sin(rotation_rad), torch.sin(rotation_rad), torch.cos(rotation_rad)
    ], dim=-1) # (N,4)
    coord_grid = rotation_matrix.reshape(N,1,2,2) @ coord_grid.reshape(1,T,2,1)  # (N,T,2,1)

    # compute distance from each pulse to each pixel
    coord_grid = torch.cat([coord_grid.reshape(N,T,2), torch.zeros((N,T,1), device=device, dtype=coord_grid.dtype)], dim=-1)  # (N,T,3)
    if planar_wave:
        mag_trajectory = torch.norm(trajectory, dim=-1, keepdim=True)  # (N,P,1)
        forward_vector = -trajectory / mag_trajectory  # (N,P,3)
        distance_to_pixel = mag_trajectory + dot_product( coord_grid.reshape(N,1,T,3), forward_vector.reshape(N,P,1,3) )  # (N,P,T)
    else:
        distance_to_pixel = torch.norm( trajectory.reshape(N,P,1,3) - coord_grid.reshape(N,1,T,3), dim=-1 )  # (N,P,T)

    # interpolate signal at distance_to_pixel
    signal_at_distance_to_pixel = torch.sum(  signal.reshape(N,P,1,D) * \
                                    torch.sinc( interpolation_fs * ((distance_to_pixel.reshape(N,P,T,1) - sample_dist.reshape(N,P,1,D)) )), # (N,P,T,D)
                                    dim=-1
                                ) # (N,P,T)
    
    # compute estimate of reflectivity
    reflectivity_estimate = torch.mean( signal_at_distance_to_pixel * \
                                        # distance_to_pixel**2 * \
                                        torch.exp(
                                            2*attenuation_coeff *distance_to_pixel - \
                                            1j*4*3.14159265358979323846264338427950288*distance_to_pixel/wavelength
                                        )
                                    , dim=1)  # (N,T)

    # reshape and convert to real-valued images
    image = reflectivity_estimate.reshape(N,image_height,image_width) # (N,H,W)
    return torch.sqrt(image.real**2 + image.imag**2)  # (N,H,W)
