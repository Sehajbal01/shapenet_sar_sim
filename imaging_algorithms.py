import numpy as np
import torch

def projected_CBP(
    signal,
    sample_z,
    forward_vector,
    cam_azimuth,
    cam_distance,
    spatial_fs,
    image_width = 64,
    image_height = 64,
    image_plane_width = 1,
    image_plane_height = 1,
):
    '''
    does some projection then runs the 2D convolutional back projection algorithm
    
    inputs:
        signal: (T,P,Z) - the signal to be back projected
        sample_z: (Z,) - the range samples
        forward_vector: (T,P,3) - the forward vector for each ray
        cam_azimuth: (T,) - the azimuth angle of the camera
        cam_distance: (T,) - the distance of the camera from the origin
        spatial_fs: float - the spatial frequency sampling rate
    outputs:
        image: (T,H,W) - the computed image
    '''
    # gather shape constants
    T,P,Z = signal.shape

    # calculate sqrt(w_1*2 + w_2*2) because i use it alot in this function
    ground_vec_mag = torch.sqrt(torch.sum(forward_vector[:,:,:2]**2,dim=-1,keepdim=True)) # (T,P,1)

    # calculate projected r from sample_z
    sample_r = sample_z.reshape(1,1,Z) - cam_distance.reshape(T,1,1) # (T,1,Z)
    projected_r = sample_r / ground_vec_mag # (T,P,Z)

    # calculate the forward vector on the x-y plane
    line_vector = forward_vector[...,:2] / ground_vec_mag # (T,P,2)

    # convert azimuth to image plane rotation
    image_plane_rotation = cam_azimuth + 90

    # convert to ground plane fs
    projected_fs = spatial_fs * ground_vec_mag.reshape(T,P) # (T,P,1)

    # run the 2D CBP
    sar_image = CBP_2D(
        signal, 
        projected_r,
        line_vector,
        projected_fs,
        image_plane_rotation_deg = image_plane_rotation,
        image_width              = image_width,
        image_height             = image_height,
        image_plane_width        = image_plane_width,
        image_plane_height       = image_plane_height,
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
    r_coord = torch.sum(line_vector[...,:2].reshape(N,P,1,2) * coord_grid.reshape(N,1,T,2), dim=-1)  # (N,P,T,1)
    interpolated_r_points = torch.sum(  filtered_pf.reshape(N,P,1,R) * \
                                        torch.sinc( interpolation_fs.reshape(N,P,1,1) * (r_coord.reshape(N,P,T,1) - r.reshape(N,P,1,R)) ), # (N,P,T,R)
                                        dim=-1
                                    ) # (N,P,T)
    
    # integrate over theta (eqation 2.31)
    image = torch.sum(interpolated_r_points, dim=1) / (4*np.pi**2)  # (N,T)

    # reshape and convert to real-valued images
    image = image.reshape(N,image_height,image_width) # (N,H,W)
    return torch.sqrt(image.real**2 + image.imag**2)  # (N,H,W)


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
        sample_dist: (D,) - the distance samples
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
        distance_to_pixel = torch.sum( trajectory.reshape(N,P,1,3) * forward_vector.reshape(N,P,1,3), dim=-1 )  - mag_trajectory # (N,P,T)
        print('distance_to_pixel.shape: ', distance_to_pixel.shape)
    else:
        distance_to_pixel = torch.norm( trajectory.reshape(N,P,1,3) - coord_grid.reshape(N,1,T,3), dim=-1 )  # (N,P,T)

    # interpolate signal at distance_to_pixel
    signal_at_distance_to_pixel = torch.sum(  signal.reshape(N,P,1,D) * \
                                    torch.sinc( interpolation_fs * ((distance_to_pixel.reshape(N,P,T,1) - sample_dist.reshape(1,1,1,D)) )), # (N,P,T,D)
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