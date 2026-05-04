import itertools

import matplotlib.pyplot as plt
import PIL
import os
import torch
import numpy as np
import imageio

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings, 
    MeshRasterizer,  
)

def gpu_mem(device=None):
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    return f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB"


def dot_product(a, b, dim = -1, keepdim=False):
    '''
    compute the dot product between two tensors along a specified dimension, and return a tensor with that dimension removed.
    For example, if a and b are both (..., 3) and dim=-1, then this function will return a tensor of shape (...) containing the dot product of the last dimension of a and b.
    '''
    return torch.sum(a * b, dim=dim, keepdim=keepdim)


def correct_material_properties(properties):
    '''
    make sure that the material properties are valid. Assume it is a tensor of shape (..., 5) where the 5 is
    r: reflectivity,
    a: absorption,
    i: directional scattering portion,
    d: diffuse scattering portion,
    s: overall scattering, 

    r+s+a = 1
    i+d = 1

    if these conditions are not met, we will normalize the properties to make them valid.
    Throw a warning if the properties are not valid, but still return a corrected version of the properties that is valid.
    '''
    # get the properties
    assert properties.shape[-1] == 5, "Properties should have shape (..., 5)"
    r, a, i, d, s = properties[..., 0], properties[..., 1], properties[..., 2], properties[..., 3], properties[..., 4]

    # make sure r,s,a adds to 1
    if not torch.allclose(r + s + a, torch.ones_like(r)):
        print("Warning: The sum of reflectivity, scattering, and absorption is not 1. Normalizing these properties.")
        total = r + s + a
        r = r / total
        s = s / total
        a = a / total

    # make sure i,d adds to 1
    if not torch.allclose(i + d, torch.ones_like(i)):
        print("Warning: The sum of directional and diffuse scattering is not 1. Normalizing these properties.")
        total = i + d
        i = i / total
        d = d / total

    # stack and return
    return torch.stack((r, a, i, d, s), dim=-1)




def spherical_to_cartesian(azimuth_deg, elevation_deg, distance):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters:
    - azimuth: (...,) The azimuth angle in degrees.
    - elevation: (...,) The elevation angle in degrees.
    - distance: (...,) The distance from the origin.

    Returns:
    - cartesian: (...,3) The tensor containing the Cartesian coordinates (x, y, z).
    """
    azimuth_rad = azimuth_deg * np.pi / 180
    elevation_rad = elevation_deg * np.pi / 180

    x = distance * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y = distance * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    z = distance * torch.sin(elevation_rad)

    cartesian = torch.stack((x, y, z), dim=-1)
    return cartesian


def cartesian_to_spherical(cartesian):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
    - cartesian: (...,3) The tensor containing the Cartesian coordinates (x, y, z).

    Returns:
    - azimuth_deg: (...,) The azimuth angle in degrees.
    - elevation_deg: (...,) The elevation angle in degrees.
    - distance: (...,) The distance from the origin.
    """
    x = cartesian[..., 0]
    y = cartesian[..., 1]
    z = cartesian[..., 2]

    distance = torch.sqrt(x**2 + y**2 + z**2)

    azimuth_rad = torch.atan2(y, x)
    elevation_rad = torch.atan2(z, torch.sqrt(x**2 + y**2))

    azimuth_deg = azimuth_rad * 180 / np.pi
    elevation_deg = elevation_rad * 180 / np.pi

    return azimuth_deg, elevation_deg, distance



def test_spherical_cartesian_consistency(num_points=100000, tol=1e-3):
    """
    Test spherical_to_cartesian and cartesian_to_spherical by converting
    random Cartesian points to spherical coordinates and back.

    Parameters
    ----------
    num_points : int
        Number of random points to generate.
    tol : float
        Allowed reconstruction error.
    """

    # Generate random 3D points across a wide range
    cartesian_original = torch.randn(num_points, 3) * 100

    # Convert to spherical
    azimuth_deg, elevation_deg, distance = cartesian_to_spherical(cartesian_original)

    # Convert back to Cartesian
    cartesian_reconstructed = spherical_to_cartesian(
        azimuth_deg,
        elevation_deg,
        distance
    )

    # Compute reconstruction error
    error = torch.norm(cartesian_original - cartesian_reconstructed, dim=-1)
    max_error = error.max()
    mean_error = error.mean()

    print(f"Tested {num_points} random points")
    print(f"Max reconstruction error: {max_error}")
    print(f"Mean reconstruction error: {mean_error}")

    if max_error < tol:
        print("PASS: spherical/cartesian conversions are consistent")
        return True
    else:
        print("FAIL: reconstruction error exceeds tolerance")
        return False


def savefig(path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_next_path(path):
    """
    Get the next available path by appending a number to the base path.
    If the path already exists, it will increment the number until it finds an available one.
    """
    split_path = path.split('.')
    assert len(split_path) > 1, "Path must have an extension to append a number."
    base_path = '.'.join(split_path[:-1])
    extension = split_path[-1]
    
    i = 0
    while True:
        new_path = '%s_%02d.%s' % (base_path, i, extension)
        if not os.path.exists(new_path):
            return new_path
        i += 1





def extract_pose_info(target_poses, format='srn_cars'):
    '''
    Extracts camera position and orientation from the target poses.
    Angles are in degrees.

    inputs:
        target_poses (...,4,4): the camera poses in world coordinates
    outputs:
        see the return statement below
    '''
    if format == 'srn_cars':
        # extract vectors
        cam_right     = target_poses[..., :3, 0] # (...,3)
        cam_up        = target_poses[..., :3, 1] # (...,3)
        cam_forward   = target_poses[..., :3, 2] # (...,3)
        cam_center    = target_poses[..., :3, 3] # (...,3)

        # calculate distance, azimuth, and elevation
        cam_distance  = torch.norm(cam_center, dim=-1)  # (...,)
        cam_azimuth   = torch.where(cam_right[...,0] < 0, torch.acos(cam_right[...,1]), 2*np.pi - torch.acos(cam_right[...,1]))  # (...)
        cam_elevation = torch.asin(cam_center[..., 2] / cam_distance)  # (...)
        

    else:
        raise NotImplementedError("Unknown format for extract_pose_info(): %s" % format)

    return cam_center, cam_right, cam_up, cam_forward, cam_distance, cam_elevation*180/np.pi, cam_azimuth*180/np.pi


def generate_pose_mat(azimuth,elevation,distance,device='cpu',format='srn_cars'):
    '''
    Generates a camera pose matrix from azimuth, elevation, and distance.

    inputs:
        azimuth   float or (...,): azimuth angle in radians
        elevation float or (...,): elevation angle in radians
        distance  float or (...,): distance from the origin
        device    (str): device to use for the tensors, e.g., 'cpu'
    outputs:
        pose_mat  (...,4,4): the camera pose matrix in world coordinates
    '''
    type_conv = lambda x: torch.tensor(x, device=device, dtype=torch.float32) if type(x) is not torch.Tensor else x
    azimuth   = 3.14159/180 * type_conv(azimuth)
    elevation = 3.14159/180 * type_conv(elevation)
    distance  = type_conv(distance)

    if format == 'srn_cars':

        # compute right, up, and forward vectors
        u = torch.stack([
            -torch.sin(azimuth),
            torch.cos(azimuth),
            torch.zeros_like(azimuth),
        ], dim = -1) # (...,3)
        v = torch.stack([
            -torch.cos(azimuth) * torch.sin(elevation),
            -torch.sin(azimuth) * torch.sin(elevation),
             torch.cos(elevation)
        ],dim = -1) # (...,3)
        w = -torch.stack([
            torch.cos(azimuth) * torch.cos(elevation),
            torch.sin(azimuth) * torch.cos(elevation),
            torch.sin(elevation),
        ], 
        dim = -1) # (...,3)

        # Normalize all directions
        u = torch.nn.functional.normalize(u, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)
        w = torch.nn.functional.normalize(w, dim=-1)


        # Create the pose matrix 
        # [[u, v, w, c]
        #  [0, 0, 0, 1]]
        c = -distance * w  # (...,3) # camera center
        pose_mat = torch.zeros((*w.shape[:-1], 4, 4), device=device)  # (...,4,4)
        pose_mat[..., :3, 0] = u  
        pose_mat[..., :3, 1] = v  
        pose_mat[..., :3, 2] = w  
        pose_mat[..., :3, 3] = c  
        pose_mat[...,  3, 3] = 1.0

        return pose_mat

    else:
        raise NotImplementedError("Unknown format for generate_pose_mat(): %s" % format)


def plot_angular_response():
    '''
    plotting models for returned energy
    '''

    r = 1 # reflectivity
    s = 1 # directional dependent scattering
    a = 1 # absorbtion
    d = 0 # diffusion scattering
    i = 1 # directional scattering intensity
    energy_in = 1

    theta_deg = np.linspace(0,180,1000)
    theta = theta_deg*np.pi/180

    for i in [.2,.5,1,2,5,10,20,50,100]:
        # energy_returned = energy_in * (s*(np.cos(theta)**i) + d)
        # energy_returned = energy_in * (s*(np.cos(theta)+1)**i + d)
        energy_returned = energy_in * (s*(np.cos(theta/2))**i + d)
        plt.plot(theta_deg, energy_returned, label='$\\alpha$=%.1f'%i)
    plt.xlabel('Reflected angular difference (degrees)')
    plt.ylabel('Energy returned')
    plt.legend()
    plt.grid()
    plt.title('Angular response of returned energy for different scattering intensities\n(s=%d, d=%d, r=%d, a=%d)'%(s,d,r,a))
    savefig(get_next_path('figures/angular_response.png'))


def numerically_analyze_directional_scattering(alpha=100):
    '''
    I want to compute how much i should multiply the returned energy ray by such that we satisfy the conservation of energy.
    This means the diffuse and the directional scattering

    We need to numerically compute the integral of returned energy over a hemisphere for different values of alpha in np.cos(theta/2)**alpha
    the integral may changer with differnt reflected ray directions, so we need to compute it for a bunch of az_r, el_r, i combinations
    Then we are going to find a good approximation for the multiplier using dot product and some kind of nonlinear function, like an SVM
    '''

    # integrate over many small rays in the hemisphere
    n_numer = 1000
    az = np.linspace(0,2*np.pi,n_numer).reshape(-1, 1, 1) # (n_numer, 1, 1)
    el = np.linspace(0,np.pi/2,n_numer).reshape( 1,-1, 1) # (1, n_numer, 1)

    # select the values of outgoing ray direction, and alpha
    # actually, lets fix alpha
    n_r = 100
    az_r = 0
    el_r = np.linspace(0,np.pi/2,n_r).reshape( 1, 1,-1) # (1, 1, n_r)

    # compute the integral for each az_r, el_r combination, without a for loop...

    # useful intermediate function to convert azimuth and elevation to a unit vector
    def vec(azimuth, elevation):
        x = np.cos(elevation) * np.cos(azimuth)
        y = np.cos(elevation) * np.sin(azimuth)
        z = np.sin(elevation) + np.zeros_like(azimuth)  # add zeros to ensure the shape is correct for broadcasting
        return np.stack((x, y, z), axis=-1)

    # (n_r, n_r)
    total_energy =  np.sum(
                        np.sum(
                            np.sqrt((1+
                                np.sum( vec(az,el) * vec(az_r,el_r),axis=-1) # (n_numer, n_numer, n_r)
                            )/2)**alpha * np.cos(el)
                            , axis= 0
                        )
                        ,axis=0
                    ) * (2*np.pi/n_numer) * (np.pi/2/n_numer) # need to multiply by the width of each tiny rectangle of integration
    
    # plot the total_energy as a heat map like confusion matricies are plotted
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(el_r[0,0,:]*180/np.pi, total_energy)
    plt.xlabel('$\\vec{e}_r$', fontsize=18)
    plt.ylabel('$\\oiint \\cos({\\theta/2})^\\alpha d \\Omega$', fontsize=18)
    plt.title('$\\alpha$=%d'%alpha, fontsize=18)
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(np.cos(np.pi/2 - el_r[0,0,:]), total_energy)
    plt.xlabel('$\\cos(90\\degree - \\vec{e}_r) = -\\vec{u}_{in} \\cdot \\vec{n}$', fontsize=18)
    plt.ylabel('$\\oiint \\cos({\\theta/2})^\\alpha d \\Omega$', fontsize=18)
    plt.title('$\\alpha$=%d'%alpha, fontsize=18)
    plt.grid()
    savefig('figures/total_returned_energy.png')

    print('total_energy: ', total_energy)

    # so the azimuth of the reflected ray doesn't matter, but the elevation does, which makes sense because the scattering is symmetric around the normal direction.
    # we can simply fit a polynomial to the relationship between the elevation of the reflected ray and the total returned energy, and use that as our multiplier for the returned energy ray.

    x = np.cos(np.pi/2 - el_r[0,0,:]) # (n_r,)
    y = total_energy # (n_r,)

    for order in [1,2,3,4,5]:
        coeffs = np.polyfit(x, y, order)
        print('Coeffs for order %d: '%order, coeffs)
        poly = np.poly1d(coeffs)
        plt.plot(x, poly(x), label='order %d'%order)
    plt.plot(x, y, label='numerical integral', color='black', linewidth=2)
    plt.xlabel('$\\cos(90\\degree - \\vec{e}_r) = -\\vec{u}_{in} \\cdot \\vec{n}$', fontsize=18)
    plt.ylabel('$\\oiint \\cos({\\theta/2})^\\alpha d \\Omega$', fontsize=18)
    plt.title('$\\alpha$=%d'%alpha, fontsize=18)
    plt.legend()
    plt.grid()
    savefig('figures/total_returned_energy_fit.png')

    # make sure i get the equation right in pytorch
    order = 1
    coeffs = np.polyfit(x, y, order)

    # print the function so i can copy it into my code
    print('Directional scatter multiplier function for alpha=%d:'%alpha)
    print('def directional_scatter_polynomial_alpha%d(cos_90_minus_elevation):'%alpha)
    s = '    return '
    for i in range(order):
        s += '%.8f*cos_90_minus_elevation**%d + '%( coeffs[i], order-i )
    s += '%.8f'%coeffs[order]
    print(s)


def directional_scatter_polynomial_alpha100(cos_90_minus_elevation):
    '''
    this is the function we will use to multiply the returned energy ray by, to ensure conservation of energy when alpha=100
    '''
    return  -0.27239384*cos_90_minus_elevation**4 + \
            0.94296234*cos_90_minus_elevation**3 + \
            -1.19504825*cos_90_minus_elevation**2 + \
            0.65042818*cos_90_minus_elevation + \
            0.12070119
def directional_scatter_polynomial_alpha5(cos_90_minus_elevation):
    return 1.46792856*cos_90_minus_elevation**1 + 1.81188202


    

if __name__ == '__main__':
    numerically_analyze_directional_scattering(alpha=1)

  