import sys
import torch
from pytorch3d.io import load_objs_as_meshes
from moller_trumbore import ray_triangle_intersect, load_obj_vertices_faces


def extract_pose_info(target_poses):
    '''
    Extracts camera position and orientation from the target poses.
    inputs:
        target_poses (T,4,4): the camera poses in world coordinates
    outputs:
        see the return statement below
    '''
    cam_center    = target_poses[:, :3, 3] # (T,3)
    cam_right     = target_poses[:, :3, 0]  # (T,3)
    cam_up        = target_poses[:, :3, 1]     # (T,3)
    cam_forward   = target_poses[:, :3, 2] # (T,3)
    cam_distance  = torch.norm(cam_center, dim=-1, keepdim=True) # (T,)
    cam_elevation = torch.asin(cam_center[:, 2:3] / cam_distance) # (T,)
    cam_azimuth   = torch.acos(cam_center[:, 0:1] / (cam_distance * torch.cos(cam_elevation))) # (T,)
    cam_azimuth   = torch.where(cam_center[:, 1:2] < 0, 2 * torch.pi - cam_azimuth, cam_azimuth) # (T,)
    return cam_center, cam_right, cam_up, cam_forward, cam_distance, cam_elevation, cam_azimuth
    

def generate_ray_grid(azimuth, elevation, side_len, distance, num_ray_side):
    '''
    Create a grid of rays on a square plane facing the origin.
    The center of the square is determined by azimuth,elevation,distance.

    inputs:
        azimuth (...): angle of the sensor from the x to y axis
        elevation (...): angle of the sensor from the origin to z axis
        side_len (float): length of the side of the square of rays
        distance (float): distance form the origin for the
        num_ray_side (int): the number of rays on each side of the triangle, yielding num_ray_side**2 total rays

    outputs:
        ray_origins (...,R,3):
        ray_directions (...,R,3):

    '''
    device = azimuth.device
    shape_prefix = azimuth.shape
    assert(azimuth.shape == elevation.shape), 'Incompatible shapes, got {} and {}'.format(azimuth.shape, elevation.shape)
    azimuth = azimuth.reshape(-1)  # (N,)
    elevation = elevation.reshape(-1)  # (N,)
    N = azimuth.shape[0]  # number of rays
    R = num_ray_side ** 2  # number of rays per pulse

    # Figure out forward/right/up of the sensor
    u = torch.stack([
        -torch.sin(azimuth), 
        torch.cos(azimuth),
        torch.zeros_like(azimuth)], dim=-1) # (N,3)
    v = torch.stack([
        -torch.cos(azimuth) * torch.sin(elevation),
        -torch.sin(azimuth) * torch.sin(elevation),
         torch.cos(elevation)
    ], dim=-1) # (N,3)
    w = -torch.stack([
        torch.cos(azimuth) * torch.cos(elevation),
        torch.sin(azimuth) * torch.cos(elevation),
        torch.sin(elevation),
    ], dim=-1)
    
    # Normalize all directions
    u = torch.nn.functional.normalize(u, dim=-1) # (N,3)
    v = torch.nn.functional.normalize(v, dim=-1) # (N,3)
    w = torch.nn.functional.normalize(w, dim=-1) # (N,3)

    # Make a square grid centered on the camera, in the u-v plane
    lin = torch.linspace(-0.5*side_len, 0.5*side_len, num_ray_side, device=device) # (r,)
    grid_x, grid_y = torch.meshgrid(lin, lin, indexing='ij') # (r,r) (r,r)
    grid_x = grid_x.reshape(-1)  # (R,)
    grid_y = grid_y.reshape(-1)  # (R,)
    offsets = grid_x.reshape(1,R,1) * u.reshape(N,1,3) + grid_y.reshape(1,R,1) * v.reshape(N,1,3) # (N,R,3)

    # Move the grid to the sensor location
    sensor_loc = -w * distance # (N,3)
    ray_origins = sensor_loc.reshape(N, 1, 3) + offsets # (N,R,3)

    # All rays go forward in the same direction
    ray_directions = w.reshape(N, 1, 3).tile(1, R, 1) # (N,R,3)

    return ray_origins.reshape(*shape_prefix, R, 3), ray_directions.reshape(*shape_prefix, R, 3)


def get_range_and_energy(ray_origins, ray_directions, object_filename, alpha_1=0.9, alpha_2=0.1):
    '''
    finds where each ray hits the object and returns the range and energy of each ray
    only returns rays that hit the object
    There are R rays and R' rays that hit the object, so the output is (...,R').

    inputs:
        ray_origins (...,R,3): the origin of each ray
        ray_directions (...,R,3): the direction of each ray
        object_filename (str): path to the .obj file
        alpha_1 (float): scaling factor for the energy return
        alpha_2 (float): offset for the energy return
    outputs:
        dist (...,R'): the distance of each ray that hit the object
        energy (...,R'): the energy of each ray that hit the object
    '''
    device = ray_origins.device
    assert(ray_origins.shape == ray_directions.shape), 'Incompatible shapes, got {} and {}'.format(ray_origins.shape, ray_directions.shape)

    # find where/how rays hit the object
    mesh = load_obj_vertices_faces(object_filename, device) # (F,3), (F,3), (F,3)
    hit, dist, hit_pos, normals = ray_triangle_intersect(ray_origins.reshape(-1,3), ray_directions.reshape(-1,3), *mesh) # (N*R,) (N*R,), (N*R,3), (N*R,3)

    # eliminate rays that didn't hit
    dist = dist[hit] # (...,R')
    hit_pos = hit_pos[hit] # (...,R',3)
    normals = normals[hit] # (...,R',3)
    ray_directions = ray_directions[hit] # (...,R',3)
    ray_origins = ray_origins[hit] # (...,R',3)

    # Use cosine(angle) between ray and surface normal to get returned energy
    cosine_similarity = torch.abs(torch.sum(-ray_directions * normals, dim=-1)) # (...,R')
    energy = cosine_similarity * alpha_1 + alpha_2 # (...,R')

    return dist, energy


def accumulate_scatters(target_poses, z_near, z_far, object_filename,
               azimuth_spread=15, n_pulses=30, n_rays_per_side=4):
    '''
    returns the energy and range for a bunch of rays for each pulse

    inputs:
        target_poses (T,4,4): the rgb pose for which we want to get a sar image from
        z_near (int): near distance of the obj
        z_far (int): far distance of the obj
        object_filename (str): path to the .obj file
        azimuth_spread (float): the range of azimuth angles for sar rendering
        n_pulses (int): the number of pulses for sar rendering
        n_rays_per_side (int): the number of rays on each side of the triangle, yielding num_ray_side**2 total rays

    outputs:
        range (T,P,R'): the range of all the rays that hit the object
        energy (T,P,R'): the simulated energy of all the rays that hit the object

    '''
    device = target_poses.device
    T = target_poses.shape[0]  # no. of camera views
    P = n_pulses               # no. of pulses per view

    # Pull out camera positions info
    cam_center, _, _, _, cam_distance, cam_elevation, cam_azimuth = extract_pose_info(target_poses)

    # Spread the pulses across a small range of azimuth angles
    azimuth_offsets = torch.linspace(-azimuth_spread / 2, azimuth_spread / 2, P, device=device) * torch.pi / 180 # (P,)

    # Side length of the raycasting plane
    side_len = abs(z_far - z_near)

    # get the azimuth and elevation for each pulse
    azimuth = cam_azimuth.reshape(T, 1) + azimuth_offsets.reshape(1, P) # (T,P)
    elevation = torch.tile(cam_elevation.reshape(T, 1), (1, P)) # (T,P)

    # Generate rays for all pulses at once
    ray_origins, ray_directions = generate_ray_grid(
        azimuth, elevation, side_len, cam_distance,
        n_rays_per_side
    )  # (T,P,R,3), (T,P,R,3)
    
    # compute the range and energy for all rays, ommitting the ones that miss
    scatter_ranges, scatter_energies = get_range_and_energy( ray_origins, ray_directions, object_filename) # (T,P,R'), (T,P,R')

    return scatter_ranges, scatter_energies, azimuth, elevation



def interpolate_signals(all_ranges,all_energy,sensor_position,view_dir,radar_bw,radar_fs,near_range,far_range):
    # calculate the center of each spatial sample
    first_z = int(math.ceil(near_range*radar_fs))
    last_z =  int(math.floor(far_range*radar_fs))
    sample_z = torch.arange(first_z, last_z+1, device=device, dtype=target_poses.dtype)/radar_fs # (N,)
    N = len(sample_z)

    #interpolate the signal
    # signal shape will be (T,P,N), but we need to sum over all R rays, so lets go for (T,P,N,R) within the sum
    received_signal = torch.sum(
                all_energy.reshape(T,P,R,1) * \
                torch.sinc( radar_bw * \
                            (all_ranges.reshape(T,P,R,1) - sample_z.reshape(1,1,1,N))
                          ),
                dim=-1
            ) # (T,P,N)

    #Calculate sample positions
    sensor_position = sensor_position.reshape(T,P,1,3) + sampel_z.reshape(1,1,N,3) * view_directions.reshape(T,P,1,3) # (T,P,N,3)


    return signals, sample_positions
           #(T,P,N)         (T,P,N,3)

# Take all the ranges and energies and generate a radar-like echo signal
def simulate_echo_signal(ray_ranges, ray_energies, z_near, z_far, spatial_fs):
    Z = int((z_far - z_near) * spatial_fs)
    z_signal = torch.linspace(z_near, z_far, Z, device=ray_ranges.device)

    # Add up each echo at its range using a sinc kernel
    signal = torch.sum(
        ray_energies[..., None] *
        torch.sinc(ray_ranges[..., None] - z_signal[None, None, None, :]),
        dim=-2  # sum across rays
    )

    return z_signal, signal
