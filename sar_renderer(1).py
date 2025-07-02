import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import ray_mesh_intersect

# Create a grid of rays on a square plane facing a given direction
def generate_ray_grid(c, azimuth, elevation, side_len, r):
    # Figure out which way the camera is facing
    w = -torch.tensor([
        torch.cos(azimuth) * torch.cos(elevation),
        torch.sin(azimuth) * torch.cos(elevation),
        torch.sin(elevation)
    ], device=c.device)

    # Figure out "right" and "up" directions for the plane
    u = torch.tensor([-torch.sin(azimuth), torch.cos(azimuth), 0.0], device=c.device)
    v = torch.tensor([
        -torch.cos(azimuth) * torch.sin(elevation),
        -torch.sin(azimuth) * torch.sin(elevation),
         torch.cos(elevation)
    ], device=c.device)

    # Normalize all directions
    w = torch.nn.functional.normalize(w, dim=0)
    u = torch.nn.functional.normalize(u, dim=0)
    v = torch.nn.functional.normalize(v, dim=0)

    # Make a square grid centered on the camera, in the u-v plane
    lin = torch.linspace(-0.5, 0.5, r, device=c.device)
    grid_x, grid_y = torch.meshgrid(lin, lin, indexing='ij')
    offsets = (grid_x[..., None] * u + grid_y[..., None] * v) * side_len

    # Move the grid to the camera position
    ray_origins = c[None, None, :] + offsets

    # All rays go forward in the same direction
    ray_directions = w[None, None, :].expand(r, r, 3)

    return ray_origins.view(-1, 3), ray_directions.view(-1, 3)


# Shoot rays into a 3D mesh and figure out where they hit and how strong the return signal is
def get_range_and_energy(ray_origins, ray_directions, object_filename, alpha_1=0.9, alpha_2=0.1):
    mesh = load_objs_as_meshes([object_filename], device=ray_origins.device)

    # Find out where the rays hit the mesh
    hit_bool, hit_coords, hit_normals = ray_mesh_intersect(mesh, ray_origins, ray_directions, max_hits=1)

    # For rays that missed, mark their hits as NaN and give them a dummy normal
    missed = ~hit_bool.squeeze(-1)
    hit_coords[missed] = float('nan')
    hit_normals[missed] = torch.tensor([0.0, 0.0, 1.0], device=ray_origins.device)

    # Measure how far each ray traveled
    ray_range = torch.norm(hit_coords - ray_origins, dim=-1)

    # Use cosine(angle) between ray and surface normal to get intensity
    dot = torch.sum(-ray_directions * hit_normals, dim=-1)
    energy = torch.clamp(dot, 0.0, 1.0) * alpha_1 + alpha_2

    return ray_range, energy


# For each camera view, simulate sending radar pulses and collecting returns
def sar_render(target_poses, z_near, z_far, object_filename,
               azimuth_spread=15, n_pulses=30, n_rays_per_side=4):
    device = target_poses.device
    T = target_poses.shape[0]  # no. of camera views
    P = n_pulses               # no. of pulses per view

    # Pull out camera positions
    cam_center = target_poses[:, :3, 3]

    # Figure out where each camera is pointing
    cam_distance = torch.norm(cam_center, dim=-1, keepdim=True)
    cam_elevation = torch.asin(cam_center[:, 2:3] / cam_distance)
    cam_azimuth = torch.acos(cam_center[:, 0:1] / (cam_distance * torch.cos(cam_elevation)))
    cam_azimuth = torch.where(cam_center[:, 1:2] < 0, 2 * torch.pi - cam_azimuth, cam_azimuth)

    # Spread the pulses across a small range of azimuth angles
    azimuth_offsets = torch.linspace(-azimuth_spread / 2, azimuth_spread / 2, P, device=device) * torch.pi / 180

    # Side length of the raycasting plane
    side_len = abs(z_far - z_near)

    # Store results here
    all_ranges, all_energies = [], []

    for t in range(T):
        pulse_ranges, pulse_energies = [], []

        for p in range(P):
            # Set azimuth and elevation for this pulse
            a = cam_azimuth[t, 0] + azimuth_offsets[p]
            e = cam_elevation[t, 0]

            # Making grid of rays for this pulse
            origins, directions = generate_ray_grid(cam_center[t], a, e, side_len, n_rays_per_side)

            # See where they hit the mesh and how much energy they return
            ray_range, energy = get_range_and_energy(origins, directions, object_filename)

            pulse_ranges.append(ray_range)
            pulse_energies.append(energy)

        # Store everything for this camera
        all_ranges.append(torch.stack(pulse_ranges))       # shape (P, R)
        all_energies.append(torch.stack(pulse_energies))   # shape (P, R)

    # Final shape: (T, P, R)
    return torch.stack(all_ranges), torch.stack(all_energies)


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
