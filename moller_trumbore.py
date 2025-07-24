import torch
from pytorch3d.io import load_objs_as_meshes
import tqdm

# Load vertex positions and face indices from .obj file
def load_obj_vertices_faces(obj_filename, device):
    mesh = load_objs_as_meshes([obj_filename], device=device)
    verts = mesh.verts_padded()[0]  # (V, 3)
    faces = mesh.faces_padded()[0]  # (F, 3)
    v0 = verts[faces[:, 0]]         # (F, 3)
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return v0, v1, v2

# Batch ray-triangle intersection using Möller–Trumbore
def ray_triangle_intersect(ray_origins, ray_directions, v1, v2, v3, eps=1e-8, batch_size=2**25, progress_bar=True):
    '''
    PyTorch implementation of the Möller–Trumbore algorithm to find where rays intersect with triangles.
    Only returns the first intersection for each ray.
    Also returns whether a ray hits a triangle or nothing.

    see https://en.wikipedia.org/wiki/Möller-Trumbore_intersection_algorithm for details/derivation

    R is the number of rays, F is the number of triangles

    inputs:
        ray_origins (R, 3): origins of the rays
        ray_directions (R, 3): unit vector directions of the rays
        v0, v1, v2 (F, 3): vertices of the triangles
        batch_size (int): number of rays to process at once, to avoid memory issues
        progress_bar (bool): whether to show a progress bar

    outputs:
        hit (R): boolean mask indicating which rays hit the triangles
        dist (R): distance of each ray that hit the triangle, inf if it missed
        normals (R, 3): surface normals at the intersection points
    '''
    R = ray_origins.shape[0]  # number of rays
    F = v1.shape[0]           # number of triangles
    N = R * F

    # edge vectors
    e1 = v2 - v1 # (F,3)
    e2 = v3 - v1 # (F,3)

    # normal vector
    n = torch.cross(e1, e2, dim=-1)  # (F,3)
    n = n / torch.norm(n, dim=-1, keepdim=True)  # normalize the normal vector
    n = torch.tile(n.reshape(1, F, 3), (R, 1, 1))  # (R,F,3)

    # set up the system of equations
    ray_directions = torch.tile(ray_directions.reshape(R, 1, 3), (1, F, 1))  # (R,F,3)
    e1 = torch.tile(e1.reshape(1, F, 3), (R, 1, 1))  # (R,F,3)
    e2 = torch.tile(e2.reshape(1, F, 3), (R, 1, 1))  # (R,F,3)
    arg_mat = torch.stack([ ray_directions, e1, e2 ], dim=2) # (R,F,3,3)
    arg_vec = ray_origins.reshape(R, 1, 3) - v1.reshape(1, F, 3) # (R,F,3)

    # set up the tensors for batch processing
    if batch_size is None:
        batch_size = N
    batch_size = min(int(batch_size), N)
    arg_mat = arg_mat.reshape(N,3,3)  # (N,3,3)
    arg_vec = arg_vec.reshape(N,3,1)  # (N,3,1)
    n       =       n.reshape(N,3  )  # (N,3)
    ray_directions = ray_directions.reshape(N,3)  # (N,3)
    ray_origins = torch.tile(ray_origins.reshape(R, 1, 3), (1, F, 1)).reshape(N, 3)  # (N,3)

    # init tqdm progress bar
    pbar = tqdm.tqdm(total=N, desc='Ray-triangle intersection')

    start = 0

    dist, hit = [], []

    while start < N:

        end = min(start + batch_size, N)

        # sequester batch
        batch_mat = arg_mat[start:end] # (B,3,3)
        batch_vec = arg_vec[start:end] # (B,3,1)
        batch_n   =       n[start:end] # (B,3) 
        batch_d   = ray_directions[start:end] # (B,3)
        batch_o   = ray_origins[start:end]   # (B,3)

        # move batch to gpu
        batch_mat = batch_mat.to('cuda')
        batch_vec = batch_vec.to('cuda')
        batch_n   = batch_n.to(  'cuda')
        batch_d   = batch_d.to(  'cuda')

        # solve the system of equations
        batch_t_u_v = torch.linalg.solve(batch_mat, batch_vec)  # (B,3,1)
        del batch_mat, batch_vec  # free memory

        # seperate t, u, v
        t = batch_t_u_v[:, 0, 0] # (B,)
        u = batch_t_u_v[:, 1, 0] # (B,)
        v = batch_t_u_v[:, 2, 0] # (B,)
        del batch_t_u_v  # free memory

        # determine hit/miss
        parallel = torch.sum(batch_d * batch_n, dim=-1).abs() < eps  # (B,)
        hit_batch = (t > 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~parallel # (B,)
        del u, v, parallel  # free memory

        # save results
        dist.append(t)  # (B,)
        hit.append(hit_batch)  # (B,)

        # free memory
        del batch_d, batch_o, t, hit_batch, batch_n

        start = end

        # update progress bar
        pbar.update(batch_size)

    pbar.close()

    # concatenate results
    dist = torch.cat(dist, dim=0)  # (N,)
    hit = torch.cat(hit, dim=0)    # (N,)

    # reshape results to original shape
    dist = dist.reshape(R, F)  # (R,F)
    hit = hit.reshape(R, F)      # (R,F)

    # find the minimum distance and corresponding hit position
    dist,min_idx = torch.min(dist, dim=1) # (R,)
    hit = torch.any(hit, dim=1)  # (R,)

    # get the normals of the first triangle hit for each ray
    n = n.reshape(R,F,3).to('cuda')
    normals = n[torch.arange(R), min_idx, :]  # (R,3)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)  # normalize the normals

    return hit.to('cpu'), dist.to('cpu'), normals.to('cpu')


# Wrapper function for computing ray-mesh returns
def get_range_and_energy(ray_origins, ray_directions, object_filename, alpha_1=0.9, alpha_2=0.1):
    device = ray_origins.device
    v0, v1, v2 = load_obj_vertices_faces(object_filename, device)

    hit, hit_coords, hit_normals = ray_triangle_intersect(ray_origins, ray_directions, v0, v1, v2)

    # Handle rays that missed
    missed = ~hit
    hit_coords[missed] = float('nan')
    hit_normals[missed] = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Compute range and return energy
    ray_range = torch.norm(hit_coords - ray_origins, dim=-1)  # (R,)
    dot = torch.sum(-ray_directions * hit_normals, dim=-1)    # (R,)
    energy = torch.clamp(dot, 0.0, 1.0) * alpha_1 + alpha_2   # (R,)

    return ray_range, energy
