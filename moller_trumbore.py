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
def ray_triangle_intersect(ray_origins, ray_directions, v1, v2, v3, eps=1e-8, batch_size=2**25):
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

    outputs:
        hit (R): boolean mask indicating which rays hit the triangles
        dist (R): distance of each ray that hit the triangle, inf if it missed
        hit_pos (R, 3): coordinates of the intersection points
        normals (R, 3): surface normals at the intersection points
    '''
    R = ray_origins.shape[0]  # number of rays
    F = v1.shape[0]           # number of triangles

    # edge vectors
    e1 = v2 - v1 # (F,3)
    e2 = v3 - v1 # (F,3)

    # normal vector
    n = torch.cross(e1, e2, dim=-1)  # (F,3)

    # set up the system of equations
    arg_mat = torch.stack([ torch.tile(ray_directions.reshape(R,1,3),(1,F,1)), 
                            torch.tile(e1.reshape(1,F,3),            (R,1,1)), 
                            torch.tile(e2.reshape(1,F,3),            (R,1,1)) ], dim=2) # (R,F,3,3) # might need to be dim=3
    arg_vec = ray_origins.reshape(R,1,3,1) - v1.reshape(1,F,3,1)  # (R,F,3,1)

    # solve the system of equations in batches to avoid memory issues
    N = R * F
    if batch_size is None:
        batch_size = N
    batch_size = min(int(batch_size), N)
    arg_mat = arg_mat.reshape(N,3,3)  # (N,3,3)
    arg_vec = arg_vec.reshape(N,3,1)  # (N,3,1)


    # init tqdm progress bar
    pbar = tqdm.tqdm(total=N, desc='Ray-triangle intersection')

    start = 0
    t_u_v = []
    while start < N:
        end = min(start + batch_size, N)
        batch_mat = arg_mat[start:end] # (batch_size,3,3)
        batch_vec = arg_vec[start:end] # (batch_size,3,1)
        batch_mat = batch_mat.to('cuda')
        batch_vec = batch_vec.to('cuda')
        batch_mat = batch_mat.inverse()  # (batch_size,3,3)
        batch_t_u_v = batch_mat @ batch_vec  # (batch_size,3,1)
        t_u_v.append(batch_t_u_v.to('cpu'))
        start = end

        # update progress bar
        pbar.update(batch_size)
    pbar.close()

    # concatenate results
    t_u_v = torch.cat(t_u_v, dim=0)  # (N,3,1)
    t_u_v = t_u_v.reshape(R, F, 3)  # (R,F,3)

    # seperate the results
    dist = t_u_v[:, :, 0, 0] # (R,F)

    return hit, min_t, hit_pos, normals

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
