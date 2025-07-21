import torch
from pytorch3d.io import load_objs_as_meshes

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
def ray_triangle_intersect(ray_origins, ray_directions, v0, v1, v2, eps=1e-8):
    ray_origins = ray_origins[:, None, :]        # (R, 1, 3)
    ray_directions = ray_directions[:, None, :]  # (R, 1, 3)

    e1 = v1 - v0         # (F, 3)
    e2 = v2 - v0
    h = torch.cross(ray_directions, e2, dim=-1)
    a = torch.sum(e1[None, :, :] * h, dim=-1)

    valid_mask = (a > eps) | (a < -eps)
    f = 1.0 / (a + eps * (~valid_mask))  # avoid division by 0
    s = ray_origins - v0
    u = f * torch.sum(s * h, dim=-1)

    q = torch.cross(s, e1[None, :, :], dim=-1)
    v = f * torch.sum(ray_directions * q, dim=-1)
    t = f * torch.sum(e2[None, :, :] * q, dim=-1)

    # Filter valid intersections
    hit = (valid_mask & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps))
    t[~hit] = float('inf')

    # Get closest hit triangle for each ray
    min_t, min_idx = torch.min(t, dim=1)  # (R,), (R,)
    hit_pos = ray_origins.squeeze(1) + min_t.unsqueeze(-1) * ray_directions.squeeze(1)  # (R, 3)

    # Compute triangle normals
    n = torch.cross(e1, e2, dim=-1)
    n = n / (torch.norm(n, dim=-1, keepdim=True) + eps)
    normals = n[min_idx]  # (R, 3)

    return hit, hit_pos, normals

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
