# written by JhihYang Wu <jhihyangwu@arizona.edu>

from ..core import EPSILON
import torch

def triangles_rays_intersection(ray_origins, ray_directions,
                                triangles_A, triangles_edge1, triangles_edge2, triangles_normal, triangles_indices,
                                ray_batch_size=4096):
    """
    Computes the minimum intersection distance between multiple rays and multiple triangles.
    Refer to Fast, Minimum Storage Ray Triangle Intersection 1997 paper.
    Moller-Trumbore intersection algorithm.

    Args:
        ray_origins (torch.Tensor): (N, 3) tensor of ray origins.
        ray_directions (torch.Tensor): (N, 3) tensor of ray directions.
        triangles_A (torch.Tensor): (M, 3) tensor of triangle vertices A.
        triangles_edge1 (torch.Tensor): (M, 3) tensor of triangle edges 1.
        triangles_edge2 (torch.Tensor): (M, 3) tensor of triangle edges 2.
        triangles_normal (torch.Tensor): (M, 3) tensor of triangle normals.
        triangles_indices (torch.Tensor): (M,) tensor of triangle indices / IDs.
        ray_batch_size (int): number of rays to process per batch to bound peak VRAM.

    Returns:
        min_intersections (torch.Tensor): (N,) tensor of minimum intersection distances for each ray.
                                               -1.0 indicates no intersection with any triangle.
        min_indices (torch.Tensor): (N,) tensor of triangle indices corresponding to the minimum intersections.
                                          -1 indicates no intersection with any triangle.
    """
    N = ray_origins.shape[0]
    device = ray_origins.device
    min_intersections = torch.full((N,), -1.0, dtype=torch.float32, device=device)
    min_triangle_ids  = torch.full((N,), -1,   dtype=torch.long,    device=device)

    edge1_expanded = triangles_edge1.unsqueeze(0)  # (1, M, 3)
    edge2_expanded = triangles_edge2.unsqueeze(0)  # (1, M, 3)
    A_expanded     = triangles_A.unsqueeze(0)      # (1, M, 3)

    for start in range(0, N, ray_batch_size):
        end = min(start + ray_batch_size, N)

        ray_dirs_expanded  = ray_directions[start:end].unsqueeze(1)  # (B, 1, 3)
        ray_origs_expanded = ray_origins[start:end].unsqueeze(1)     # (B, 1, 3)

        # https://github.com/JhihYangWu/miniRT/blob/main/src/geometry/triangle.cpp
        pVec      = torch.cross(ray_dirs_expanded, edge2_expanded, dim=2)       # (B, M, 3)
        det       = torch.sum(pVec * edge1_expanded, dim=2)                     # (B, M)
        valid_det = torch.abs(det) > EPSILON                                    # (B, M)
        invDet    = torch.where(valid_det, 1.0 / det, torch.zeros_like(det))   # (B, M)
        tVec      = ray_origs_expanded - A_expanded                             # (B, M, 3)
        u         = torch.sum(pVec * tVec, dim=2) * invDet                     # (B, M)
        valid_u   = (u >= 0.0) & (u <= 1.0)                                    # (B, M)
        qVec      = torch.cross(tVec, edge1_expanded, dim=2)                   # (B, M, 3)
        v         = torch.sum(qVec * ray_dirs_expanded, dim=2) * invDet        # (B, M)
        valid_v   = (v >= 0.0) & ((u + v) <= 1.0)                             # (B, M)
        t         = torch.sum(qVec * edge2_expanded, dim=2) * invDet           # (B, M)
        valid_t   = t >= 0.0                                                    # (B, M)

        hit_mask      = valid_det & valid_u & valid_v & valid_t                 # (B, M)
        intersections = torch.where(hit_mask, t, torch.full_like(t, float("inf")))  # (B, M)

        min_dist, min_tri_idx = torch.min(intersections, dim=1)                # (B,), (B,)

        hit = min_dist != float("inf")
        min_intersections[start:end][hit] = min_dist[hit]
        min_triangle_ids[start:end][hit]  = triangles_indices[min_tri_idx[hit]]

    return min_intersections, min_triangle_ids
