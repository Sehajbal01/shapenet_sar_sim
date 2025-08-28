# written by JhihYang Wu <jhihyangwu@arizona.edu>

from ..core import EPSILON
import torch

def triangles_rays_intersection(ray_origins, ray_directions,
                                triangles_A, triangles_edge1, triangles_edge2, triangles_normal, triangles_indices):
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

    Returns:
        min_intersections (torch.Tensor): (N,) tensor of minimum intersection distances for each ray.
                                               -1.0 indicates no intersection with any triangle.
        min_indices (torch.Tensor): (N,) tensor of triangle indices corresponding to the minimum intersections.
                                          -1 indicates no intersection with any triangle.
    """
    # Expand dimensions for broadcasting: rays (N,1,3), triangles (1,M,3)
    ray_dirs_expanded = ray_directions.unsqueeze(1)  # (N, 1, 3)
    ray_origs_expanded = ray_origins.unsqueeze(1)    # (N, 1, 3)
    
    edge1_expanded = triangles_edge1.unsqueeze(0)    # (1, M, 3)
    edge2_expanded = triangles_edge2.unsqueeze(0)    # (1, M, 3)
    A_expanded = triangles_A.unsqueeze(0)            # (1, M, 3)

    # https://github.com/JhihYangWu/miniRT/blob/main/src/geometry/triangle.cpp
    
    # Vector3 pVec = cross(r.d, edge2);
    pVec = torch.cross(ray_dirs_expanded, edge2_expanded, dim=2)  # (N, M, 3)
    
    # float det = dot(pVec, edge1);
    det = torch.sum(pVec * edge1_expanded, dim=2)  # (N, M)
    
    # if (det > -EPSILON && det < EPSILON) return -1.0f;
    valid_det = torch.abs(det) > EPSILON  # (N, M)
    
    # float invDet = 1.0f / det;
    invDet = torch.where(valid_det, 1.0 / det, torch.zeros_like(det))  # (N, M)
    
    # Vector3 tVec = r.o - A;
    tVec = ray_origs_expanded - A_expanded  # (N, M, 3)
    
    # float u = dot(pVec, tVec) * invDet;
    u = torch.sum(pVec * tVec, dim=2) * invDet  # (N, M)
    
    # if (u < 0.0f || u > 1.0f) return -1.0f; // missed triangle
    valid_u = (u >= 0.0) & (u <= 1.0)  # (N, M)
    
    # Vector3 qVec = cross(tVec, edge1);
    qVec = torch.cross(tVec, edge1_expanded, dim=2)  # (N, M, 3)
    
    # float v = dot(qVec, r.d) * invDet;
    v = torch.sum(qVec * ray_dirs_expanded, dim=2) * invDet  # (N, M)
    
    # if (v < 0.0f || u + v > 1.0f) return -1.0f; // missed triangle
    valid_v = (v >= 0.0) & ((u + v) <= 1.0)  # (N, M)
    
    # float t = dot(qVec, edge2) * invDet; // time of intersection
    t = torch.sum(qVec * edge2_expanded, dim=2) * invDet  # (N, M)
    
    # if (t < 0.0f) return -1.0f;
    valid_t = t >= 0.0  # (N, M)
    
    # Combine all validity checks
    hit_mask = valid_det & valid_u & valid_v & valid_t  # (N, M)
    
    # Set intersection distances to infinity where there's no intersection
    # This allows us to use min() to find the closest valid intersection
    intersections = torch.where(hit_mask, t, torch.full_like(t, float("inf")))  # (N, M)
    
    # Find minimum intersection distance for each ray by taking min over triangles
    min_distances, min_triangle_indices = torch.min(intersections, dim=1)  # (N,), (N,)
    
    # Set rays with no intersections to -1.0
    min_intersections = torch.where(min_distances == float("inf"), 
                                    torch.full_like(min_distances, -1.0), 
                                    min_distances)
    
    # Get the actual triangle IDs for the minimum intersections
    # For rays that hit nothing, set triangle index to -1
    min_triangle_ids = torch.where(min_distances == float("inf"),
                                   torch.full_like(min_triangle_indices, -1),
                                   triangles_indices[min_triangle_indices])
    
    return min_intersections, min_triangle_ids
