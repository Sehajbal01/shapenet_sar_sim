import torch
import tqdm

EPSILON = 1e-8


def ray_trace(ray_origins, ray_directions, mesh, face_normals, batch_size=2**20):
    """
    Möller–Trumbore ray-triangle intersection for R rays against F triangles.
    Processes rays in batches to bound peak GPU memory at O(batch_size * F).

    Args:
        ray_origins:    (R, 3) world-space ray start points
        ray_directions: (R, 3) ray directions (need not be normalized)
        mesh:           Meshes object with pre-attached .edge_1 (F,3) and .edge_2 (F,3)
        face_normals:   (F, 3) unused by MT — kept for API consistency with rasterizer path
        batch_size:     max rays processed at once

    Returns:
        hit_face_ids: (R,) LongTensor  — index of nearest hit face, -1 for miss
        distances:    (R,) FloatTensor — distance along ray to nearest hit, -1.0 for miss
        hit_bool:     (R,) BoolTensor  — True where a hit occurred
    """
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    v0 = verts[faces[:, 0]]  # (F, 3) first vertex of each triangle
    e1 = mesh.edge_1          # (F, 3) v1 - v0, pre-computed in load_mesh
    e2 = mesh.edge_2          # (F, 3) v2 - v0, pre-computed in load_mesh
    F = e1.shape[0]
    R = ray_origins.shape[0]
    device = ray_origins.device

    hit_face_ids = torch.full((R,), -1, dtype=torch.long, device=device)
    distances    = torch.full((R,), -1.0, dtype=torch.float32, device=device)

    # pre-unsqueeze face tensors so they broadcast across each batch dimension
    e1_f = e1.unsqueeze(0)  # (1, F, 3)
    e2_f = e2.unsqueeze(0)  # (1, F, 3)
    v0_f = v0.unsqueeze(0)  # (1, F, 3)

    for start in tqdm.tqdm(range(0, R, batch_size), desc='ray tracing', unit='batch', leave=False):
        end = min(start + batch_size, R)
        B = end - start

        ro = ray_origins[start:end].unsqueeze(1)     # (B, 1, 3)
        rd = ray_directions[start:end].unsqueeze(1)  # (B, 1, 3)

        # h = cross(rd, e2)
        h = torch.cross(rd.expand(B, F, 3), e2_f.expand(B, F, 3), dim=-1)  # (B, F, 3)

        # det = dot(e1, h) — near zero means ray is parallel to triangle plane
        det = (e1_f.expand(B, F, 3) * h).sum(dim=-1)  # (B, F)
        valid = det.abs() > EPSILON

        # f = 1/det, safe against division by zero (invalids are masked out later)
        f = 1.0 / det.masked_fill(~valid, 1.0)  # (B, F)

        # s = ray_origin - v0
        s = ro.expand(B, F, 3) - v0_f.expand(B, F, 3)  # (B, F, 3)

        # u = f * dot(s, h) — first barycentric coordinate, must be in [0, 1]
        u = f * (s * h).sum(dim=-1)  # (B, F)
        valid &= (u >= 0.0) & (u <= 1.0)

        # q = cross(s, e1)
        q = torch.cross(s, e1_f.expand(B, F, 3), dim=-1)  # (B, F, 3)

        # v = f * dot(rd, q) — second barycentric coordinate, u+v must be in [0, 1]
        v = f * (rd.expand(B, F, 3) * q).sum(dim=-1)  # (B, F)
        valid &= (v >= 0.0) & (u + v <= 1.0)

        # t = f * dot(e2, q) — signed distance along ray to intersection
        t = f * (e2_f.expand(B, F, 3) * q).sum(dim=-1)  # (B, F)
        valid &= (t > EPSILON)  # must be strictly in front of the ray origin

        # find the nearest valid hit face per ray
        t_masked = t.masked_fill(~valid, float('inf'))  # (B, F)
        nearest_t, nearest_face = t_masked.min(dim=1)   # (B,)

        hit_mask = nearest_t < float('inf')
        hit_face_ids[start:end] = nearest_face.masked_fill(~hit_mask, -1)
        distances[start:end]    = nearest_t.masked_fill(~hit_mask, -1.0)

    return hit_face_ids, distances