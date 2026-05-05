import torch
import numpy as np
import tqdm

EPSILON = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# GPU-resident flat octree
# ─────────────────────────────────────────────────────────────────────────────

class Octree:
    """
    Flat GPU representation of an octree over mesh triangles.

    Triangles are assigned to leaves by centroid.  Each leaf's AABB is the
    tight bound over all its triangles, so no triangle is ever missed.

    Attributes
    ----------
    leaf_mins, leaf_maxs       : (L, 3) float32 — per-leaf AABB bounds
    leaf_face_starts_gpu       : (L,) long       — start index into packed_face_indices
    leaf_face_counts_gpu       : (L,) long       — number of faces in each leaf
    leaf_face_starts           : Python list[int] (kept for slicing in fallback path)
    leaf_face_counts           : Python list[int]
    packed_face_indices        : (T,) long       — face IDs packed by leaf
    L                          : int — number of leaves
    """

    def __init__(self, leaf_mins, leaf_maxs, leaf_face_starts, leaf_face_counts,
                 packed_face_indices):
        self.leaf_mins = leaf_mins
        self.leaf_maxs = leaf_maxs
        self.leaf_face_starts = leaf_face_starts   # Python list
        self.leaf_face_counts = leaf_face_counts   # Python list
        self.leaf_face_starts_gpu = torch.tensor(
            leaf_face_starts, dtype=torch.long, device=leaf_mins.device)
        self.leaf_face_counts_gpu = torch.tensor(
            leaf_face_counts, dtype=torch.long, device=leaf_mins.device)
        self.packed_face_indices = packed_face_indices
        self.L = leaf_mins.shape[0]


def build_octree(mesh, max_depth=8, max_tris_per_leaf=32):
    """
    Build a GPU octree from a PyTorch3D Meshes object.

    The recursive subdivision runs on CPU/numpy (one-time cost), then all
    leaf data is uploaded to the same device as the mesh.

    Args:
        mesh             : PyTorch3D Meshes with verts_packed() / faces_packed()
        max_depth        : maximum recursion depth
        max_tris_per_leaf: stop splitting when a node has ≤ this many triangles

    Returns:
        Octree object with all tensors on mesh.device
    """
    device = mesh.verts_packed().device
    verts_np = mesh.verts_packed().detach().cpu().numpy().astype(np.float32)
    faces_np = mesh.faces_packed().detach().cpu().numpy()
    F = len(faces_np)

    v0 = verts_np[faces_np[:, 0]]
    v1 = verts_np[faces_np[:, 1]]
    v2 = verts_np[faces_np[:, 2]]
    tri_min  = np.minimum(np.minimum(v0, v1), v2)  # (F, 3)
    tri_max  = np.maximum(np.maximum(v0, v1), v2)  # (F, 3)
    centroid = (v0 + v1 + v2) / 3.0               # (F, 3)

    leaf_mins_list  = []
    leaf_maxs_list  = []
    leaf_faces_list = []

    def subdivide(fi, depth):
        if len(fi) == 0:
            return
        node_min = tri_min[fi].min(axis=0)
        node_max = tri_max[fi].max(axis=0)
        if depth >= max_depth or len(fi) <= max_tris_per_leaf:
            leaf_mins_list.append(node_min)
            leaf_maxs_list.append(node_max)
            leaf_faces_list.append(fi)
            return
        mid = (node_min + node_max) * 0.5
        c   = centroid[fi]
        for bits in range(8):
            ox, oy, oz = (bits >> 2) & 1, (bits >> 1) & 1, bits & 1
            mask = (
                ((c[:, 0] >= mid[0]) == bool(ox)) &
                ((c[:, 1] >= mid[1]) == bool(oy)) &
                ((c[:, 2] >= mid[2]) == bool(oz))
            )
            if mask.any():
                subdivide(fi[mask], depth + 1)

    subdivide(np.arange(F, dtype=np.int64), 0)

    starts, counts, packed = [], [], []
    cursor = 0
    for fi in leaf_faces_list:
        starts.append(cursor)
        counts.append(len(fi))
        packed.extend(fi.tolist())
        cursor += len(fi)

    return Octree(
        leaf_mins=torch.from_numpy(
            np.array(leaf_mins_list, dtype=np.float32)).to(device),
        leaf_maxs=torch.from_numpy(
            np.array(leaf_maxs_list, dtype=np.float32)).to(device),
        leaf_face_starts=starts,
        leaf_face_counts=counts,
        packed_face_indices=torch.tensor(
            packed, dtype=torch.long, device=device),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ray tracer
# ─────────────────────────────────────────────────────────────────────────────

def ray_trace(ray_origins, ray_directions, mesh, face_normals,
              octree=None, batch_size=2**20, show_pbar=False):
    """
    Möller–Trumbore ray-triangle intersection for R rays against F triangles.

    With an Octree the traversal is fully vectorised on GPU:
      1. Test all B rays against all L leaf AABBs simultaneously → (B, L) bool.
      2. Expand active (ray, leaf) pairs to a padded (K, max_Fl) face matrix.
      3. Run MT on all (K, max_Fl) pairs in one kernel.
      4. Sort by distance (descending) and scatter to per-ray minimum — no loop.

    GPU↔CPU syncs per outer batch:
      • ceil(B / aabb_chunk) vectorised AABB ops    (large tensors, few launches)
      • 1 sync for nonzero() on (B, L) hit matrix
      • 1 sync for max leaf face count               (to size the padded matrix)
      • 0 syncs in any inner loop

    Args:
        ray_origins    : (R, 3) world-space ray start points
        ray_directions : (R, 3) ray directions (need not be normalised)
        mesh           : Meshes with pre-attached .edge_1 (F,3) and .edge_2 (F,3)
        face_normals   : (F, 3) unused — kept for API compatibility
        octree         : Octree from build_octree(), or None for brute-force
        batch_size     : max rays processed per GPU batch
        show_pbar      : show tqdm progress bar

    Returns:
        hit_face_ids : (R,) LongTensor  — nearest hit face, -1 for miss
        distances    : (R,) FloatTensor — distance along ray to hit, -1.0 for miss
    """
    verts  = mesh.verts_packed()
    faces  = mesh.faces_packed()
    v0_all = verts[faces[:, 0]]  # (F, 3)
    e1_all = mesh.edge_1          # (F, 3)
    e2_all = mesh.edge_2          # (F, 3)
    F      = e1_all.shape[0]
    R      = ray_origins.shape[0]
    device = ray_origins.device

    hit_face_ids = torch.full((R,), -1,   dtype=torch.long,    device=device)
    distances    = torch.full((R,), -1.0, dtype=torch.float32, device=device)

    batches = range(0, R, batch_size)
    if show_pbar:
        label   = 'ray tracing (octree)' if octree is not None else 'ray tracing'
        batches = tqdm.tqdm(batches, desc=label, unit='batch', leave=False)

    if octree is None:
        # ── Brute-force: O(B × F) ────────────────────────────────────────────
        e1_f = e1_all.unsqueeze(0)  # (1, F, 3)
        e2_f = e2_all.unsqueeze(0)
        v0_f = v0_all.unsqueeze(0)

        for start in batches:
            end = min(start + batch_size, R)
            B   = end - start

            ro = ray_origins[start:end].unsqueeze(1)     # (B, 1, 3)
            rd = ray_directions[start:end].unsqueeze(1)  # (B, 1, 3)

            h     = torch.cross(rd.expand(B, F, 3), e2_f.expand(B, F, 3), dim=-1)
            det   = (e1_f.expand(B, F, 3) * h).sum(-1)
            valid = det.abs() > EPSILON
            f_inv = 1.0 / det.masked_fill(~valid, 1.0)
            s     = ro.expand(B, F, 3) - v0_f.expand(B, F, 3)
            u     = f_inv * (s * h).sum(-1)
            valid &= (u >= 0.0) & (u <= 1.0)
            q     = torch.cross(s, e1_f.expand(B, F, 3), dim=-1)
            v     = f_inv * (rd.expand(B, F, 3) * q).sum(-1)
            valid &= (v >= 0.0) & (u + v <= 1.0)
            t     = f_inv * (e2_f.expand(B, F, 3) * q).sum(-1)
            valid &= t > EPSILON

            t_masked = t.masked_fill(~valid, float('inf'))
            nearest_t, nearest_face = t_masked.min(dim=1)
            hit_mask = nearest_t < float('inf')
            hit_face_ids[start:end] = nearest_face.masked_fill(~hit_mask, -1)
            distances[start:end]    = nearest_t.masked_fill(~hit_mask, -1.0)

    else:
        # ── Octree-accelerated: fully vectorised, no Python loop over leaves ──
        #
        # Algorithm per outer batch of B rays:
        #   A) AABB test: (B, L) hit matrix in ceil(B/aabb_chunk) GPU ops
        #   B) K = nonzero pairs (ray, leaf) that pass AABB  [1 GPU sync]
        #   C) Build padded face matrix (K, max_Fl) via gather ops [1 GPU sync]
        #   D) MT on (K, max_Fl) in one kernel
        #   E) Lex-sort pairs by (ray_id, t), pick first per ray  [1 GPU sync]
        #
        leaf_mins = octree.leaf_mins           # (L, 3)
        leaf_maxs = octree.leaf_maxs           # (L, 3)
        fs_gpu    = octree.leaf_face_starts_gpu  # (L,)
        fc_gpu    = octree.leaf_face_counts_gpu  # (L,)
        pack      = octree.packed_face_indices   # (T,)
        T         = pack.shape[0]
        L         = octree.L

        # Cap AABB sub-chunk size to ~64 MB of float32 (t1 + t2 = 2×(C,L,3))
        AABB_MAX_BYTES = 64 << 20
        aabb_chunk = max(1, AABB_MAX_BYTES // (L * 3 * 4 * 2))

        for start in batches:
            end = min(start + batch_size, R)
            B   = end - start

            ro = ray_origins[start:end]     # (B, 3)
            rd = ray_directions[start:end]  # (B, 3)

            # Safe reciprocal: preserves sign, avoids zero-division in slab test
            sign_rd = torch.sign(rd)
            sign_rd = torch.where(sign_rd != 0, sign_rd, torch.ones_like(sign_rd))
            inv_rd  = sign_rd / rd.abs().clamp(min=EPSILON)  # (B, 3)

            # ── A: vectorised AABB test, all L leaves at once ─────────────
            hit_matrix = torch.empty(B, L, dtype=torch.bool, device=device)
            for ci in range(0, B, aabb_chunk):
                cj    = min(ci + aabb_chunk, B)
                ro_c  = ro[ci:cj].unsqueeze(1)      # (C, 1, 3)
                irc   = inv_rd[ci:cj].unsqueeze(1)  # (C, 1, 3)
                t1    = (leaf_mins.unsqueeze(0) - ro_c) * irc   # (C, L, 3)
                t2    = (leaf_maxs.unsqueeze(0) - ro_c) * irc
                t_nr  = torch.max(torch.minimum(t1, t2), dim=-1).values  # (C, L)
                t_fr  = torch.min(torch.maximum(t1, t2), dim=-1).values
                hit_matrix[ci:cj] = (t_nr <= t_fr) & (t_fr > EPSILON)

            # ── B: expand to (ray, leaf) pairs — 1 GPU sync ───────────────
            ray_ids_k, leaf_ids_k = hit_matrix.nonzero(as_tuple=True)  # (K,)
            K = ray_ids_k.shape[0]

            best_t    = torch.full((B,), float('inf'), dtype=torch.float32, device=device)
            best_face = torch.full((B,), -1,           dtype=torch.long,    device=device)

            if K > 0:
                # ── C: build padded (K, max_Fl) face index matrix ─────────
                fs_k   = fs_gpu[leaf_ids_k]   # (K,) start offset per pair
                fc_k   = fc_gpu[leaf_ids_k]   # (K,) face count per pair
                max_Fl = int(fc_k.max().item())   # 1 GPU sync

                col         = torch.arange(max_Fl, device=device)  # (max_Fl,)
                flat_idx    = (fs_k.unsqueeze(1) + col.unsqueeze(0)).clamp(0, T - 1)  # (K, max_Fl)
                face_valid  = col.unsqueeze(0) < fc_k.unsqueeze(1)  # (K, max_Fl) padding mask
                face_id_mat = pack[flat_idx]                         # (K, max_Fl) global face IDs

                # ── D: Möller–Trumbore on all (K, max_Fl) pairs ───────────
                ro_k = ro[ray_ids_k].unsqueeze(1)    # (K, 1, 3)
                rd_k = rd[ray_ids_k].unsqueeze(1)    # (K, 1, 3)
                e1_m = e1_all[face_id_mat]            # (K, max_Fl, 3)
                e2_m = e2_all[face_id_mat]
                v0_m = v0_all[face_id_mat]

                h     = torch.cross(rd_k.expand(K, max_Fl, 3), e2_m, dim=-1)
                det   = (e1_m * h).sum(-1)              # (K, max_Fl)
                valid = det.abs() > EPSILON
                f_inv = 1.0 / det.masked_fill(~valid, 1.0)
                s     = ro_k.expand(K, max_Fl, 3) - v0_m
                u     = f_inv * (s * h).sum(-1)
                valid &= (u >= 0.0) & (u <= 1.0)
                q     = torch.cross(s, e1_m, dim=-1)
                v     = f_inv * (rd_k.expand(K, max_Fl, 3) * q).sum(-1)
                valid &= (v >= 0.0) & (u + v <= 1.0)
                t     = f_inv * (e2_m * q).sum(-1)
                valid &= t > EPSILON
                valid &= face_valid                     # mask padding

                t_masked        = t.masked_fill(~valid, float('inf'))
                pair_t, pair_fi = t_masked.min(dim=1)  # (K,) best per pair

                # ── E: reduce pairs to per-ray minimum (deterministic) ────
                # Repeated-index scatter on CUDA is non-deterministic, so we
                # instead sort pairs lexicographically by (ray_id ASC, pair_t ASC)
                # using two stable sorts, then take the first element per ray
                # group.  After this, win_rays is unique → single write per ray.
                pair_face = face_id_mat[torch.arange(K, device=device), pair_fi]

                ord_t   = pair_t.argsort()                    # secondary: sort by t asc
                s_ray   = ray_ids_k[ord_t]
                s_t     = pair_t[ord_t]
                s_face  = pair_face[ord_t]

                ord_ray = s_ray.argsort(stable=True)          # primary: stable sort by ray
                s_ray   = s_ray[ord_ray]
                s_t     = s_t[ord_ray]
                s_face  = s_face[ord_ray]

                # First element per ray = minimum t for that ray
                is_first        = torch.ones(K, dtype=torch.bool, device=device)
                is_first[1:]    = s_ray[1:] != s_ray[:-1]
                win             = is_first & (s_t < float('inf'))
                win_idx         = win.nonzero(as_tuple=True)[0]   # 1 GPU sync
                win_rays        = s_ray[win_idx]                   # unique per ray

                best_t[win_rays]    = s_t[win_idx]
                best_face[win_rays] = s_face[win_idx]

            miss = best_t == float('inf')
            hit_face_ids[start:end] = best_face.masked_fill(miss, -1)
            distances[start:end]    = best_t.masked_fill(miss, -1.0)

    return hit_face_ids, distances
