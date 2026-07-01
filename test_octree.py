"""
Tests and benchmarks for the GPU octree and Möller–Trumbore ray tracer.

Run with:
    /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 test_octree.py
"""

import time
import numpy as np
import torch
from pytorch3d.structures import Meshes
from ray_tracer_v2 import build_octree, ray_trace, Octree

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_mesh(verts_np, faces_np, device=DEVICE):
    """Build a Meshes object with edge_1 / edge_2 attached (as load_mesh would)."""
    verts = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_np, dtype=torch.long,    device=device)
    mesh  = Meshes(verts=[verts], faces=[faces])
    v0 = verts[faces[:, 0]]
    mesh.edge_1 = verts[faces[:, 1]] - v0
    mesh.edge_2 = verts[faces[:, 2]] - v0
    return mesh


def flat_grid_mesh(n=50, device=DEVICE):
    """Return a flat (z=0) grid mesh with 2*n^2 triangles."""
    xs = np.linspace(-1, 1, n + 1, dtype=np.float32)
    ys = np.linspace(-1, 1, n + 1, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    pts = np.stack([xg.ravel(), yg.ravel(), np.zeros(len(xg.ravel()), np.float32)], 1)
    tris = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b, c, d = a + 1, a + (n + 1), a + (n + 2)
            tris += [[a, b, c], [b, d, c]]
    return make_mesh(pts, np.array(tris, np.int64), device)


def sphere_mesh(subdivisions=3, device=DEVICE):
    """Return an approximate sphere mesh via icosphere subdivision."""
    # Start from icosahedron
    t = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [-1,  t, 0], [ 1,  t, 0], [-1, -t, 0], [ 1, -t, 0],
        [ 0, -1, t], [ 0,  1, t], [ 0, -1,-t], [ 0,  1,-t],
        [ t,  0,-1], [ t,  0, 1], [-t,  0,-1], [-t,  0, 1],
    ], dtype=np.float32)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int64)

    mid_cache = {}
    def midpoint(v, i, j):
        key = (min(i,j), max(i,j))
        if key not in mid_cache:
            m = (v[i] + v[j]) / 2
            m /= np.linalg.norm(m)
            mid_cache[key] = len(v)
            v.append(m)
        return mid_cache[key]

    vlist = list(verts)
    for _ in range(subdivisions):
        new_faces = []
        for f in faces:
            a = midpoint(vlist, f[0], f[1])
            b = midpoint(vlist, f[1], f[2])
            c = midpoint(vlist, f[2], f[0])
            new_faces += [[f[0],a,c],[f[1],b,a],[f[2],c,b],[a,b,c]]
        faces = np.array(new_faces, np.int64)
    verts = np.array(vlist, np.float32)
    return make_mesh(verts, faces, device)


def time_fn(fn, warmup=2, runs=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs


def results_match(hf_b, d_b, hf_o, d_o, dist_tol=1e-4):
    face_ok = (hf_b == hf_o).all().item()
    dist_ok = (d_b - d_o).abs().max().item() < dist_tol
    return face_ok and dist_ok


# ─────────────────────────────────────────────────────────────────────────────
# Correctness tests
# ─────────────────────────────────────────────────────────────────────────────

def test_single_hit():
    """Ray hits triangle 0 of a two-triangle mesh."""
    verts = np.array([[0,0,0],[1,0,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]], np.float32)
    faces = np.array([[0,1,2],[3,4,5]], np.int64)
    mesh  = make_mesh(verts, faces)
    oct   = build_octree(mesh, max_depth=3, max_tris_per_leaf=1)

    ray_o = torch.tensor([[0.2, 0.2, 1.0]], dtype=torch.float32, device=DEVICE)
    ray_d = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=DEVICE)
    fn    = torch.zeros(2, 3, device=DEVICE)

    hf_b, d_b = ray_trace(ray_o, ray_d, mesh, fn, octree=None, batch_size=16)
    hf_o, d_o = ray_trace(ray_o, ray_d, mesh, fn, octree=oct,  batch_size=16)

    assert hf_b.item() == 0,   f'brute-force hit wrong face: {hf_b.item()}'
    assert abs(d_b.item() - 1.0) < 1e-5, f'brute-force wrong dist: {d_b.item()}'
    assert results_match(hf_b, d_b, hf_o, d_o), \
        f'octree mismatch: face {hf_b.item()} vs {hf_o.item()}, dist {d_b.item()} vs {d_o.item()}'
    print('PASS  test_single_hit')


def test_single_miss():
    """Ray misses the mesh entirely."""
    verts = np.array([[0,0,0],[1,0,0],[0,1,0]], np.float32)
    faces = np.array([[0,1,2]], np.int64)
    mesh  = make_mesh(verts, faces)
    oct   = build_octree(mesh, max_depth=2, max_tris_per_leaf=1)

    ray_o = torch.tensor([[10.0, 10.0, 1.0]], dtype=torch.float32, device=DEVICE)
    ray_d = torch.tensor([[0.0,  0.0, -1.0]], dtype=torch.float32, device=DEVICE)
    fn    = torch.zeros(1, 3, device=DEVICE)

    hf_b, d_b = ray_trace(ray_o, ray_d, mesh, fn, octree=None, batch_size=16)
    hf_o, d_o = ray_trace(ray_o, ray_d, mesh, fn, octree=oct,  batch_size=16)

    assert hf_b.item() == -1 and d_b.item() == -1.0, 'brute-force should miss'
    assert hf_o.item() == -1 and d_o.item() == -1.0, 'octree should miss'
    print('PASS  test_single_miss')


def test_nearest_hit():
    """Two coplanar triangles at different depths — ray should return the closer one."""
    verts = np.array([
        # z = 2  (farther)
        [0,0,2],[1,0,2],[0,1,2],
        # z = 1  (closer)
        [0,0,1],[1,0,1],[0,1,1],
    ], np.float32)
    faces = np.array([[0,1,2],[3,4,5]], np.int64)
    mesh  = make_mesh(verts, faces)
    oct   = build_octree(mesh, max_depth=4, max_tris_per_leaf=1)

    ray_o = torch.tensor([[0.2, 0.2, 0.0]], dtype=torch.float32, device=DEVICE)
    ray_d = torch.tensor([[0.0, 0.0,  1.0]], dtype=torch.float32, device=DEVICE)
    fn    = torch.zeros(2, 3, device=DEVICE)

    hf_b, d_b = ray_trace(ray_o, ray_d, mesh, fn, octree=None, batch_size=16)
    hf_o, d_o = ray_trace(ray_o, ray_d, mesh, fn, octree=oct,  batch_size=16)

    assert hf_b.item() == 1,              f'brute-force should hit face 1 (nearer): got {hf_b.item()}'
    assert abs(d_b.item() - 1.0) < 1e-5, f'brute-force wrong dist: {d_b.item()}'
    assert results_match(hf_b, d_b, hf_o, d_o), \
        f'octree mismatch: face {hf_b.item()} vs {hf_o.item()}'
    print('PASS  test_nearest_hit')


def test_batch_across_chunks():
    """R > batch_size so the outer loop runs multiple times."""
    mesh = flat_grid_mesh(n=20)
    oct  = build_octree(mesh, max_depth=6, max_tris_per_leaf=16)
    F    = mesh.faces_packed().shape[0]
    R    = 512

    torch.manual_seed(0)
    ray_o = torch.rand(R, 2, device=DEVICE) * 2 - 1
    ray_o = torch.cat([ray_o, torch.ones(R, 1, device=DEVICE) * 2], dim=1)
    ray_d = torch.zeros(R, 3, device=DEVICE); ray_d[:, 2] = -1.0
    fn    = torch.zeros(F, 3, device=DEVICE)

    hf_b, d_b = ray_trace(ray_o, ray_d, mesh, fn, octree=None, batch_size=128)
    hf_o, d_o = ray_trace(ray_o, ray_d, mesh, fn, octree=oct,  batch_size=128)

    assert results_match(hf_b, d_b, hf_o, d_o), \
        f'multi-batch mismatch: max face diff={(hf_b!=hf_o).sum()}, max dist diff={(d_b-d_o).abs().max():.2e}'
    print('PASS  test_batch_across_chunks')


def test_grid_mesh_correctness():
    """All R rays hit the flat grid; octree and brute-force must agree on every ray."""
    mesh = flat_grid_mesh(n=30)
    oct  = build_octree(mesh, max_depth=7, max_tris_per_leaf=16)
    F    = mesh.faces_packed().shape[0]
    R    = 1024

    torch.manual_seed(1)
    ray_o = torch.rand(R, 2, device=DEVICE) * 1.8 - 0.9
    ray_o = torch.cat([ray_o, torch.full((R, 1), 2.0, device=DEVICE)], dim=1)
    ray_d = torch.zeros(R, 3, device=DEVICE); ray_d[:, 2] = -1.0
    fn    = torch.zeros(F, 3, device=DEVICE)

    hf_b, d_b = ray_trace(ray_o, ray_d, mesh, fn, octree=None, batch_size=R)
    hf_o, d_o = ray_trace(ray_o, ray_d, mesh, fn, octree=oct,  batch_size=R)

    n_miss_b = (hf_b == -1).sum().item()
    n_miss_o = (hf_o == -1).sum().item()
    assert n_miss_b == 0, f'{n_miss_b} brute-force misses on full-coverage grid'
    assert n_miss_o == 0, f'{n_miss_o} octree misses on full-coverage grid'
    assert results_match(hf_b, d_b, hf_o, d_o), \
        f'grid mismatch: {(hf_b!=hf_o).sum()} faces differ, max dist diff={(d_b-d_o).abs().max():.2e}'
    print('PASS  test_grid_mesh_correctness')


def test_sphere_mesh_correctness():
    """Rays from outside a sphere; octree and brute-force must agree."""
    mesh = sphere_mesh(subdivisions=3)
    oct  = build_octree(mesh, max_depth=7, max_tris_per_leaf=16)
    F    = mesh.faces_packed().shape[0]
    R    = 512

    torch.manual_seed(2)
    # Rays from radius 3, pointing toward origin
    dirs = torch.randn(R, 3, device=DEVICE)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    ray_o = -dirs * 3.0
    ray_d =  dirs
    fn    = torch.zeros(F, 3, device=DEVICE)

    hf_b, d_b = ray_trace(ray_o, ray_d, mesh, fn, octree=None, batch_size=R)
    hf_o, d_o = ray_trace(ray_o, ray_d, mesh, fn, octree=oct,  batch_size=R)

    assert results_match(hf_b, d_b, hf_o, d_o), \
        f'sphere mismatch: {(hf_b!=hf_o).sum()} faces differ, max dist diff={(d_b-d_o).abs().max():.2e}'
    print('PASS  test_sphere_mesh_correctness')


def test_random_mesh_correctness():
    """Random triangles (pathological for octree pruning) — must still be correct."""
    torch.manual_seed(3)
    F = 5000
    verts = torch.randn(F * 3, 3, device=DEVICE)
    faces = torch.arange(F * 3, device=DEVICE).reshape(F, 3)
    mesh  = Meshes(verts=[verts], faces=[faces])
    v0 = verts[faces[:, 0]]
    mesh.edge_1 = verts[faces[:, 1]] - v0
    mesh.edge_2 = verts[faces[:, 2]] - v0
    oct = build_octree(mesh, max_depth=6, max_tris_per_leaf=32)

    R    = 256
    ray_o = torch.randn(R, 3, device=DEVICE) * 5
    ray_d = torch.randn(R, 3, device=DEVICE)
    ray_d /= ray_d.norm(dim=-1, keepdim=True)
    fn    = torch.zeros(F, 3, device=DEVICE)

    hf_b, d_b = ray_trace(ray_o, ray_d, mesh, fn, octree=None, batch_size=R)
    hf_o, d_o = ray_trace(ray_o, ray_d, mesh, fn, octree=oct,  batch_size=R)

    assert results_match(hf_b, d_b, hf_o, d_o), \
        f'random-mesh mismatch: {(hf_b!=hf_o).sum()} faces, max dist={(d_b-d_o).abs().max():.2e}'
    print('PASS  test_random_mesh_correctness')


def test_behind_ray_origin():
    """Intersections behind the ray origin (t < 0) must be ignored."""
    # Triangle at z = -1, ray points in +z
    verts = np.array([[0,0,-1],[1,0,-1],[0,1,-1]], np.float32)
    faces = np.array([[0,1,2]], np.int64)
    mesh  = make_mesh(verts, faces)
    oct   = build_octree(mesh, max_depth=2, max_tris_per_leaf=1)

    ray_o = torch.tensor([[0.2, 0.2, 0.0]], dtype=torch.float32, device=DEVICE)
    ray_d = torch.tensor([[0.0, 0.0,  1.0]], dtype=torch.float32, device=DEVICE)
    fn    = torch.zeros(1, 3, device=DEVICE)

    hf_b, d_b = ray_trace(ray_o, ray_d, mesh, fn, octree=None, batch_size=16)
    hf_o, d_o = ray_trace(ray_o, ray_d, mesh, fn, octree=oct,  batch_size=16)

    assert hf_b.item() == -1, f'brute-force should miss (behind origin): face={hf_b.item()}'
    assert hf_o.item() == -1, f'octree should miss (behind origin): face={hf_o.item()}'
    print('PASS  test_behind_ray_origin')


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(label, mesh, oct, ray_o, ray_d, batch_size):
    F  = mesh.faces_packed().shape[0]
    fn = torch.zeros(F, 3, device=DEVICE)

    t_b = time_fn(lambda: ray_trace(ray_o, ray_d, mesh, fn, octree=None, batch_size=batch_size))
    t_o = time_fn(lambda: ray_trace(ray_o, ray_d, mesh, fn, octree=oct,  batch_size=batch_size))

    print(f'BENCH {label:40s}  '
          f'brute {t_b*1000:7.1f} ms  '
          f'octree {t_o*1000:7.1f} ms  '
          f'speedup {t_b/t_o:5.2f}x')


def bench_grid():
    mesh = flat_grid_mesh(n=70)       # 9800 triangles
    oct  = build_octree(mesh, max_depth=7, max_tris_per_leaf=16)
    F    = mesh.faces_packed().shape[0]
    R    = 1024
    torch.manual_seed(0)
    ray_o = torch.rand(R, 2, device=DEVICE) * 1.8 - 0.9
    ray_o = torch.cat([ray_o, torch.full((R,1), 2.0, device=DEVICE)], dim=1)
    ray_d = torch.zeros(R, 3, device=DEVICE); ray_d[:, 2] = -1.0
    benchmark(f'flat grid  F={F}  B={R}', mesh, oct, ray_o, ray_d, R)


def bench_sphere_small():
    mesh = sphere_mesh(subdivisions=3)   # ~1280 triangles
    oct  = build_octree(mesh, max_depth=6, max_tris_per_leaf=16)
    F    = mesh.faces_packed().shape[0]
    R    = 1024
    torch.manual_seed(1)
    dirs = torch.randn(R, 3, device=DEVICE); dirs /= dirs.norm(dim=-1, keepdim=True)
    ray_o = -dirs * 3.0; ray_d = dirs
    benchmark(f'sphere     F={F}  B={R}', mesh, oct, ray_o, ray_d, R)


def bench_sphere_large():
    mesh = sphere_mesh(subdivisions=5)   # ~20K triangles
    oct  = build_octree(mesh, max_depth=7, max_tris_per_leaf=32)
    F    = mesh.faces_packed().shape[0]
    R    = 1024
    torch.manual_seed(2)
    dirs = torch.randn(R, 3, device=DEVICE); dirs /= dirs.norm(dim=-1, keepdim=True)
    ray_o = -dirs * 3.0; ray_d = dirs
    benchmark(f'sphere     F={F}  B={R}', mesh, oct, ray_o, ray_d, R)


def bench_large_rays():
    """Many total rays (R=2^20) batched at a memory-safe size."""
    mesh = flat_grid_mesh(n=50)      # 5000 triangles
    oct  = build_octree(mesh, max_depth=7, max_tris_per_leaf=16)
    F    = mesh.faces_packed().shape[0]
    R    = 2**20
    # batch_size=2^12 keeps peak GPU memory manageable for both paths
    BS   = 2**12
    torch.manual_seed(3)
    ray_o = torch.rand(R, 2, device=DEVICE) * 1.8 - 0.9
    ray_o = torch.cat([ray_o, torch.full((R,1), 2.0, device=DEVICE)], dim=1)
    ray_d = torch.zeros(R, 3, device=DEVICE); ray_d[:, 2] = -1.0
    benchmark(f'flat grid  F={F}  R={R}  batch={BS}', mesh, oct, ray_o, ray_d, BS)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'device: {DEVICE}\n')

    print('─── Correctness ─────────────────────────────────────────────────')
    test_single_hit()
    test_single_miss()
    test_nearest_hit()
    test_batch_across_chunks()
    test_grid_mesh_correctness()
    test_sphere_mesh_correctness()
    test_random_mesh_correctness()
    test_behind_ray_origin()

    print()
    print('─── Benchmarks ──────────────────────────────────────────────────')
    bench_grid()
    bench_sphere_small()
    bench_sphere_large()
    bench_large_rays()
