import torch
import numpy as np
from pytorch3d.structures import Meshes
from ray_tracer_v2 import ray_trace, build_octree
from utils import generate_pose_mat, plot_image, savefig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

# --- single large equilateral triangle on the XY plane, centred at origin ---
R = 1.5  # circumradius (metres)
verts = torch.tensor([
    [ 0.0,                  R,   0.0],   # top
    [-R * np.sqrt(3) / 2,  -R / 2, 0.0],   # bottom-left
    [ R * np.sqrt(3) / 2,  -R / 2, 0.0],   # bottom-right
], device=device, dtype=torch.float32)       # (3, 3)
faces = torch.tensor([[0, 1, 2]], device=device, dtype=torch.long)  # (1, 3)

v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
edge_1       = v1 - v0                                                         # (1, 3)
edge_2       = v2 - v0                                                         # (1, 3)
face_normals = torch.nn.functional.normalize(torch.cross(edge_1, edge_2, dim=1), dim=1)  # (1, 3)

mesh         = Meshes(verts=[verts], faces=[faces])
mesh.edge_1  = edge_1
mesh.edge_2  = edge_2

octree = build_octree(mesh)

# --- camera pose ---
azimuth   = 30.0   # degrees
elevation = 30.0   # degrees
distance  = 3.0    # metres

pose           = generate_pose_mat(azimuth, elevation, distance, device=device)
right_vector   = pose[:3, 0]
up_vector      = pose[:3, 1]
forward_vector = pose[:3, 2]
sensor_pos     = pose[:3, 3]

print(f"sensor_pos:     {sensor_pos.tolist()}")
print(f"forward_vector: {forward_vector.tolist()}")
print(f"right_vector:   {right_vector.tolist()}")
print(f"up_vector:      {up_vector.tolist()}")
print(f"face_normals:   {face_normals.tolist()}")

# --- ray grid (orthographic, top→bottom row convention) ---
n_ray_width  = 50
n_ray_height = 50
grid_width   = 2.0   # metres
grid_height  = 2.0   # metres

x_offsets = torch.linspace(-grid_width / 2,  grid_width / 2,  n_ray_width,  device=device)
y_offsets = torch.linspace( grid_height / 2, -grid_height / 2, n_ray_height, device=device)
grid_y, grid_x = torch.meshgrid(y_offsets, x_offsets, indexing='ij')   # (H, W)
origins    = (sensor_pos.reshape(1, 1, 3)
              + grid_x.unsqueeze(-1) * right_vector
              + grid_y.unsqueeze(-1) * up_vector).reshape(-1, 3)         # (H*W, 3)
directions = forward_vector.unsqueeze(0).expand(origins.shape[0], -1)   # (H*W, 3)

# --- ray trace ---
hit_face_ids, distances = ray_trace(origins, directions, mesh, face_normals, octree=octree)

n_hits = (distances >= 0).sum().item()
print(f"hits: {n_hits} / {distances.numel()}")
if n_hits > 0:
    d_hit = distances[distances >= 0]
    print(f"depth range: {d_hit.min().item():.4f} m  –  {d_hit.max().item():.4f} m")

# --- depth map (missed rays → NaN so they show as blank) ---
depth_map = distances.reshape(n_ray_height, n_ray_width).clone()
depth_map[depth_map < 0] = float('nan')

fig, ax = plot_image(depth_map.cpu().numpy(),
                     title=f"Single triangle  az={azimuth}°  el={elevation}°  dist={distance}m",
                     cmap='viridis')
savefig('figures/test_raytrace_triangle.png')
print("saved figures/test_raytrace_triangle.png")
