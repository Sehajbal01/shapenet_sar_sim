from my_ray_tracer.geometry.trimesh import TriMesh
from my_ray_tracer.camera.orthographic import OrthographicCamera
from my_ray_tracer.accelerator.octree import Octree
from my_ray_tracer.geometry.bbox import BBox
import torch
import imageio
import time

tik = time.time()

W, H = 400, 300
DEVICE_ID = 0  # which gpu to use

device = torch.device(f"cuda:{DEVICE_ID}")
print(f"Using device: {device}")

obj_trimesh = TriMesh()
obj_trimesh.load_obj_file("/workspace/data/srncars/02958343/fcd90d547fdeb629f200a72c9245aee7/models/model_normalized.obj")
obj_trimesh = obj_trimesh.to(device)
print("Building Octree from mesh...")
octree = Octree(
    max_depth=2,
    approx_trig_per_bbox=256,
    mesh=obj_trimesh,
    device=device
)

# === hack to add a ground ===
# add two triangles as ground to all octree leaf elements
# needs to be done after octree construction because we want octree to be based on scale of the obj file
ground_size = 1e5
min_pt, _ = obj_trimesh.get_bounds()
lowest_y = min_pt[1].item()
# y is lowest point of obj file, x and z are ground_size
ground_trig_1 = torch.tensor([[-ground_size, lowest_y, -ground_size],
                              [-ground_size, lowest_y, ground_size],
                              [ground_size, lowest_y, ground_size]]).to(device)
ground_trig_2 = torch.tensor([[ground_size, lowest_y, ground_size],
                              [ground_size, lowest_y, -ground_size],
                              [-ground_size, lowest_y, -ground_size]]).to(device)
# add them to mesh
obj_trimesh.triangles_A = torch.cat([obj_trimesh.triangles_A, ground_trig_1[0:1], ground_trig_2[0:1]], dim=0)
obj_trimesh.triangles_B = torch.cat([obj_trimesh.triangles_B, ground_trig_1[1:2], ground_trig_2[1:2]], dim=0)
obj_trimesh.triangles_C = torch.cat([obj_trimesh.triangles_C, ground_trig_1[2:3], ground_trig_2[2:3]], dim=0)
obj_trimesh.triangles_edge1 = obj_trimesh.triangles_B - obj_trimesh.triangles_A  # recompute
obj_trimesh.triangles_edge2 = obj_trimesh.triangles_C - obj_trimesh.triangles_A
obj_trimesh.triangles_normal = torch.cross(obj_trimesh.triangles_edge1, obj_trimesh.triangles_edge2)
# add them to octree
num_trigs = len(obj_trimesh.triangles_A)
last_two_indices = torch.tensor([num_trigs - 2, num_trigs - 1], device=device)
def add_ground(octree_node):
    if octree_node.is_leaf:
        octree_node.triangles = torch.cat([octree_node.triangles, last_two_indices], dim=0)
    else:
        for child in octree_node.children:
            if child is not None:
                add_ground(child)
add_ground(octree)
# add two octree nodes because lots of primary rays miss the first octree right now
octree_parent = Octree(
    max_depth=0,
    approx_trig_per_bbox=256,
    mesh=obj_trimesh,
    device=device,
    bbox=BBox(torch.full((3,), -ground_size, dtype=torch.float32, device=device),
              torch.full((3,), ground_size, dtype=torch.float32, device=device)),
    triangles=[]
)
octree_parent.is_leaf = False
octree_second_child = Octree(
    max_depth=0,
    approx_trig_per_bbox=256,
    mesh=obj_trimesh,
    device=device,
    bbox=BBox(torch.full((3,), -ground_size, dtype=torch.float32, device=device),
              torch.full((3,), ground_size, dtype=torch.float32, device=device)),
    triangles=last_two_indices
)
octree_parent.children[0] = octree
octree_parent.children[1] = octree_second_child
# start tracing from octree_parent
octree = octree_parent

# ============================

print("Generating camera rays...")
ortho_cam = OrthographicCamera(
    torch.Tensor([10, 10, 10]),  # position
    torch.Tensor([-1, -1, -1]) / torch.sqrt(torch.tensor(3.0)),  # direction
    4.0/2.0,  # sensor width in world space
    3.0/2.0,  # sensor height in world space
    W,  # number of rays to shoot in width dimension
    H,  # number of rays to shoot in height dimension
)
ray_origins, ray_directions = ortho_cam.generate_rays()

ray_origins = ray_origins.to(device)
ray_directions = ray_directions.to(device)

# reshape it down
ray_origins = ray_origins.reshape(H * W, 3)
ray_directions = ray_directions.reshape(H * W, 3)

# intersect rays with octree (which intersects with triangles inside)
print("Intersecting all rays with the Octree...")
ray_hit_times, ray_hit_triangle_ids = octree.intersect_rays(ray_origins, ray_directions)

ray_hit_times[ray_hit_times < 0] = torch.max(ray_hit_times)
ray_hit_times -= torch.min(ray_hit_times)
ray_hit_times /= torch.max(ray_hit_times)
ray_hit_times = 1 - ray_hit_times  # invert to make further = darker

# reshape it back to image
ray_hit_times = ray_hit_times.reshape(H, W)
imageio.imwrite("output_depth.png", (ray_hit_times.cpu().detach().numpy() * 255).astype("uint8"))

# create a diffuse image
diffuse_image = torch.zeros((H*W,), dtype=torch.float32, device=device)
mask = torch.argwhere(ray_hit_triangle_ids != -1).squeeze()
normals = obj_trimesh.triangles_normal[ray_hit_triangle_ids[mask]]
diffuse_image[mask] = torch.sum(-ortho_cam.direction.to(normals.device) * normals, dim=1)
diffuse_image[mask] = torch.abs(diffuse_image[mask])  # we do not differentiate between front and back side of triangle
diffuse_image[mask] = torch.clamp(diffuse_image[mask], 0, 1)
diffuse_image /= torch.max(diffuse_image)
diffuse_image = diffuse_image.reshape(H, W)

imageio.imwrite("output_diffuse.png", (diffuse_image.cpu().detach().numpy() * 255).astype("uint8"))

tok = time.time()
print("Total time: ", tok - tik)
