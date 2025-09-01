import torch
import imageio
from my_ray_tracer.core.scene import Scene
from my_ray_tracer.camera.orthographic import OrthographicCamera
import matplotlib.pyplot as plt

W, H = 400, 300
DEVICE_ID = 0  # which gpu to use

device = torch.device(f"cuda:{DEVICE_ID}")
print(f"Using device: {device}")

ortho_cam = OrthographicCamera(
    torch.Tensor([1, 1, 1]) / torch.sqrt(torch.tensor(3.0)) * 1.3,  # position
    torch.Tensor([-1, -1, -1]) / torch.sqrt(torch.tensor(3.0)),  # direction
    4.0/2.0,  # sensor width in world space
    3.0/2.0,  # sensor height in world space
    W,  # number of rays to shoot in width dimension
    H,  # number of rays to shoot in height dimension
)

scene = Scene(
    obj_filename="/workspace/data/srncars/02958343/fcd90d547fdeb629f200a72c9245aee7/models/model_normalized.obj",
    device=device,
)
scene.add_ground()
depth, diffuse = scene.get_depth_and_diffuse(ortho_cam)
imageio.imwrite("output_depth.png", depth)
imageio.imwrite("output_diffuse.png", diffuse)

# now lets test the energy range
num_bounces = 3
energy_range_values = scene.get_energy_range_values([ortho_cam], num_bounces=num_bounces)
e_r_values = energy_range_values[0]  # list[(n, 2)]  # just the first camera for now

for i in range(len(e_r_values)):
    xy = e_r_values[i].cpu().numpy()
    plt.scatter(xy[:, 0], xy[:, 1], s=1)
    plt.savefig(f"energy_range_bounce_{i}.png")
    plt.close()
