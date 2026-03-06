try:
    import torch
except ImportError:
    print("PyTorch not installed.\nPlease activate your conda environment (e.g. `conda activate sarrender`) and make sure `torch` is available.")
    import sys
    sys.exit(1)

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRasterizer,)
from signal_simulation import load_mesh
from utils import extract_pose_info
from tqdm import tqdm as tqdm
import numpy as np
import cv2
import os
import sys



def ideal_height_map_render(
    file_name=None,                    # path to obj (ignored if `mesh` is supplied)
    poses=None,
    mesh=None,                         # optional pre-loaded mesh returned by load_mesh

    # image size stuff
    image_width=128,
    image_height=128,
    image_plane_width=1,
    image_plane_height=1,

    override_obj_path=None,           # for debugging purposes
):
    '''
    Render an ideal height map from the given mesh and camera poses. 
    It's basically a top-down orthoographic depth map, rotated according to the azimuth

    Args:
        file_name: path to the obj file to render
        poses: (..., 4, 4) tensor of camera poses
        image_width: width of the output image in pixels
        image_height: height of the output image in pixels
        image_plane_width: width of the image plane in world units (e.g., meters)
        image_plane_height: height of the image plane in world units (e.g., meters)
    
    Returns:
        height_maps: (..., r, r) tensor of height maps

    '''
    
    # allow overriding the obj path for debugging purposes
    mesh_scale = None
    if override_obj_path is not None:
        file_name = override_obj_path
        mesh_scale = 0.07

    # set device
    device = poses.device

    # extract camera pose info
    cam_center, cam_right, cam_up, cam_forward, cam_distance, cam_elevation_deg, cam_azimuth_deg = extract_pose_info(poses)

    # load the mesh (unless one was provided)
    if mesh is None:
        mesh, normals, material_properties = load_mesh(
            file_name,
            device=device,
            make_ground=True,
            scale=mesh_scale,
        )
    else:
        # assume the caller already put everything on the correct device
        normals = None
        material_properties = None

    # prepare rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(image_height, image_width), 
        blur_radius=0.0, 
        faces_per_pixel=1, 

        bin_size=0,  # set to 0 to use naive rasterization and avoid bin overflow
        max_faces_per_bin=100000  # try increasing from the default (e.g., 10000)
    )

    # get shape info and flatten pose to (N,4,4)
    shape_prefix = poses.shape[:-2]
    poses = poses.reshape(-1, 4, 4)
    N = poses.shape[0]

    # we don't need gradients for rendering
    with torch.no_grad():
        # compute the camera transforms for all poses at once
        # elevation is fixed to 90° for a top‑down view; azimuth rotates with the pose
        rotation, translation = look_at_view_transform(
            cam_distance,              # (...,) or scalar
            90,                        # elevation
            cam_azimuth_deg + 90,      # azimuth
            device=device,
        )

        # build a batched camera and rasterizer on the correct device
        cameras = FoVOrthographicCameras(
            device=device,
            R=rotation,
            T=translation,
            min_x=-image_plane_width/2, max_x=image_plane_width/2,
            min_y=-image_plane_height/2, max_y=image_plane_height/2,
        )
        # MeshRasterizer does not accept a `device` argument; it uses
        # the device of the cameras and mesh tensors instead.
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        # ensure mesh lives on the correct device before batching
        mesh = mesh.to(device)
        mesh_batch = mesh.extend(N)
        fragments = rasterizer(mesh_batch)

        # extract height maps (N, H, W)
        height_maps = fragments.zbuf[..., 0]
    height_maps = height_maps.reshape(*shape_prefix, image_height, image_width)
    return height_maps



def make_height_map_dataset(
    dataset_dir = '/home/berian/Documents/shapenet/cars_train/',
    models_dir  = '/home/berian/Documents/shapenet/object-models/02958343/',
    device = 'cuda',
):
    # normalise device string
    device = torch.device(device)
    '''
    Make a dataset of height maps for all the objects and poses in the dataset. 
    This is used for training the height map predictor.
    '''
    all_obj_id = os.listdir(dataset_dir)  # list all object IDs in the dataset

    # loop through each object ID and render height maps for all poses
    # for obj_id in all_obj_id:
    for obj_id in tqdm(all_obj_id, desc='Rendering height maps'):
        
        # get .obj path for this object
        mesh_path = os.path.join(models_dir, obj_id, 'models', 'model_normalized.obj')

        # get all pose numbers for this object
        all_pose_paths = os.path.join(dataset_dir,obj_id,'pose')
        all_pose_nums  = os.listdir(all_pose_paths)
        all_pose_nums = [pose_num.split('.')[0] for pose_num in all_pose_nums] # remove .txt extension

        # print(f"Object {obj_id} has {len(all_pose_nums)} poses")  # debug: check number of poses per object

        # read all poses into a list
        pose_tensors = []
        for pose_num in all_pose_nums:
            pose_path = os.path.join(dataset_dir, obj_id, 'pose', '%s.txt' % pose_num)
            with open(pose_path, 'r') as f:
                pose_lines = f.readlines()
            pose = torch.tensor([float(x) for x in pose_lines[0].split()], device=device).reshape(4, 4)
            pose_tensors.append(pose)

        # preload the mesh once and keep it on the device
        mesh_cache, _, _ = load_mesh(mesh_path, device=device, make_ground=True)

        # print(f"Mesh has {mesh_cache.verts_packed().shape[0]} vertices")  # debug: check mesh size

        # process poses in batches to avoid OOM
        batch_size = 10  # increased back to 10 with binning enabled for speed
        for i in range(0, len(pose_tensors), batch_size):
            pose_batch = pose_tensors[i:i + batch_size]
            pose_nums_batch = all_pose_nums[i:i + batch_size]
            poses_tensor = torch.stack(pose_batch, dim=0)  # (batch_size, 4, 4)

            # render height maps for this batch
            height_maps = ideal_height_map_render(mesh=mesh_cache, poses=poses_tensor)  # (batch_size, H, W)

            # save each result individually
            for j, pose_num in enumerate(pose_nums_batch):
                height_map = height_maps[j]
                height_map_path = os.path.join(dataset_dir, obj_id, 'height_map', '%s.npy' % pose_num)
                os.makedirs(os.path.dirname(height_map_path), exist_ok=True)
                np.save(height_map_path, height_map.cpu().numpy())

            # clear cache to free memory
            torch.cuda.empty_cache()



if __name__ == '__main__':
    print('Making height map for train dataset...')
    make_height_map_dataset('/home/berian/Documents/shapenet/cars_train/')
    print('Making height map for test dataset...')
    make_height_map_dataset('/home/berian/Documents/shapenet/cars_test/')
    print('Making height map for val dataset...')
    make_height_map_dataset('/home/berian/Documents/shapenet/cars_val/')