import matplotlib.pyplot as plt
import PIL
import os
import torch
import numpy as np
import imageio

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings, 
    MeshRasterizer,  
)



def get_next_path(path):
    """
    Get the next available path by appending a number to the base path.
    If the path already exists, it will increment the number until it finds an available one.
    """
    split_path = path.split('.')
    assert len(split_path) > 1, "Path must have an extension to append a number."
    base_path = '.'.join(split_path[:-1])
    extension = split_path[-1]
    
    i = 0
    while True:
        new_path = '%s_%02d.%s' % (base_path, i, extension)
        if not os.path.exists(new_path):
            return new_path
        i += 1





def extract_pose_info(target_poses, format='srn_cars'):
    '''
    Extracts camera position and orientation from the target poses.

    inputs:
        target_poses (...,4,4): the camera poses in world coordinates
    outputs:
        see the return statement below
    '''
    if format == 'srn_cars':
        # extract vectors
        cam_center    = target_poses[..., :3, 3] # (...,3)
        cam_right     = target_poses[..., :3, 0] # (...,3)
        cam_up        = target_poses[..., :3, 1] # (...,3)
        cam_forward   = target_poses[..., :3, 2] # (...,3)

        # calculate distance, azimuth, and elevation
        cam_distance = torch.norm(cam_center, dim=-1)  # (...,)
        cam_elevation = torch.asin(cam_center[..., 2] / cam_distance)  # (...)
        cam_azimuth = torch.acos(cam_center[..., 0] / cam_distance / torch.cos(cam_elevation))  # (...)
        cam_azimuth = torch.where(cam_center[..., 1] < 0, 2 * np.pi - cam_azimuth, cam_azimuth)  # (...)

    else:
        raise NotImplementedError("Unknown format for extract_pose_info(): %s" % format)

    return cam_center, cam_right, cam_up, cam_forward, cam_distance, cam_elevation, cam_azimuth


if __name__ == '__main__':
    # from the pytorch3d tutorial: https://pytorch3d.org/tutorials/render_textured_meshes

    # constants
    device          = 'cuda'
    half_side_len   = 0.5
    n_rays_per_side = 128
    azimuth_spread  = 0
    num_pulse       = 1
    
    # get an object and pose
    all_obj_id = os.listdir('/workspace/data/srncars/cars_train/')  # list all object IDs in the dataset
    obj_id     = np.random.choice(all_obj_id, 1)[0]  # randomly select an object ID from the dataset
    print('Selected object ID: ', obj_id)

    all_pose_paths = '/workspace/data/srncars/cars_train/%s/pose/'%obj_id
    all_pose_nums  = os.listdir(all_pose_paths)
    pose_num       = np.random.choice(all_pose_nums, 1)[0].split('.')[0]
    print('Selected pose number: ', pose_num)

    # load image, pose, and mesh
    rgb_path  = '/workspace/data/srncars/cars_train/%s/rgb/%s.png' % (obj_id, pose_num)
    pose_path = '/workspace/data/srncars/cars_train/%s/pose/%s.txt' % (obj_id, pose_num)
    mesh_path = '/workspace/data/srncars/02958343/%s/models/model_normalized.obj' % obj_id
    rgb  = np.array(PIL.Image.open(rgb_path))[...,:3][...,:3]
    pose = np.loadtxt(pose_path).reshape(1,4,4).astype(np.float32)  # (4, 4)
    mesh = load_objs_as_meshes([mesh_path], device=device)

    # get azimuth, elevation, and distance from the pose
    target_poses = torch.tensor(pose, device=device) # (1, 4, 4)
    cam_center, cam_right, cam_up, cam_forward, distance, elevation, azimuth = extract_pose_info(target_poses, format='srn_cars')
    azimuth *= 180 / np.pi  # convert to degrees
    elevation *= 180 / np.pi  # convert to degrees
    # (1,3)     (1,3)      (1,3)   (1,3)        (1,)      (1,)       (1,)


    ############################ replicating the srn pose ############################
    # get verticies
    verts      = mesh.verts_packed()  # (V, 3)
    faces      = mesh.faces_packed()  # (F, 3)
    face_verts = verts[faces]  # (F, 3, 3)

    # compute face normals for all faces
    v0, v1, v2   = face_verts[:, 0], face_verts[:, 1], face_verts[:,2] # (F, 3), (F, 3), (F, 3)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
    face_normals = torch.nn.functional.normalize(face_normals, dim=1)  # (F, 3)

    # prepare rasterization settings
    raster_settings = RasterizationSettings(
        image_size=n_rays_per_side, 
        blur_radius=0.0, 
        faces_per_pixel=1, 

        bin_size=0,  # or set to a small value
        max_faces_per_bin=100000  # try increasing from the default (e.g., 10000)
    )


    # perform rasterization to find where the rays hit the mesh
    rotation, translation = look_at_view_transform(distance.item(), elevation.item(), 90+azimuth.item(), device=device) # distance, elevation, azimuth
    cameras = FoVOrthographicCameras(device=device, R=rotation, T=translation, 
                                 min_x = -half_side_len, max_x = half_side_len,
                                 min_y = -half_side_len, max_y = half_side_len,
                                 )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    fragments = rasterizer(mesh)

    # get depth map
    depth_map  = fragments.zbuf[0, ..., 0]    # (r, r) # missed rays are -1.0

    # compute surface normals from face indices and mesh vertices/faces
    face_ids = fragments.pix_to_face[0, ..., 0]  # (r, r) face indices
    hit = (depth_map >= 0) # (r, r) valid hits
    
    # create normal map by indexing face normals with face IDs
    valid_face_ids = face_ids[hit] # (R',)

    # compute returned energy (cosine similarity between ray direction and surface normal * alpha_1 + alpha_2)
    # ray direction is the same for all rays because we are using orthographic projection, so we can simply grab the forward vector from the rotation matrix
    forward_vector = rotation[0,:,2] # (3,)
    energy_map = torch.zeros(n_rays_per_side, n_rays_per_side, device=device)  # (r, r)
    energy_map[hit] = torch.abs(torch.sum(face_normals[valid_face_ids] * forward_vector, dim=-1)) # (r, r)

    # produce a frame of the depth and energy maps
    masked_dm = depth_map[hit]
    masked_dm = masked_dm - masked_dm.min()  # shift to start from 0
    masked_dm = masked_dm / masked_dm.max()  # normalize to [0, 1]
    masked_dm = 1 - masked_dm  # invert the depth map
    dm_im = torch.zeros((n_rays_per_side, n_rays_per_side), device=device)  # (r, r)
    dm_im[hit] = masked_dm  # apply the mask

    masked_e = energy_map[hit]
    masked_e = masked_e - masked_e.min()  # shift to start from 0
    masked_e = masked_e / masked_e.max()  # normalize to [0,1]
    e_im = torch.zeros((n_rays_per_side, n_rays_per_side), device=device)
    e_im[hit] = masked_e  # apply the mask

    # normalize the depth, energy, and RGB images
    e_im = e_im.cpu().numpy()  # convert to numpy for saving
    dm_im = dm_im.cpu().numpy()  # convert to numpy for saving
    dm_e_im = np.concatenate((dm_im, e_im), axis=1)  # concatenate depth and energy maps horizontally
    dm_e_im = (dm_e_im * 255).astype(np.uint8)  # scale to [0, 255] for saving

    # repeat for 3 channels and concatenate the rgb image
    dm_e_im = np.tile(dm_e_im[..., np.newaxis], (1, 1, 3))  # (r, r, 3)
    print('rgb.dtype: ', rgb.dtype)
    print('rgb.shape: ', rgb.shape)
    dm_e_rgb_im = np.concatenate((dm_e_im, rgb.astype(np.uint8)), axis=1)
    PIL.Image.fromarray(dm_e_rgb_im).save(get_next_path('figures/depth_energy_rgb.png'))  # save the image
    ############################ replicating the srn pose ############################




    # # sar rendering
    # all_azimuths = np.linspace(azimuth - azimuth_spread, azimuth + azimuth_spread, num_pulse)  # (num_pulse,)
    # print('Azimuths: ', all_azimuths)
    # print('Elevation: ', elevation)

    # images = []
    # for azimuth in all_azimuths:

    #     # perform rasterization to find where the rays hit the mesh
    #     rotation, translation = look_at_view_transform(distance, elevation, azimuth, device=device) # distance, elevation, azimuth
    #     cameras = FoVOrthographicCameras(device=device, R=rotation, T=translation, 
    #                                  min_x = -half_side_len, max_x = half_side_len,
    #                                  min_y = -half_side_len, max_y = half_side_len,
    #                                  )
    #     rasterizer = MeshRasterizer(
    #         cameras=cameras,
    #         raster_settings=raster_settings
    #     )
    #     fragments = rasterizer(mesh)

    #     # get depth map
    #     depth_map  = fragments.zbuf[0, ..., 0]    # (r, r) # missed rays are -1.0

    #     # compute surface normals from face indices and mesh vertices/faces
    #     face_ids = fragments.pix_to_face[0, ..., 0]  # (r, r) face indices
    #     hit = (depth_map >= 0) # (r, r) valid hits
    
    #     # create normal map by indexing face normals with face IDs
    #     valid_face_ids = face_ids[hit] # (R',)

    #     # compute returned energy (cosine similarity between ray direction and surface normal * alpha_1 + alpha_2)
    #     # ray direction is the same for all rays because we are using orthographic projection, so we can simply grab the forward vector from the rotation matrix
    #     forward_vector = rotation[0,:,2] # (3,)
    #     energy_map = torch.zeros(n_rays_per_side, n_rays_per_side, device=device)  # (r, r)
    #     energy_map[hit] = torch.abs(torch.sum(face_normals[valid_face_ids] * forward_vector, dim=-1)) # (r, r)

    #     # produce a frame of the depth and energy maps
    #     masked_dm = depth_map[hit]
    #     masked_dm = masked_dm - masked_dm.min()  # shift to start from 0
    #     masked_dm = masked_dm / masked_dm.max()  # normalize to [0, 1]
    #     masked_dm = 1 - masked_dm  # invert the depth map
    #     dm_im = torch.zeros((n_rays_per_side, n_rays_per_side), device=device)  # (r, r)
    #     dm_im[hit] = masked_dm  # apply the mask

    #     masked_e = energy_map[hit]
    #     masked_e = masked_e - masked_e.min()  # shift to start from 0
    #     masked_e = masked_e / masked_e.max()  # normalize to [0,1]
    #     e_im = torch.zeros((n_rays_per_side, n_rays_per_side), device=device)
    #     e_im[hit] = masked_e  # apply the mask

    #     e_im = e_im.cpu().numpy()  # convert to numpy for saving
    #     dm_im = dm_im.cpu().numpy()  # convert to numpy for saving
    #     dm_e_im = np.concatenate((dm_im, e_im), axis=1)  # concatenate depth and energy maps horizontally
    #     dm_e_im = (dm_e_im * 255).astype(np.uint8)  # scale to [0, 255] for saving

    #     # save the image
    #     images.append(dm_e_im)

    # # save the images as a gif
    # gif_path = get_next_path('figures/depth_energy_maps.gif')
    # fps = len(images) / 15.0
    # print('Saving GIF with %.1f fps...' % fps)
    # imageio.mimsave(gif_path, images, fps=fps, format='GIF', loop=0)
