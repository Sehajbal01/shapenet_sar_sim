import matplotlib.pyplot as plt
import os
import imageio
import torch
import numpy as np

from utils import get_next_path, extract_pose_info
from ray_tracer.core.scene import Scene
from ray_tracer.camera.orthographic import OrthographicCamera




def accumulate_scatters(target_poses,
                        mesh, face_normals, material_properties,
                        trajectory,
                        wavelength=None,
                        debug_gif=False,
                        grid_width=1, grid_height=1,
                        n_ray_width=1, n_ray_height=1,
                        num_bounces=2,
                    ):
    '''
    returns the energy and range for a bunch of rays for each pulse

    inputs:
        target_poses (T,4,4): the camera poses for which we want to get a sar image from
        mesh: ray tracer Scene object containing the 3D model
        face_normals: unused (embedded in the Scene object)
        material_properties: unused (embedded in the Scene object)
        trajectory (T,P,3): sensor positions in Z-up cartesian coordinates for each pulse for each view
        wavelength (float): wavelength of the radar signal; if None, energy is real-valued
        debug_gif (bool): whether to save debug images of depth, energy, and range
        grid_width (float): sensor width in world space for the orthographic camera
        grid_height (float): sensor height in world space for the orthographic camera
        n_ray_width (int): number of rays along the width dimension
        n_ray_height (int): number of rays along the height dimension
        num_bounces (int): number of ray bounces to simulate

    outputs:
        scatter_ranges (T,P,R): range of each ray return
        scatter_energies (T,P,R): simulated energy of each ray return

    '''
    scene = mesh

    device = target_poses.device
    T = target_poses.shape[0]   # no. of camera views
    P = trajectory.shape[1]     # no. of pulses per view


    # loop over each pulse and compute the depth map and surface normal
    scatter_ranges = []
    scatter_energies = []
    for t in range(T):  # for each camera
        scatter_ranges.append([])
        scatter_energies.append([])

        # construct P number of cameras due to azimuth spread
        cameras = []
        for p in range(P):  # for each pulse
            pos = trajectory[t, p]
            position_vector = torch.stack([pos[0], pos[2], -pos[1]])  # Z-up (trajectory) -> Y-up (ray tracer)
            direction_vector = -position_vector / torch.norm(position_vector)
            ortho_cam = OrthographicCamera(
                position_vector.cpu(),  # position
                direction_vector.cpu(),  # direction
                grid_width,  # sensor width in world space
                grid_height,  # sensor height in world space
                n_ray_width,  # number of rays to shoot in width dimension
                n_ray_height,  # number of rays to shoot in height dimension
            )
            cameras.append(ortho_cam)
        
        # # trace rays in parallel
        # with torch.no_grad():
        #     energy_range_values = scene.get_energy_range_values(cameras, num_bounces=num_bounces)

        # trace rays somewhat in parallel
        with torch.no_grad():
            num_cams_at_once = int(5 / (num_bounces+1))
            energy_range_values = []
            for i in range(0, len(cameras), num_cams_at_once):
                energy_range_values.extend(scene.get_energy_range_values(cameras[i:i+num_cams_at_once], num_bounces=num_bounces, debug=debug_gif))

        if debug_gif:
            os.makedirs('figures/tmp_ray_tracer', exist_ok=True)
            # depth and diffuse images
            for p in range(P):
                depth, diffuse = scene.get_depth_and_diffuse(cameras[p])
                # concatenate along the width dimension
                dm_e_im = np.concatenate((depth, diffuse), axis=1)  # (h, 2w)
                path = get_next_path(f'figures/tmp_ray_tracer/depth_energy.png')
                imageio.imwrite(path, dm_e_im)
            # range energy plots
            for p in range(P):
                e_r_values = energy_range_values[p]  # list[(n, 2)]
                for i in range(len(e_r_values)):  # generate a plot for each bounce
                    xy = e_r_values[i].cpu().numpy()
                    plt.scatter(xy[:, 0], xy[:, 1], s=1)
                    plt.xlabel("Range")
                    plt.ylabel("Energy")
                    plt.xlim(0, 6)
                    plt.ylim(-0.5, 1.5)
                    plt.title(f"Energy vs Range Plot for Bounce {i} for Pulse {p}")
                    plt.savefig(os.path.join("figures", "tmp_ray_tracer", f"energy_range_bounce_{i}_pulse_{p}.png"))
                    plt.close()

        # save the energy range values
        energy_range_values = [torch.cat(e_r_values, dim=0) for e_r_values in energy_range_values]  # join plots for different number of bounces. (list[r, 2])
        
        max_R = max([e_r_values.shape[0] for e_r_values in energy_range_values])
        scatter_ranges[t] = torch.zeros((P, max_R), device=device)  # different pulses will have different number of returned hits, pad with zero
        scatter_energies[t] = torch.zeros((P, max_R), device=device)
        # fill in the values
        for p in range(P):
            e_r_values = energy_range_values[p]
            scatter_ranges[t][p, :e_r_values.shape[0]] = e_r_values[:, 0]
            scatter_energies[t][p, :e_r_values.shape[0]] = e_r_values[:, 1]
    scatter_ranges = torch.stack(scatter_ranges, dim=0)  # (T, P, R)
    scatter_energies = torch.stack(scatter_energies, dim=0)  # (T, P, R)

    # apply complex value to the energy according to wavelength
    if wavelength is not None:
        scatter_energies = scatter_energies * torch.exp(1j * 2 * np.pi / wavelength * scatter_ranges)

    if debug_gif and num_bounces == 1:
        # convert range energy to images for visualization/debugging
        print('scatter_ranges.shape, scatter_energies.shape:', scatter_ranges.shape, scatter_energies.shape)

        # turn them into images and save them
        scatter_ranges_images = []
        scatter_energies_images = []
        for p in range(P):
            range_image = scatter_ranges[0, p, :].cpu().numpy()  # (65536,)
            energy_image = scatter_energies[0, p, :].cpu().numpy()  # (65536,)
            range_image = range_image.reshape((n_ray_height, n_ray_width))
            energy_image = energy_image.reshape((n_ray_height, n_ray_width))
            energy_image = np.abs(energy_image)  # because energy is complex valued
            # normalize to 0-255
            range_image = (range_image / np.max(range_image)) * 255
            energy_image = (energy_image / np.max(energy_image)) * 255
            range_image = range_image.astype(np.uint8)
            energy_image = energy_image.astype(np.uint8)
            scatter_ranges_images.append(range_image)
            scatter_energies_images.append(energy_image)
        scatter_ranges_images = np.concatenate(scatter_ranges_images, axis=1)  # (h, P*w)
        scatter_energies_images = np.concatenate(scatter_energies_images, axis=1)  # (h, P*w)
        os.makedirs('figures/tmp_ray_tracer', exist_ok=True)
        path = get_next_path(f'figures/tmp_ray_tracer/scatter_ranges.png')
        imageio.imwrite(path, scatter_ranges_images)
        path = get_next_path(f'figures/tmp_ray_tracer/scatter_energies.png')
        imageio.imwrite(path, scatter_energies_images)

    return scatter_ranges, scatter_energies
    #      (T, P, R)       (T, P, R)


def load_mesh_raytracing(  file_name,
                obj_rsa = (0.3,0.3,0.3),
                make_ground = True,
                ground_below = True,
                ground_rsa = (0.3,0.3,0.3),
                device = 'cuda',
        ):  
    '''
    Load a mesh from an obj file.
    Inputs:
        file_name: str - path to the obj file
        obj_rsa: tuple - reflectivity, specular, ambient for the object material
        make_ground: bool - whether to add a ground plane
        ground_rsa: tuple - roughness, specular, ambient for the ground material
        device: str - device to load the mesh onto
    Outputs:
        mesh: ray_tracer.core.scene.Scene - the loaded mesh with material properties
    '''
    scene = Scene(
        obj_filename=file_name,
        device=device,
        obj_rsa=obj_rsa,
    )  # will automatically build octree for this mesh

    # add a ground if desired to the mesh
    if make_ground:
        scene.add_ground(ground_below=ground_below, ground_rsa=ground_rsa)

    return scene
