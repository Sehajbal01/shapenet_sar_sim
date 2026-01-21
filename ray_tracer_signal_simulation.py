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
                        azimuth_spread=15, n_pulses=30,
                        pulse_azimuths=None, pulse_elevations=None, pulse_distances=None,
                        wavelength=None,
                        pulse_type="cw", lfm_bandwidth=None, lfm_pulse_duration=None, speed_of_light=3e8,
                        debug_gif=False,
                        grid_width=1, grid_height=1,
                        n_ray_width=1, n_ray_height=1,

                        num_bounces=2,
                    ):
    '''
    returns the energy and range for a bunch of rays for each pulse

    inputs:
        target_poses (T,4,4): the rgb pose for which we want to get a sar image from
        object_filename (str): path to the .obj file
        azimuth_spread (float): the range of azimuth angles for sar rendering
        n_pulses (int): the number of pulses for sar rendering
        alpha_1 (float): scaling factor for the energy return
        alpha_2 (float): offset for the energy return
        use_ground (bool): whether to use the ground plane for rendering
        debug_gif (bool): whether to save a gif of the depth and energy images
        num_bounces (int): number of ray bounces to simulate

    outputs:
        range (T,P,R): the range of all the rays
        energy (T,P,R): the simulated energy of all the rays

    '''
    scene = mesh

    device = target_poses.device
    T = target_poses.shape[0]  # no. of camera views
    P = n_pulses               # no. of pulses per view

    # Pull out camera positions info 
    _, _, _, _, cam_distance, cam_elevation, cam_azimuth = extract_pose_info(target_poses)
    #           (T,)          (T,)           (T,)

    def _coerce_pulse_values(values, label):
        # Normalized per-pulse inputs to a (T,P) tensor to make life easier
        # Ques: should we accept lists of (az, el, dist) tuples and split them here
        # kept this strict on shape so callers immediately see if the trajectory vector is off
        if values is None:
            return None
        values = torch.as_tensor(values, device=device)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError(f"{label} must be 1D or 2D, got shape {values.shape}.")
        if values.shape[0] == 1 and T > 1:
            values = values.repeat(T, 1)
        if values.shape[0] != T:
            raise ValueError(f"{label} must have shape (T, P) or (P,), got {values.shape}.")
        return values

    pulse_azimuths = _coerce_pulse_values(pulse_azimuths, "pulse_azimuths")
    pulse_elevations = _coerce_pulse_values(pulse_elevations, "pulse_elevations")
    pulse_distances = _coerce_pulse_values(pulse_distances, "pulse_distances")

    if pulse_azimuths is None and pulse_elevations is None and pulse_distances is None:
        # Spread the pulses across a small range of azimuth angles
        # preserve existing SAR spread 
        # this keeps old behavior so downstream configs don't unregulated
        azimuth_offsets = torch.linspace(-azimuth_spread / 2, azimuth_spread / 2, P, device=device) # (P,)
        azimuth = cam_azimuth.reshape(T, 1) + azimuth_offsets.reshape(1, P) # (T,P)
        elevation = torch.tile(cam_elevation.reshape(T, 1), (1, P))  # (T, P)
        distance  = torch.tile(cam_distance.reshape(T, 1), (1, P))  # (T, P)
    else:
        # Manual trajectory mode: use per-pulse geometry given by the caller (us)
        # Ques: do we want to warn if only one of az/elev/dist is provided?-unlikely
 
        pulse_shapes = [
            values.shape[1]
            for values in (pulse_azimuths, pulse_elevations, pulse_distances)
            if values is not None
        ]
        if len(set(pulse_shapes)) > 1:
            raise ValueError(f"pulse_* inputs must share the same P dimension, got {pulse_shapes}.")
        P = pulse_shapes[0]
        if pulse_azimuths is None:
            pulse_azimuths = cam_azimuth.reshape(T, 1).repeat(1, P)
        if pulse_elevations is None:
            pulse_elevations = cam_elevation.reshape(T, 1).repeat(1, P)
        if pulse_distances is None:
            pulse_distances = cam_distance.reshape(T, 1).repeat(1, P)
        azimuth = pulse_azimuths
        elevation = pulse_elevations
        distance = pulse_distances


    # loop over each pulse and compute the depth map and surface normal
    scatter_ranges = []
    scatter_energies = []
    for t in range(T):  # for each camera
        scatter_ranges.append([])
        scatter_energies.append([])

        # construct P number of cameras due to azimuth spread
        cameras = []
        for p in range(P):  # for each pulse
            # building a camera per-pulse so manual trajectories really do place the sensor where you specify.
            elevation_rad = elevation[t, p] / 180 * torch.pi  # in radians now
            azimuth_rad = (90 + azimuth[t, p]) / 180 * torch.pi  # in radians now
            position_vector = torch.tensor([
                torch.cos(elevation_rad) * torch.sin(azimuth_rad),
                torch.sin(elevation_rad),
                torch.cos(elevation_rad) * torch.cos(azimuth_rad)
            ], device=device)
            position_vector = position_vector / torch.norm(position_vector) * distance[t, p]
            direction_vector = torch.tensor([0, 0, 0], device=device) - position_vector
            direction_vector = direction_vector / torch.norm(direction_vector)
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
    # NOTE: matches the old CW phase behavior 
    if wavelength is not None:
        scatter_energies = scatter_energies * torch.exp(1j * 2 * np.pi / wavelength * scatter_ranges)
    if pulse_type == "lfm":
        # LFM/chirp phase term for strip-map style pulses.
        # Q: should we also window/gate by pulse duration to avoid phase outside the chirp?
        if lfm_bandwidth is None or lfm_pulse_duration is None:
            raise ValueError("lfm_bandwidth and lfm_pulse_duration must be set when pulse_type='lfm'.")
        chirp_rate = lfm_bandwidth / lfm_pulse_duration
        tau = 2 * scatter_ranges / speed_of_light
        scatter_energies = scatter_energies * torch.exp(1j * np.pi * chirp_rate * tau ** 2)
    elif pulse_type != "cw":
        raise ValueError(f"Unsupported pulse_type '{pulse_type}'. Expected 'cw' or 'lfm'.")

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

    return scatter_ranges, scatter_energies, azimuth, elevation, distance, cam_azimuth, cam_distance
    #      (T, P, R)       (T, P, R)         (T, P)   (T, P)     (T, P)    (T,)         (T,)


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
