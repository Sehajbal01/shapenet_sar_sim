import os
import sys
import torch
import numpy as np
from matplotlib import pyplot as plt

from render_images import sar_render_image
from utils import generate_pose_mat, plot_image, savefig

MODELS_DIR = '/workspace/data/cv_domes_cad_models_ojb_mtl_blend'
FIGURES_DIR = 'figures'

CENTER_AZIMUTH   = 210  # degrees
CENTER_ELEVATION = 32   # degrees
SENSOR_DISTANCE  = 1.3

RESOLUTION_MM = 100

AZ_SPREADS  = [30,90,180]
NUM_PULSES  = 30

# Generic args from default_kwargs at the bottom of render_images.py
GENERIC_KWARGS = dict(
    spatial_bw          = 3680/RESOLUTION_MM,
    spatial_fs          = 3680/RESOLUTION_MM,
    wavelength          = 0.5,
    use_sig_magnitude   = False,
    snr_db              = 50,
    image_width         = 128,
    image_height        = 128,
    image_plane_width   = 1,
    image_plane_height  = 1,
    grid_width          = 1.2,
    grid_height         = 1.2,
    n_ray_width         = 128,
    n_ray_height        = 128,
    region_radius       = 1.7,
    obj_raids           = (0.8, 0.0, 0.9, 0.1, 0.2),
    ground_raids        = (0.5, 0.0, 0.8, 0.2, 0.5),
    imaging_algorithm   = 'cbp',
    cbp_batch_size      = 4096,
    trajectory_type     = 'circular',
    trajectory_noise_var= 0,
    debug_gif           = True,
    mesh_scale          = 0.05,
    num_bounce          = 2,
    object_x_flip       = False,
    object_rotate_xyz   = (90.0, 0.0, 90.0),
)


def model_name(obj_path):
    filename = os.path.basename(obj_path)
    if '_' in filename:
        return filename.split('_')[0]
    return filename.split('0')[0]


def save_with_colorbar(sar_image, path):
    plot_image(sar_image, title=None, cmap='gray', db=True, relative_db=True)
    savefig(path)


def save_image_only(sar_image, path):
    if hasattr(sar_image, 'detach'):
        img = sar_image.detach().cpu().numpy()
    else:
        img = np.asarray(sar_image)
    img = np.squeeze(img)
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min + 1e-8)
    plt.imsave(path, img, cmap='gray')


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_obj_paths = sorted(
        os.path.join(MODELS_DIR, f)
        for f in os.listdir(MODELS_DIR)
        if f.endswith('.obj')
    )
    print(f'Found {len(all_obj_paths)} .obj models')

    pose = generate_pose_mat(CENTER_AZIMUTH, CENTER_ELEVATION, SENSOR_DISTANCE, device='cuda')
    pose = pose.reshape(1, 4, 4)

    for obj_path in all_obj_paths:
        name = model_name(obj_path)
        print(f'\nProcessing: {name}')

        for az_spread in AZ_SPREADS:
            print(f'  az_spread={az_spread}')

            sar_image = sar_render_image(
                obj_path,           # file_name (overridden by override_obj_path)
                NUM_PULSES,
                pose,
                az_spread,
                **GENERIC_KWARGS,
            )

            base = f'cvdomes_{name}_{az_spread}azspread'

            colorbar_path = os.path.join(FIGURES_DIR, f'{base}.png')
            save_with_colorbar(sar_image, colorbar_path)
            print(f'    saved: {colorbar_path}')

            nobar_path = os.path.join(FIGURES_DIR, f'{base}_nobar.png')
            save_image_only(sar_image, nobar_path)
            print(f'    saved: {nobar_path}')

        # only do 1 object
        break


if __name__ == '__main__':
    main()
