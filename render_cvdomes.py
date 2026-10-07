'''
Render a CVDomes car with each of the two .mp files convert_cvdomes.py writes next to the .obj --
one constant PEC material, and the per-triangle CVDomes materials -- and write the paper's 2x2
comparison as four untitled panels (the subfloats of fig:materials): first-bounce energy of the
center pulse on top, SAR image below, plus a titled stitch of all four in figures/.
'''
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt

from render_images import sar_render_image
from signal_simulation import load_mesh
from signal_visualization import signal_gif
from convert_cvdomes import MODELS_DIR, PEC_RAIDS, cvdomes_ground_raids
from imaging_algorithms import db_compress
from paper_figure_layout import stitch_panels
from utils import generate_pose_mat, plot_image, savefig

FIGURES_DIR = 'figures'
PAPER_FIG_DIR = 'latex/figs'
OBJ_NAME    = 'Camry_06212012.obj'

# .mp suffix -> panel title
MATERIALS = {'_constant': 'constant', '': 'per-triangle'}
SAR_DB_FLOOR = -40

CENTER_AZIMUTH   = 210  # degrees
CENTER_ELEVATION = 32   # degrees
SENSOR_DISTANCE  = 1.3

RESOLUTION_MM = 100

AZ_SPREAD  = 90
NUM_PULSES = 30

MESH_KWARGS = dict(
    make_ground         = True,
    scale               = 0.05,
    obj_raids           = PEC_RAIDS,   # unused, every render has a .mp
    ground_raids        = cvdomes_ground_raids(),
    x_flip              = False,
    rotate_xyz          = (90.0, 0.0, 90.0),
)

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
    imaging_algorithm   = 'cbp',
    cbp_batch_size      = 4096,
    trajectory_type     = 'circular',
    trajectory_noise_var= 0,
    num_bounce          = 2,
)


def save_with_colorbar(sar_image, path):
    plot_image(sar_image, title=None, cmap='gray', db=True, relative_db=True)
    savefig(path)


def save_image_only(sar_image, path):
    img = np.squeeze(sar_image.detach().cpu().numpy())
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.imsave(path, img, cmap='gray')


def render(preloaded_mesh, pose, base):
    sar_image, pulses = sar_render_image(None, NUM_PULSES, pose, AZ_SPREAD,
                                         preloaded_mesh=preloaded_mesh, return_pulses=True,
                                         **GENERIC_KWARGS)
    save_with_colorbar(sar_image, os.path.join(FIGURES_DIR, f'{base}.png'))
    save_image_only(sar_image, os.path.join(FIGURES_DIR, f'{base}_nobar.png'))

    signal_gif(pulses['signals'], pulses['sample_z'], pulses['debugging_maps'], pulses['ranges'],
               pulses['energies'], GENERIC_KWARGS['region_radius'], suffix=base, use_mp4_format=False)
    shutil.move(os.path.join(FIGURES_DIR, f'dm_em_sc_si_{base}.gif'), os.path.join(FIGURES_DIR, f'{base}.gif'))
    print(f'saved: {FIGURES_DIR}/{base}.png, {base}_nobar.png, {base}.gif')

    energy = pulses['debugging_maps'][(0, NUM_PULSES // 2)]['energy'].cpu().numpy()
    return energy, np.abs(np.squeeze(sar_image.detach().cpu().numpy()))


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    obj_path = os.path.join(MODELS_DIR, OBJ_NAME)
    name = OBJ_NAME.split('_')[0]

    pose = generate_pose_mat(CENTER_AZIMUTH, CENTER_ELEVATION, SENSOR_DISTANCE, device='cuda').reshape(1, 4, 4)

    energies, sar_images = [], []
    for suffix, title in MATERIALS.items():
        mp_path = os.path.splitext(obj_path)[0] + suffix + '.mp'
        preloaded = load_mesh(obj_path, device='cuda', material_file=mp_path, **MESH_KWARGS)
        energy, sar_image = render(preloaded, pose, f'cvdomes_{name}_{AZ_SPREAD}azspread_{title}')
        energies.append(energy)
        sar_images.append(sar_image)

    # both columns share one scale per row, so the materials compare by level as well as shape
    energy_peak = max(e.max() for e in energies)
    sar_peak = max(a.max() for a in sar_images)
    print('per-triangle re constant: SAR peak %.1f dB, SAR energy %.1f dB, first-bounce energy %.1f dB' % (
        20*np.log10(sar_images[1].max() / sar_images[0].max()),
        10*np.log10((sar_images[1]**2).sum() / (sar_images[0]**2).sum()),
        10*np.log10(energies[1].sum() / energies[0].sum())))
    panels = [e / energy_peak for e in energies] + [db_compress(a, sar_peak, SAR_DB_FLOOR) for a in sar_images]
    keys = [f'{t}_{row}' for row in ('energy', 'sar') for t in MATERIALS.values()]
    titles = [k.replace('_', ', ') for k in keys]
    style = dict(vmin=[0, 0, SAR_DB_FLOOR, SAR_DB_FLOOR], vmax=[1, 1, 0, 0],
                 cbar_label=['energy / shared peak'] * 2 + ['dB re shared peak'] * 2,
                 cbar_tick_fmt=['%.2g', '%.2g', '%.0f dB', '%.0f dB'])
    stitch_panels(panels, titles, os.path.join(FIGURES_DIR, f'cvdomes_{name}_materials.png'), ncols=2, **style)

    # one file per subfloat, the titles go in the caption
    for i, key in enumerate(keys):
        stitch_panels([panels[i]], [''], os.path.join(PAPER_FIG_DIR, f'cvdomes_materials_{key}.png'),
                      **{k: v[i] for k, v in style.items()})


if __name__ == '__main__':
    main()
