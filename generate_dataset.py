'''
Render the side scan sonar, CBP SAR and strip map SAR views of every srn_cars object, one image
per rgb pose, into the split's own object directories:

    /workspace/data/srncars/cars_test/<obj_id>/side_scan_sonar_135azspread/<pose_num>.png
    /workspace/data/srncars/cars_test/<obj_id>/cbp_sar_135azspread/<pose_num>.png
    /workspace/data/srncars/cars_test/<obj_id>/strip_map_sar_135azspread/<pose_num>.png

beside the rgb/, pose/, sonar/ and raysar/ directories already there. 128x128 8-bit gray PNGs
named after the pose, so every rgb frame has one image per modality, as check_sonar_exists.py
and check_raysar_exists.py assert of the existing modalities.

The physics comes from the paper figure baselines rather than being restated here, so the
dataset tracks whatever those figures show: sonar_paper_figures.SONAR_PAPER_BASELINE for the
side scan and paper_figures.PAPER_BASELINE for both SAR images, with the trajectory forced
linear at AZIMUTH_SPREAD_DEG -- which is what the directory suffix records.

Each object is loaded and octree-built once and then imaged from all of its poses, and each
pose is ray traced twice, not three times: the CBP and strip map images are two imaging
algorithms run on one SAR ray trace, which is the bulk of the cost. Only the side scan needs
its own trace, since its sensor is a point flying a track rather than a distant plane.

Run one process per GPU, each on its own contiguous slice of the object list:

    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
            generate_dataset.py -num_chunks 4 -chunk_id $i &
    done; wait

Already-rendered poses are skipped, so an interrupted chunk is resumed by rerunning it.

-test_run renders exactly the same images but writes them to ./figures as
test_run_<modality>_<obj_id>_az<spread>_<pose>.png and creates nothing under the dataset, so a
chunk can be checked end to end before the real run. See test.sh.
'''
import argparse
import contextlib
import io
import os
import time
import zlib

# MKL (libiomp5) and PyTorch (libomp) each link their own OpenMP runtime; the second to
# initialize aborts with "OMP: Error #15". Allow the duplicate, as paper_figures.py does.
# Must be set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import PIL.Image
import torch

from paper_figures import PAPER_BASELINE
from ray_tracer_v2 import build_octree
from render_images import sar_render_image
from imaging_algorithms import to_asinh, to_db_uint8
from sidescansonar import side_scan_sonar_image
from signal_simulation import load_mesh
from sonar_paper_figures import SONAR_PAPER_BASELINE
from utils import extract_pose_info, generate_pose_mat


MODELS_DIR = '/workspace/data/srncars/02958343'
SPLITS_DIR = '/workspace/data/srncars'

# the one trajectory this dataset is rendered on. Linear and strictly below 180 deg, which is
# where generate_trajectory's linear track runs off to infinity
AZIMUTH_SPREAD_DEG = 135.0
TRAJECTORY_TYPE    = 'linear'

# the three modalities, in the order they are rendered and saved. Each names a directory under
# the object, carrying the suffix so a rerun at another spread lands beside this one rather than
# overwriting it
MODALITIES = ('side_scan_sonar', 'cbp_sar', 'strip_map_sar')
DIR_SUFFIX = '_%dazspread' % AZIMUTH_SPREAD_DEG

# where -test_run writes instead, alongside the paper figures
FIGURES_DIR = 'figures'


# how a raw amplitude becomes an 8-bit pixel. The amplitudes span orders of magnitude, so a
# linear stretch is a few specular returns over a near-black field (measured: mean 20/255 for
# the SAR modalities, 7/255 for the side scan). 'asinh' stays linear near zero and goes
# logarithmic past ASINH_K_RATIO * the image's own 99.9th percentile, which is what the paper
# figures display with. 'linear' is the plain min-max stretch, 'db' is dB below the peak.
# Referenced per image, as every asinh call site in this codebase is
COMPRESSION   = 'asinh'   # 'linear' | 'db' | 'asinh'
ASINH_K_RATIO = 0.1
DB_FLOOR      = -60.0

# srn_cars poses spiral from 0 to 90 deg of elevation, and both ends are degenerate looks rather
# than hard ones: at 90 the ground projection of the slant range vanishes, so projected_CBP
# divides by zero and the linear track collapses to a point, and at 0 the sensor sits in the
# seafloor and side_scan_sonar_image's track direction, cross(line of sight, +z), is undefined.
# A pose outside the band is re-aimed to the nearest edge of it, keeping its azimuth and range,
# so every rgb frame still gets an image. 17 of the 251 test poses are re-aimed: 3 at the bottom
# and 14 at the top. Raise MIN_ELEVATION_DEG to trade more re-aimed poses for fewer grazing ones.
MIN_ELEVATION_DEG = 1.0
MAX_ELEVATION_DEG = 85.0

# the mesh settings both baselines carry and must agree on, since one loaded mesh serves all
# three modalities. Asserted rather than assumed, so a future edit that moves one baseline's
# geometry without the other's is caught here instead of silently rendering the SAR on the
# sonar's car. make_ground and level_with_ground are not on this list because only
# SONAR_PAPER_BASELINE states them -- the SAR path leaves both at load_mesh's default, which is
# the True that the sonar baseline asks for, so the two still agree
SHARED_MESH_KEYS = ('object_x_flip', 'object_rotate_xyz', 'obj_raids', 'ground_raids')

# baseline keys forwarded to each renderer. Spelled out rather than filtered by signature so a
# renamed baseline key raises a KeyError here instead of quietly falling back to a default.
# obj_raids/ground_raids are inert while preloaded_mesh is set, since they only ever reach
# load_mesh -- they are forwarded anyway so that dropping preloaded_mesh keeps the same mesh
# rather than silently falling back to sar_render_image's own defaults, which differ from the
# baseline's (ground d of 0.9 against the baseline's 5, which moves the image substantially)
SAR_KEYS = ('spatial_bw', 'spatial_fs', 'waveform', 'snr_db', 'wavelength', 'use_sig_magnitude',
            'cbp_batch_size', 'trajectory_noise_var', 'num_bounce',
            'image_width', 'image_height', 'image_plane_width', 'image_plane_height',
            'grid_width', 'grid_height', 'n_ray_width', 'n_ray_height', 'region_radius',
            'obj_raids', 'ground_raids', 'object_x_flip', 'object_rotate_xyz')
SIDE_SCAN_KEYS = ('image_width', 'image_height', 'image_plane_width', 'image_plane_height',
                  'wavelength', 'num_bounce', 'spherical_spread', 'water_absorption',
                  'tvg_exponent', 'spatial_bw', 'spatial_fs', 'waveform', 'use_sig_magnitude')


def _quiet(fn, *args, **kwargs):
    '''Run fn with stdout swallowed -- every renderer prints a timing line per call.'''
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def output_path(object_dir, obj_id, pose_num, modality, test_run):
    '''
    Where one rendered image goes. The one place the two output layouts are decided, so the
    skip-what-is-done check and the save below can never disagree about it.

    inputs:
        object_dir (str): the object's directory in the split
        obj_id (str), pose_num (str), modality (str): what is being rendered
        test_run (bool): True writes a flat, self-describing name under FIGURES_DIR instead of
            into the dataset, so a trial run can be eyeballed without touching the real dataset
    outputs:
        path (str): the png to write
    '''
    if test_run:
        return os.path.join(FIGURES_DIR, 'test_run_%s_%s_az%d_%s.png'
                            % (modality, obj_id, AZIMUTH_SPREAD_DEG, pose_num))
    return os.path.join(object_dir, modality + DIR_SUFFIX, '%s.png' % pose_num)


def save_gray_png(amplitude, path):
    '''
    Write one raw amplitude image as a 128x128 8-bit gray PNG, compressed by COMPRESSION and
    referenced to this one image, the same three-way branch sidescansonar.render_side_scan_image
    saves its composite with. Per image, so the level between views is not kept.

    inputs:
        amplitude (H,W): raw amplitude, numpy or torch
        path (str): png to write
    '''
    amplitude = np.asarray(amplitude, dtype=np.float32)
    amplitude = np.nan_to_num(amplitude, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(amplitude.max())

    if COMPRESSION == 'db' and peak > 0.0:
        image = to_db_uint8(amplitude, peak, DB_FLOOR)  # (H,W)
    elif COMPRESSION == 'asinh':
        # ref is this image's own 99.9th percentile, as panel_display references its asinh
        # panels. It can round to 0 on a nearly all-dark image even when the peak is positive,
        # which would make k 0 and arcsinh divide by it
        ref = float(np.percentile(amplitude, 99.9))
        if ref > 0.0:
            image = to_asinh(amplitude, ASINH_K_RATIO * ref, ref)  # (H,W)
        else:
            image = np.zeros(amplitude.shape, dtype=np.uint8)
    else:
        low = float(amplitude.min())
        span = max(peak - low, 1e-12)  # an all-dark image would divide by 0
        image = ((amplitude - low) / span * 255.0).clip(0, 255).astype(np.uint8)  # (H,W)

    PIL.Image.fromarray(image, mode='L').save(path)


def aimed_pose(pose_path, device):
    '''
    Read one srn_cars pose and re-aim it into the elevation band both sensor geometries can fly.

    inputs:
        pose_path (str): the pose txt of one rgb frame
        device (str): device to build the pose on
    outputs:
        poses (1,4,4): the pose to render, clamped in elevation and otherwise unchanged
        elevation_deg (float): the elevation actually rendered
        clamped (bool): whether the pose file's own elevation was outside the band
    '''
    pose = np.loadtxt(pose_path).reshape(1, 4, 4).astype(np.float32)
    poses = torch.tensor(pose, device=device)  # (1,4,4)

    pose_info = extract_pose_info(poses)
    distance, elevation_deg, azimuth_deg = (pose_info[4].item(), pose_info[5].item(),
                                            pose_info[6].item())

    aimed_elevation_deg = min(max(elevation_deg, MIN_ELEVATION_DEG), MAX_ELEVATION_DEG)
    clamped = aimed_elevation_deg != elevation_deg
    if clamped:
        poses = generate_pose_mat(azimuth_deg, aimed_elevation_deg, distance,
                                  device=device).reshape(1, 4, 4)

    return poses, aimed_elevation_deg, clamped


def render_object(obj_id, split, device='cuda', overwrite=False, max_poses=None, verbose=False,
                  test_run=False):
    '''
    Render every pose of one object, in all three modalities, off a single mesh load.

    inputs:
        obj_id (str): srn_cars object id, a directory under both MODELS_DIR and the split
        split (str): 'cars_train' | 'cars_val' | 'cars_test'
        device (str): device to render on
        overwrite (bool): re-render poses whose three PNGs are already on disk
        max_poses (int): stop after this many poses, for a smoke test; None renders them all
        verbose (bool): let the renderers' own per-call prints through
        test_run (bool): render exactly as usual but write the PNGs into FIGURES_DIR under
            self-describing names, leaving the dataset untouched
    outputs:
        n_rendered (int): poses rendered, not counting the ones skipped as already done
        n_clamped (int): of those, how many had their elevation re-aimed into the band
    '''
    object_dir = os.path.join(SPLITS_DIR, split, obj_id)
    pose_dir   = os.path.join(object_dir, 'pose')
    mesh_path  = os.path.join(MODELS_DIR, obj_id, 'models', 'model_normalized.obj')

    # a test run creates nothing under the object, so it cannot leave half-filled modality
    # directories behind in the dataset
    if test_run:
        os.makedirs(FIGURES_DIR, exist_ok=True)
    else:
        for modality in MODALITIES:
            os.makedirs(os.path.join(object_dir, modality + DIR_SUFFIX), exist_ok=True)

    pose_nums = sorted(os.path.splitext(f)[0] for f in os.listdir(pose_dir) if f.endswith('.txt'))
    if max_poses is not None:
        pose_nums = pose_nums[:max_poses]

    # figure out what is left to do before touching the GPU, so a finished object costs no mesh load
    todo = pose_nums if overwrite else [
        p for p in pose_nums
        if not all(os.path.exists(output_path(object_dir, obj_id, p, m, test_run))
                   for m in MODALITIES)
    ]
    if not todo:
        return 0, 0

    # the one mesh load and the one octree build this object pays for, shared by every pose and
    # every modality. Both baselines' mesh settings must match for that to be legitimate
    for key in SHARED_MESH_KEYS:
        assert PAPER_BASELINE[key] == SONAR_PAPER_BASELINE[key], \
            'PAPER_BASELINE[%r] != SONAR_PAPER_BASELINE[%r]; the two modalities no longer ' \
            'share one mesh, so they can no longer share one load' % (key, key)

    run = (lambda fn, *a, **k: fn(*a, **k)) if verbose else _quiet
    mesh_bundle = run(load_mesh, mesh_path,
                      device            = device,
                      make_ground       = SONAR_PAPER_BASELINE['make_ground'],
                      level_with_ground = SONAR_PAPER_BASELINE['level_with_ground'],
                      obj_raids         = PAPER_BASELINE['obj_raids'],
                      ground_raids      = PAPER_BASELINE['ground_raids'],
                      x_flip            = PAPER_BASELINE['object_x_flip'],
                      rotate_xyz        = PAPER_BASELINE['object_rotate_xyz'],
                      )
    octree = build_octree(mesh_bundle[0])

    sar_kwargs        = {k: PAPER_BASELINE[k] for k in SAR_KEYS}
    side_scan_kwargs  = {k: SONAR_PAPER_BASELINE[k] for k in SIDE_SCAN_KEYS}

    n_clamped = 0
    for pose_num in todo:
        poses, _, clamped = aimed_pose(os.path.join(pose_dir, '%s.txt' % pose_num), device)
        n_clamped += clamped

        # seed per pose, so a rerun of one pose reproduces its image instead of redrawing the
        # receiver noise. np.random is the one that matters -- apply_snr draws the noise from
        # numpy, not torch -- but seed both, since that is what multi_param_experiment does and
        # which generator a renderer reaches for is not something a caller should have to track.
        # crc32 and not hash(), whose string seed changes every interpreter
        seed = zlib.crc32(('%s/%s' % (obj_id, pose_num)).encode())
        np.random.seed(seed)
        torch.manual_seed(seed)

        # both SAR images off one ray trace: the trace and the signal interpolation are the same
        # for either algorithm, and only the imaging step differs
        sar_images = run(sar_render_image, mesh_path,
                         PAPER_BASELINE['num_pulse'],
                         poses,
                         AZIMUTH_SPREAD_DEG,
                         imaging_algorithm = ('cbp', 'stripmap'),
                         trajectory_type   = TRAJECTORY_TYPE,
                         preloaded_mesh    = mesh_bundle,
                         octree            = octree,
                         **sar_kwargs)  # {'cbp': (1,H,W), 'stripmap': (1,H,W)}

        # the side scan reads only the sensor *direction* off the pose: it flies its own straight
        # track at SONAR_PAPER_BASELINE's sensor_distance, not the pose file's own 1.3
        sensor_position = extract_pose_info(poses)[0].reshape(3)  # (3,)
        sensor_position = torch.nn.functional.normalize(sensor_position, dim=-1) \
                          * SONAR_PAPER_BASELINE['sensor_distance']
        side_scan_image = run(side_scan_sonar_image,
                              sensor_position,
                              SONAR_PAPER_BASELINE['track_length'],
                              SONAR_PAPER_BASELINE['num_pings'],
                              SONAR_PAPER_BASELINE['elevation_fov_deg'],
                              SONAR_PAPER_BASELINE['azimuth_beam_width_deg'],
                              *mesh_bundle,
                              SONAR_PAPER_BASELINE['num_ray_width'],
                              SONAR_PAPER_BASELINE['num_ray_height'],
                              SONAR_PAPER_BASELINE['region_radius'],
                              octree = octree,
                              **side_scan_kwargs)[0]  # (T,H,W), one track

        rendered = {'side_scan_sonar': side_scan_image[0],
                    'cbp_sar':         sar_images['cbp'][0],
                    'strip_map_sar':   sar_images['stripmap'][0]}
        for modality in MODALITIES:
            save_gray_png(rendered[modality].detach().cpu().numpy(),
                          output_path(object_dir, obj_id, pose_num, modality, test_run))

    return len(todo), n_clamped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-split', default='cars_test', choices=['cars_train', 'cars_val', 'cars_test'],
                        help='srn_cars split to render (default: cars_test)')
    parser.add_argument('-num_chunks', type=int, default=1,
                        help='how many processes the object list is split across (default: 1)')
    parser.add_argument('-chunk_id', type=int, default=0,
                        help='which chunk this process renders, 0 .. num_chunks-1 (default: 0)')
    parser.add_argument('-overwrite', action='store_true',
                        help='re-render poses whose PNGs already exist instead of skipping them')
    parser.add_argument('-max_objects', type=int, default=None,
                        help='stop after this many objects of the chunk, for a smoke test')
    parser.add_argument('-max_poses', type=int, default=None,
                        help='render only the first this-many poses of each object, for a smoke test')
    parser.add_argument('-verbose', action='store_true',
                        help='let the renderers print their own per-call timing lines')
    parser.add_argument('-test_run', action='store_true',
                        help='render everything the same way but write the PNGs into %s/ as '
                             'test_run_<modality>_<obj_id>_az<spread>_<pose>.png, leaving the '
                             'dataset directories untouched' % FIGURES_DIR)
    args = parser.parse_args()

    assert 0 <= args.chunk_id < args.num_chunks, \
        'chunk_id must be in 0 .. num_chunks-1, got %d of %d' % (args.chunk_id, args.num_chunks)

    split_dir = os.path.join(SPLITS_DIR, args.split)

    # sorted, so os.listdir's arbitrary order cannot decide who renders what. srn_cars ids are
    # hashes, so sorted order is already unrelated to mesh size -- measured lag-1 autocorrelation
    # of size along this list is -0.02 -- and chunks of it come out as evenly balanced as a
    # shuffle of it would. What is left is sampling variance, which only size-aware packing fixes
    obj_ids = sorted(os.listdir(split_dir))

    # chunk membership depends on the whole list, so a machine whose split listing differs by even
    # one entry cuts different chunks. Print a fingerprint of the list: two machines that print
    # the same one agree on the partition, and two that do not would silently render the wrong sets
    fingerprint = zlib.crc32('\n'.join(obj_ids).encode()) & 0xffffffff

    # contiguous slices, so chunk 0 takes the front of the list. array_split spreads the
    # remainder over the first chunks rather than piling it onto the last one
    chunk = [str(o) for o in np.array_split(np.array(obj_ids), args.num_chunks)[args.chunk_id]]
    if args.max_objects is not None:
        chunk = chunk[:args.max_objects]
    if not chunk:
        print('%s: chunk %d/%d is empty (%d objects over %d chunks), nothing to do'
              % (args.split, args.chunk_id, args.num_chunks, len(obj_ids), args.num_chunks))
        return

    print('%s: chunk %d/%d -- %d of %d objects, %s .. %s'
          % (args.split, args.chunk_id, args.num_chunks, len(chunk), len(obj_ids),
             chunk[0], chunk[-1]))
    print('partition: object list crc32 %08x -- must match on every machine sharing this run'
          % fingerprint)
    if args.test_run:
        print('TEST RUN: writing %s/test_run_<modality>_<obj_id>_az%d_<pose>.png, '
              'the dataset is not touched' % (FIGURES_DIR, AZIMUTH_SPREAD_DEG))
    else:
        print('writing %s per object'
              % ', '.join('%s%s/' % (m, DIR_SUFFIX) for m in MODALITIES))
    print('display: %s compression%s' % (
        COMPRESSION,
        ', k/ref %g' % ASINH_K_RATIO if COMPRESSION == 'asinh' else
        ', floor %g dB' % DB_FLOOR if COMPRESSION == 'db' else ''))

    t_start = time.time()
    total_rendered, total_clamped = 0, 0
    failed = []
    for i, obj_id in enumerate(chunk):
        t_object = time.time()
        # one bad object -- a malformed mesh, an OOM -- should cost that object and not the
        # rest of a chunk that runs for hours. Its poses stay un-rendered, so a rerun retries it
        try:
            n_rendered, n_clamped = render_object(obj_id, args.split,
                                                  overwrite = args.overwrite,
                                                  max_poses = args.max_poses,
                                                  verbose   = args.verbose,
                                                  test_run  = args.test_run)
        except Exception as exception:
            failed.append(obj_id)
            print('[%d/%d] %s FAILED: %s: %s' % (i + 1, len(chunk), obj_id,
                                                 type(exception).__name__, exception), flush=True)
            torch.cuda.empty_cache()
            continue
        total_rendered += n_rendered
        total_clamped  += n_clamped

        elapsed = time.time() - t_start
        eta_hours = (elapsed / (i + 1)) * (len(chunk) - i - 1) / 3600
        print('[%d/%d] %s: %d poses in %.1f s (%d re-aimed) -- %.1f h elapsed, %.1f h left'
              % (i + 1, len(chunk), obj_id, n_rendered, time.time() - t_object, n_clamped,
                 elapsed / 3600, eta_hours), flush=True)

    print('chunk %d/%d done: %d poses over %d objects in %.1f h, %d re-aimed in elevation'
          % (args.chunk_id, args.num_chunks, total_rendered, len(chunk),
             (time.time() - t_start) / 3600, total_clamped))
    if failed:
        print('%d objects failed and were left un-rendered, rerun this chunk to retry them:\n  %s'
              % (len(failed), '\n  '.join(failed)))


if __name__ == '__main__':
    main()
