'''
Debug harness for sidescansonar.render_side_scan_image.

Runs the geometry the paper figures are rendered from -- the defaults are sonar_paper_figures'
SONAR_PAPER_BASELINE, imported rather than copied so the debug output cannot drift away from the
panels it is meant to explain -- with the debug output turned on. Writes the per-ping debug movie
(debug_gif), the interpolated signals as one column per ping (debug_columns), and the
rgb-beside-sonar panel with its raw amplitudes. debug_gif and debug_columns are independent, so
the columns still can be checked without waiting on the movie -- pass debug_gif=False to skip it.
Any render_side_scan_image keyword passed here overrides the baseline.
'''
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use('Agg')

from sidescansonar import render_side_scan_image
from sonar_paper_figures import SONAR_PAPER_BASELINE


def debug_side_scan(
        suffix = 'debug',
        device = 'cuda',
        **overrides,
    ):

    kwargs = dict(SONAR_PAPER_BASELINE)
    kwargs.update(suffix=suffix, device=device, debug_gif=True, debug_columns=True)
    kwargs.update(overrides)

    if kwargs.get('pose_num') is None:  # lowest numbered pose, so reruns keep the same geometry
        dataset_dir = '/workspace/data/srncars/cars_train/'
        pose_dir = os.path.join(dataset_dir, kwargs['obj_id'], 'pose')
        kwargs['pose_num'] = sorted(os.listdir(pose_dir))[0].split('.')[0]

    os.makedirs('figures', exist_ok=True)
    return render_side_scan_image(**kwargs)


if __name__ == '__main__':
    # baseline's 000000 (40.6 deg elevation, shadow-throwing) vs a steeper 000043 (59.3 deg),
    # same obj_id -- lets the debug output be diffed across pose for a fixed target
    pose_nums = ('000000', '000043')

    for pose_num in pose_nums:
        debug_side_scan(
            suffix = 'debug_%s' % pose_num,
            pose_num = pose_num,
            num_pings  = 256,   # 4x SONAR_PAPER_BASELINE's 64
            spatial_fs = 64,    # 2x SONAR_PAPER_BASELINE's 32
            spatial_bw = 128,   # 2x SONAR_PAPER_BASELINE's 64
            region_radius = 2.0,  # 2x SONAR_PAPER_BASELINE's 1.0
            sensor_distance = 10.0,  # both poses sit at range ~1.3; push out to sonar-scale range
            elevation_fov_deg = 15.0,  # baseline's 45 deg * 1.3/10, so the fan covers the same ground swath at the new range
            # compression/db_floor/asinh_k_ratio come from SONAR_PAPER_BASELINE unless overridden here
            compression = 'asinh',  # 'linear' | 'db' | 'asinh'
            debug_gif = False,  # skip the slow per-ping movie; debug_columns still runs
        )
