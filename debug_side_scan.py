'''
Debug harness for sidescansonar.render_side_scan_image.

Runs the geometry the paper figures are rendered from -- the defaults are sonar_paper_figures'
SONAR_PAPER_BASELINE, imported rather than copied so the debug output cannot drift away from the
panels it is meant to explain -- with the debug output turned on. Writes the per-ping debug movie,
the interpolated signals as one column per ping, and the rgb-beside-sonar panel with its raw
amplitudes. Any render_side_scan_image keyword passed here overrides the baseline.
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
    kwargs.update(overrides)
    kwargs.update(suffix=suffix, device=device, debug_gif=True)

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
            # db/db_floor come from SONAR_PAPER_BASELINE unless overridden here
        )
