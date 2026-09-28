'''
Loads config.json, the one place this repo's dataset locations and paper figure baselines are set.

    SRN_CARS_DIR       srn_cars root, holding the cars_train / cars_val / cars_test splits
    SHAPENET_CARS_DIR  ShapeNet cars synset, <obj_id>/models/model_normalized.obj per object
    CONFIG             the whole file, with the three baselines under side_scan_sonar_baseline,
                       sar_baseline and range_angle_baseline

JSON has no tuples, so every list is turned back into one on load: the baselines' raids and
rotations were tuples when they lived in the figure scripts, and stay tuples for their callers.
'''
import json
import os


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')


def _tuplify(value):
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    if isinstance(value, dict):
        return {k: _tuplify(v) for k, v in value.items()}
    return value


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return _tuplify(json.load(f))


CONFIG = load_config()

SRN_CARS_DIR      = CONFIG['srn_cars_dir']
SHAPENET_CARS_DIR = CONFIG['shapenet_cars_dir']


def srn_split_dir(split='cars_train'):
    '''Directory of one srn_cars split, e.g. cars_train.'''
    return os.path.join(SRN_CARS_DIR, split)
