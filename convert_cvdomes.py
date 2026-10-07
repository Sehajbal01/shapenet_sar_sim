'''
Write two .mp material files next to each CVDomes .obj, one "r a i d s" line per triangle in the
order load_mesh reads the faces: <name>.mp with the CVDomes material of each triangle, and
<name>_constant.mp with PEC on every triangle, for comparison.

The .obj files carry Blender material names (usemtl). Each maps to a CVDomes material label
(Dungan et al., CVDomes report, Table 3), and each label to raids:
    PEC (0)                  car body and rims, all energy reflected or scattered
    glass, plastic (300/500) 6.35 mm slab, eps = 5.9 - j0.15; CVDomes drops the transmitted ray, so
                             only the band-averaged slab reflection |R|^2 ~ 0.15 comes back
    rubber (200)             absorber
A dielectric keeps its reference split of reflection to scattering, scaled by |R|^2. The asphalt
ground (501, eps = 5.9 - j0.1 half-space, |R|^2 ~ 0.17) is not in the .obj; see cvdomes_ground_raids.
'''
import os
import sys
import numpy as np

MODELS_DIR = '/workspace/data/cv_domes_cad_models_ojb_mtl_blend'

SPEED_OF_LIGHT = 299792458
GLASS_EPS       = 5.9 - 0.15j
GLASS_THICKNESS = 0.25 * 0.0254   # m
ASPHALT_EPS     = 5.9 - 0.1j
CENTER_FREQ     = 9.6e9
BANDWIDTH       = 5.351e9

# (r, a, i, d, s) of a perfect conductor, and of a perfectly reflecting rough ground
PEC_RAIDS    = (0.8, 0.0, 0.9, 0.1, 0.2)
GROUND_RAIDS = (0.5, 0.0, 0.8, 0.2, 0.5)

# blender material name (lowercase) -> CVDomes material
MTL_TO_CVDOMES = {
    'mm_default_random_col': 'pec',
    'rims':                  'pec',
    'glass':                 'glass',
    'plastic':               'plastic',
    'rubber':                'rubber',
}


def slab_reflection(eps, thickness, center_freq, bandwidth, n_freq=512):
    '''band-averaged normal-incidence power reflection of a dielectric slab in air'''
    f = np.linspace(center_freq - bandwidth/2, center_freq + bandwidth/2, n_freq)
    n = np.sqrt(eps)
    r = (1 - n) / (1 + n)
    phase = np.exp(-2j * 2*np.pi * f * n * thickness / SPEED_OF_LIGHT)
    R = r * (1 - phase) / (1 - r**2 * phase)
    return float(np.mean(np.abs(R)**2))


def half_space_reflection(eps):
    '''normal-incidence power reflection of a dielectric half-space'''
    n = np.sqrt(eps)
    return float(np.abs((1 - n) / (1 + n))**2)


def dielectric_raids(power_reflection, reference=PEC_RAIDS):
    r, _, i, d, s = reference
    return (r*power_reflection, 1 - power_reflection, i, d, s*power_reflection)


def cvdomes_ground_raids():
    return dielectric_raids(half_space_reflection(ASPHALT_EPS), GROUND_RAIDS)


def cvdomes_raids():
    glass = dielectric_raids(slab_reflection(GLASS_EPS, GLASS_THICKNESS, CENTER_FREQ, BANDWIDTH))
    return {
        'pec':     PEC_RAIDS,
        'glass':   glass,
        'plastic': glass,   # CVDomes gives plastic the glass properties
        'rubber':  (0.0, 1.0, 0.9, 0.1, 0.0),
    }


def write_mp(mp_path, lines, source):
    with open(mp_path, 'w') as f:
        f.write(f'# r a i d s per triangle, from {source} via convert_cvdomes.py\n')
        f.write('\n'.join(lines) + '\n')


def convert(obj_path, raids_by_material):
    '''write obj_path's .mp and _constant.mp, return the triangle count of each CVDomes material'''
    lines, counts = [], {}
    material = None
    with open(obj_path) as f:
        for line in f:
            if line.startswith('usemtl '):
                name = line.split(None, 1)[1].strip().lower()
                if name not in MTL_TO_CVDOMES:
                    raise ValueError(f'{obj_path}: unknown material {name}')
                material = MTL_TO_CVDOMES[name]
            elif line.startswith('f '):
                if material is None:
                    raise ValueError(f'{obj_path}: face before any usemtl')
                # pytorch3d fan-triangulates an n-gon into n-2 triangles
                n_tri = len(line.split()) - 3
                lines += [' '.join(f'{v:.6f}' for v in raids_by_material[material])] * n_tri
                counts[material] = counts.get(material, 0) + n_tri

    stem, source = os.path.splitext(obj_path)[0], os.path.basename(obj_path)
    write_mp(stem + '.mp', lines, source)
    write_mp(stem + '_constant.mp', [' '.join(f'{v:.6f}' for v in PEC_RAIDS)] * len(lines), source)
    print(f'{stem}.mp, _constant.mp: {len(lines)} triangles {counts}')
    return counts


def main():
    obj_paths = sys.argv[1:] or sorted(
        os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.endswith('.obj'))
    raids_by_material = cvdomes_raids()
    for material, raids in raids_by_material.items():
        print(f'{material:8s} r a i d s = ' + ' '.join(f'{v:.3f}' for v in raids))
    print('ground   r a i d s = ' + ' '.join(f'{v:.3f}' for v in cvdomes_ground_raids()))
    for obj_path in obj_paths:
        convert(obj_path, raids_by_material)


if __name__ == '__main__':
    main()
