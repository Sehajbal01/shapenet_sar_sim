'''
Amplitude -> display compression, shared by every place that turns a raw side scan amplitude
array into something imshow-able: render_side_scan_image's rgb-beside-sonar composite and
signal_column_image's interpolated-signal still (both in sidescansonar's render path) plus
sonar_paper_figures._panel's stitched sweep panels. Kept in its own module, rather than living on
whichever of those imports the others, so all three can import it without a cycle.
'''
import numpy as np


def asinh_compress(amplitude, k, ref):
    '''
    Core of asinh display compression: linear near zero, ~logarithmic once amplitude passes k,
    so seafloor texture stays visible without dB's hard floor clipping it to black.

    inputs:
        amplitude (H,W): raw amplitude
        k (float): softening scale -- arcsinh(x/k) is ~linear for x << k, ~logarithmic for x >> k
        ref (float): amplitude that maps to output 1.0
    outputs:
        panel (H,W): compressed amplitude in [0,1]
    '''
    amplitude = np.asarray(amplitude, dtype=np.float32)
    compressed = np.arcsinh(amplitude / k) / np.arcsinh(ref / k)
    return np.clip(compressed, 0.0, 1.0)


def to_asinh(img, k, ref):
    return (asinh_compress(img, k, ref) * 255).astype(np.uint8)


def compute_dataset_reference(input_dir):
    '''
    Dataset-wide asinh reference: median, over every saved raw-amplitude .npy in input_dir, of
    that image's 99.9th percentile pixel. Not wired in yet -- every asinh call site below
    computes ref from just the one image being displayed; this is here for when a real
    cross-dataset reference is wanted instead.

    inputs:
        input_dir (Path): directory of saved side_scan_amp_*.npy raw amplitude arrays
    outputs:
        ref (float): dataset reference level, feeds to_asinh/asinh_compress's ref argument
    '''
    refs = []
    for f in input_dir.glob('*.npy'):
        img = np.load(f)
        refs.append(np.percentile(img, 99.9))
    return np.median(refs)
