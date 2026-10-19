'''
Every noise model the simulator applies, selected by a baseline's noise_type in config.json.

    'none'     no noise
    'awgn'     complex white Gaussian receiver noise on the signal, at snr_db (none when snr_db is None)
    'speckle'  a fixed random phase on every scatter, keyed to where it sits on the surface

noise_type may also be a list, e.g. ["speckle", "awgn"], to apply both. The two stages run at
different points of a render: apply_scatter_noise right after accumulate_scatters, on the
scatters, and apply_signal_noise after interpolate_signal, on the range signals.

Speckle is the coherent sum of many sub-resolution scatterers with effectively random phase.
Smooth mesh faces have none, so each scatter gets exp(j*phi) with phi drawn from a hash of its
world position snapped to a lattice of spacing speckle_correlation_length. Every pulse that hits
the same patch of surface therefore sees the same phase, as with a real rough surface. Keep the
spacing above the spacing between ray hits, so a patch is hit on every pulse, and below the
resolution cell, so each cell sums several independent patches (fully developed speckle).
'''
import numpy as np
import torch


NOISE_TYPES = ('none', 'awgn', 'speckle')

# keeps hashed values inside 32 bits, so the int64 products below never overflow
_MASK32 = 0xFFFFFFFF


def parse_noise_type(noise_type):
    '''noise_type as a set of NOISE_TYPES names, from a string or a sequence of them.'''
    names = {noise_type} if isinstance(noise_type, str) else set(noise_type)
    unknown = names - set(NOISE_TYPES)
    assert not unknown, 'unknown noise_type %r, expected some of %r' % (sorted(unknown), NOISE_TYPES)
    return names - {'none'}


def apply_snr(signal, snr_db, dim=-1):
    """
    Apply a signal-to-noise ratio (SNR) to the input signal.

    Args:
        signal (torch.Tensor): The input signal tensor.
        snr_db (float): The desired SNR in decibels.
        dim (int): The dimension along which to compute the SNR.

    Returns:
        torch.Tensor: The signal with the applied SNR.
    """

    # N = Ni + Nr # real and imaginary parts are iid
    # SNR = Ps/Pn = E[|S|^2]/E[|N|^2]
    # E[|N|^2] = E[|S|^2]/SNR

    # E[|N|^2] = E[Nr^2 + Ni^2] = 2 * E[Nr^2]
    # E[|N|^2]/2 = E[Nr^2] = sigma^2
    # sigma = sqrt( E[|N|^2]/2 ) = sqrt( E[|S|^2]/(2*SNR) )

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Compute the signal power
    signal_power = torch.mean(signal.abs() ** 2, dim=dim, keepdim=True)
    noise_std_dev = torch.sqrt(signal_power / (2 * snr_linear)).cpu().numpy()

    # Generate noise
    noise = torch.tensor(
             np.random.normal(scale=noise_std_dev, size=signal.shape) + \
        1j * np.random.normal(scale=noise_std_dev, size=signal.shape)
    ).to(signal.device, dtype=signal.dtype)

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal


def _hash32(x):
    '''Integer avalanche hash of int64 values in [0, 2^32), returned in the same range.'''
    x = (((x >> 16) ^ x) * 0x45d9f3b) & _MASK32
    x = (((x >> 16) ^ x) * 0x45d9f3b) & _MASK32
    return (x >> 16) ^ x


def speckle_phase(positions, correlation_length, strength=1.0, seed=0):
    '''
    Random phase in [0, 2*pi*strength) for each position, constant over each lattice cell.

    inputs:
        positions (R,3): world positions of the scatters
        correlation_length (float): lattice spacing, the size of one independent surface patch
        strength (float): 1 spreads the phase over the full circle (fully developed speckle);
            below 1 a coherent part survives, as from a surface rough on less than a wavelength
        seed (int): picks the speckle realization
    outputs:
        phase (R,): one phase per position
    '''
    cells = torch.floor(positions / correlation_length).to(torch.int64) & _MASK32  # (R,3)
    h = _hash32(torch.full_like(cells[:, 0], seed & _MASK32))
    for axis in range(3):
        h = _hash32(h ^ cells[:, axis])
    return h.to(positions.dtype) * (2 * np.pi * strength / 2**32)  # (R,)


def apply_speckle(energies, positions, correlation_length, strength=1.0, seed=0):
    '''
    Multiply every scatter by its surface patch's unit phasor. Runs after accumulate_scatters.

    inputs:
        energies (T,)[P,][R']: list of lists of scatter energies, real or complex
        positions (T,)[P,][R',3]: matching scatter world positions
        correlation_length, strength, seed: see speckle_phase
    outputs:
        energies (T,)[P,][R']: complex energies with the speckle phase applied
    '''
    return [[e * torch.exp(1j * speckle_phase(x, correlation_length, strength, seed))
             for e, x in zip(energies_t, positions_t)]
            for energies_t, positions_t in zip(energies, positions)]


def apply_scatter_noise(energies, positions, noise_type, speckle_correlation_length=None,
                        speckle_strength=1.0, seed=0):
    '''
    The scatter-level noise of noise_type, applied to the output of accumulate_scatters.

    inputs:
        energies, positions: as in apply_speckle; positions may be None unless speckle is on
        noise_type (str or sequence): see parse_noise_type
        speckle_correlation_length (float): see speckle_phase, required when speckle is on
        speckle_strength (float): see speckle_phase's strength
    outputs:
        energies (T,)[P,][R']
    '''
    if 'speckle' in parse_noise_type(noise_type):
        assert speckle_correlation_length, 'speckle needs a positive speckle_correlation_length'
        energies = apply_speckle(energies, positions, speckle_correlation_length,
                                 speckle_strength, seed)
    return energies


def apply_signal_noise(signals, noise_type, snr_db=None):
    '''
    The signal-level noise of noise_type, applied to the interpolated range signals.

    inputs:
        signals (T,P,Z): range signals, real or complex
        noise_type (str or sequence): see parse_noise_type
        snr_db (float): awgn SNR over each scene's pulses; None skips the awgn
    outputs:
        signals (T,P,Z)
    '''
    if 'awgn' in parse_noise_type(noise_type) and snr_db is not None:
        # receiver noise is complex baseband; a real (incoherent sonar) signal would drop its imaginary half
        if not torch.is_complex(signals):
            signals = signals.to(torch.complex64)
        T, P, Z = signals.shape
        signals = apply_snr(signals.reshape(T, P * Z), snr_db).reshape(T, P, Z)
    return signals
