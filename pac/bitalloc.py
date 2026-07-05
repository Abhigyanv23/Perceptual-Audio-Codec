"""
Greedy bit allocation across critical bands.

Classic MPEG-style algorithm: each additional bit given to a band reduces
that band's quantization noise by ~6.02 dB (one bit ~ one doubling of
quantizer levels ~ 6.02 dB of SNR for a uniform quantizer). We repeatedly
hand the next bit to whichever band currently has the worst noise-to-mask
ratio (NMR = noise level - masking threshold, in dB), until the bit budget
for the frame is exhausted. This concentrates bits where the ear would
otherwise notice quantization noise poking above the masking threshold.
"""
import numpy as np

DB_PER_BIT = 6.02
MAX_BITS_PER_LINE = 15


def allocate_bits(band_energies, band_thresholds, band_sizes, total_bits, ref_energy=1e-12):
    """
    band_energies, band_thresholds: linear-domain per-band values (same length).
    band_sizes: number of MDCT lines in each band (allocating 1 bit to a band
        costs band_sizes[b] bits, since every line in the band gets that bit width).
    total_bits: total bit budget available for this frame's spectral data.

    Returns: integer array of bits-per-band (0..MAX_BITS_PER_LINE).
    """
    num_bands = len(band_energies)
    bits = np.zeros(num_bands, dtype=int)

    # Per-line "noise" power starts at the full unquantized signal power
    # (i.e. worst case: transmitting nothing leaves all the energy as "noise").
    per_line_energy = band_energies / np.maximum(band_sizes, 1)
    noise_db = 10 * np.log10(np.maximum(per_line_energy, ref_energy) / ref_energy)
    thresh_db = 10 * np.log10(np.maximum(band_thresholds, ref_energy) / ref_energy)

    remaining = total_bits
    # Bands with essentially no energy don't need bits.
    active = per_line_energy > ref_energy * 2

    while remaining > 0:
        nmr = noise_db - thresh_db
        nmr[~active] = -np.inf
        nmr[bits >= MAX_BITS_PER_LINE] = -np.inf
        # Also disable bands we can no longer afford.
        affordable = band_sizes <= remaining
        nmr[~affordable] = -np.inf

        if not np.any(np.isfinite(nmr)):
            break

        b = int(np.argmax(nmr))
        if nmr[b] == -np.inf:
            break

        bits[b] += 1
        noise_db[b] -= DB_PER_BIT
        remaining -= band_sizes[b]

    return bits
