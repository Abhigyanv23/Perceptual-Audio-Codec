import numpy as np

from .mdct import MDCT
from .quantize import dequantize_band
from .bitstream import BitReader, unsigned_to_signed
from .format import unpack_mono_blob


def decode_mono(blob):
    """Decode a mono blob (see format.py) back into a float64 audio array
    of length `num_samples`."""
    meta = unpack_mono_blob(blob)
    N = meta['N']
    num_bands = meta['num_bands']
    num_frames = meta['num_frames']
    band_sizes = meta['band_sizes']
    band_bits = meta['band_bits']
    band_scale = meta['band_scale']

    # Rebuild band_of_line assignment purely from band_sizes (order preserved).
    band_of_line = np.repeat(np.arange(num_bands), band_sizes)
    # band_sizes should sum to N; if not (rounding), pad/truncate defensively.
    if len(band_of_line) < N:
        band_of_line = np.concatenate([band_of_line, np.full(N - len(band_of_line), num_bands - 1)])
    elif len(band_of_line) > N:
        band_of_line = band_of_line[:N]

    mdct = MDCT(N)
    reader = BitReader(meta['packed_bits'])

    out_len = (num_frames - 1) * N + 2 * N
    out = np.zeros(out_len, dtype=np.float64)

    for f in range(num_frames):
        X = np.zeros(N, dtype=np.float64)
        for b in range(num_bands):
            mask = band_of_line == b
            nb = int(band_bits[f, b])
            scale = band_scale[f, b]
            count = int(mask.sum())
            if nb > 0 and scale > 0 and count > 0:
                ucodes = np.array([reader.read(nb) for _ in range(count)], dtype=np.int64)
                codes = unsigned_to_signed(ucodes, nb)
                X[mask] = dequantize_band(codes, nb, scale)
            # else leave zeros

        y = mdct.inverse(X)
        start = f * N
        out[start:start + 2 * N] += y

    # Undo the encoder's zero-padding: it added N samples at the front.
    trimmed = out[N:N + meta['num_samples']]
    return trimmed
