import numpy as np

from .mdct import MDCT
from .psychoacoustic import PsychoacousticModel
from .bitalloc import allocate_bits
from .quantize import quantize_band
from .bitstream import BitWriter, signed_to_unsigned
from .format import pack_mono_blob


def encode_mono(x, sample_rate, N=1024, num_bands=32, bits_per_frame=None,
                target_kbps=128.0, masking_offset_db=6.0):
    """Encode a single channel of float audio (in [-1, 1]) into a mono blob.

    bits_per_frame: explicit spectral-data bit budget per frame; if None,
        derived from target_kbps (kilobits/sec) and the frame hop N.
    """
    x = np.asarray(x, dtype=np.float64)
    num_samples = len(x)

    if bits_per_frame is None:
        bits_per_frame = int(target_kbps * 1000.0 * N / sample_rate)

    mdct = MDCT(N)
    psy = PsychoacousticModel(N, sample_rate, num_bands=num_bands,
                               masking_offset_db=masking_offset_db)

    # Zero-pad: N zeros at the front (so the first real sample lands at the
    # center of the first analysis window) and enough at the back so every
    # sample is fully covered by at least one frame, with hop = N.
    pad_front = N
    tail = (N - (num_samples % N)) % N
    pad_back = N + tail
    xp = np.concatenate([np.zeros(pad_front), x, np.zeros(pad_back)])

    num_frames = (len(xp) - 2 * N) // N + 1

    band_bits_all = np.zeros((num_frames, num_bands), dtype=np.uint8)
    band_scale_all = np.zeros((num_frames, num_bands), dtype=np.float64)
    writer = BitWriter()

    for f in range(num_frames):
        start = f * N
        block = xp[start:start + 2 * N]
        X = mdct.forward(block)

        thresh_lin, energy_lin = psy.masking_thresholds(X)
        bits = allocate_bits(energy_lin, thresh_lin, psy.band_sizes, bits_per_frame)

        for b in range(num_bands):
            mask = psy.band_of_line == b
            band_vals = X[mask]
            nb = int(bits[b])
            scale = float(np.max(np.abs(band_vals))) if len(band_vals) else 0.0
            band_bits_all[f, b] = nb
            band_scale_all[f, b] = scale
            if nb > 0 and scale > 0:
                codes = quantize_band(band_vals, nb, scale)
                ucodes = signed_to_unsigned(codes, nb)
                for c in ucodes:
                    writer.write(int(c), nb)
            # nb == 0 -> nothing written, decoder reconstructs zeros for this band

    packed_bits = writer.getvalue()

    return pack_mono_blob(sample_rate, num_samples, N, num_bands,
                           psy.band_sizes, band_bits_all, band_scale_all,
                           packed_bits)
