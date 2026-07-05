import numpy as np
from scipy.io import wavfile

from .encoder import encode_mono
from .decoder import decode_mono
from .format import pack_container, unpack_container


def _to_float(data):
    if data.dtype == np.int16:
        return data.astype(np.float64) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float64) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128) / 128.0
    return data.astype(np.float64)


def _to_int16(data):
    data = np.clip(data, -1.0, 1.0)
    return (data * 32767.0).astype(np.int16)


def encode_wav(in_path, out_path, N=1024, num_bands=32, target_kbps=128.0,
               masking_offset_db=6.0):
    sample_rate, data = wavfile.read(in_path)
    data = _to_float(data)
    if data.ndim == 1:
        channels = [data]
    else:
        channels = [data[:, c] for c in range(data.shape[1])]

    blobs = [
        encode_mono(ch, sample_rate, N=N, num_bands=num_bands,
                    target_kbps=target_kbps, masking_offset_db=masking_offset_db)
        for ch in channels
    ]
    container = pack_container(blobs)
    with open(out_path, 'wb') as f:
        f.write(container)
    return container


def decode_wav(in_path, out_path):
    with open(in_path, 'rb') as f:
        data = f.read()
    blobs = unpack_container(data)
    channels = [decode_mono(b) for b in blobs]

    # All channels share sample_rate/num_samples; read it back from the first blob.
    from .format import unpack_mono_blob
    sample_rate = unpack_mono_blob(blobs[0])['sample_rate']

    if len(channels) == 1:
        out = channels[0]
    else:
        out = np.stack(channels, axis=1)
    wavfile.write(out_path, sample_rate, _to_int16(out))
    return sample_rate, out
