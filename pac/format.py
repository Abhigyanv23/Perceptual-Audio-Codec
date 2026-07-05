"""Binary file format for the codec.

Top-level container ('PACM'):
    magic: 4 bytes = b'PACM'
    num_channels: uint8
    for each channel:
        blob_len: uint32 (little-endian)
        blob: bytes  (a self-contained mono stream, see below)

Per-channel mono blob ('PAC1'):
    magic: 4 bytes = b'PAC1'
    sample_rate: uint32
    num_samples: uint32       (original sample count, pre-padding)
    N: uint16                 (MDCT half-block size)
    num_bands: uint16
    num_frames: uint32
    band_sizes: num_bands x uint16      (# MDCT lines per critical band)
    band_bits:  num_frames x num_bands x uint8
    band_scale: num_frames x num_bands x float16
    packed_bits: remaining bytes -- continuous MSB-first bitstream of
        quantized MDCT line codes, frame by frame, band by band, line by
        line, each code `band_bits[frame,band]` bits wide (offset-binary).
"""
import struct
import numpy as np

MONO_MAGIC = b'PAC1'
TOP_MAGIC = b'PACM'


def pack_mono_blob(sample_rate, num_samples, N, num_bands, band_sizes,
                    band_bits, band_scale, packed_bits):
    num_frames = band_bits.shape[0]
    header = struct.pack('<4sIIHHI', MONO_MAGIC, sample_rate, num_samples,
                          N, num_bands, num_frames)
    header += band_sizes.astype('<u2').tobytes()
    header += band_bits.astype('<u1').tobytes()
    header += band_scale.astype('<f2').tobytes()
    return header + packed_bits


def unpack_mono_blob(blob):
    magic, sample_rate, num_samples, N, num_bands, num_frames = struct.unpack(
        '<4sIIHHI', blob[:20])
    assert magic == MONO_MAGIC, "bad mono blob magic"
    off = 20
    band_sizes = np.frombuffer(blob, dtype='<u2', count=num_bands, offset=off).copy()
    off += num_bands * 2
    band_bits = np.frombuffer(blob, dtype='<u1', count=num_frames * num_bands, offset=off) \
        .reshape(num_frames, num_bands).copy()
    off += num_frames * num_bands * 1
    band_scale = np.frombuffer(blob, dtype='<f2', count=num_frames * num_bands, offset=off) \
        .reshape(num_frames, num_bands).astype(np.float64).copy()
    off += num_frames * num_bands * 2
    packed_bits = blob[off:]
    return dict(sample_rate=sample_rate, num_samples=num_samples, N=N,
                num_bands=num_bands, num_frames=num_frames,
                band_sizes=band_sizes, band_bits=band_bits,
                band_scale=band_scale, packed_bits=packed_bits)


def pack_container(channel_blobs):
    out = bytearray()
    out += struct.pack('<4sB', TOP_MAGIC, len(channel_blobs))
    for blob in channel_blobs:
        out += struct.pack('<I', len(blob))
        out += blob
    return bytes(out)


def unpack_container(data):
    magic, num_channels = struct.unpack('<4sB', data[:5])
    assert magic == TOP_MAGIC, "bad container magic"
    off = 5
    blobs = []
    for _ in range(num_channels):
        (blen,) = struct.unpack('<I', data[off:off + 4])
        off += 4
        blobs.append(data[off:off + blen])
        off += blen
    return blobs
