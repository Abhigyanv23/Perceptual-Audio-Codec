import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pac.mdct import MDCT
from pac.bitstream import BitWriter, BitReader, signed_to_unsigned, unsigned_to_signed
from pac.encoder import encode_mono
from pac.decoder import decode_mono


def test_mdct_perfect_reconstruction():
    N = 256
    mdct = MDCT(N)
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(N * 12)
    hop = N
    out = np.zeros(len(sig) + 2 * N)
    i = 0
    while i + 2 * N <= len(sig):
        X = mdct.forward(sig[i:i + 2 * N])
        y = mdct.inverse(X)
        out[i:i + 2 * N] += y
        i += hop
    a = sig[2 * N: len(sig) - 2 * N]
    b = out[2 * N: len(sig) - 2 * N]
    assert np.max(np.abs(a - b)) < 1e-9


def test_bitwriter_reader_roundtrip():
    rng = np.random.default_rng(1)
    widths = [3, 7, 1, 12, 4, 0, 9, 15]
    values = [int(rng.integers(0, max(1, 2 ** w))) for w in widths]
    w = BitWriter()
    for v, wd in zip(values, widths):
        w.write(v, wd)
    data = w.getvalue()
    r = BitReader(data)
    for v, wd in zip(values, widths):
        if wd == 0:
            continue
        assert r.read(wd) == v


def test_signed_unsigned_roundtrip():
    bits = 6
    codes = np.arange(-(2 ** (bits - 1)), 2 ** (bits - 1))
    u = signed_to_unsigned(codes, bits)
    s = unsigned_to_signed(u, bits)
    assert np.array_equal(codes, s)


def test_encode_decode_roundtrip_quality():
    sr = 22050
    t = np.arange(sr * 1) / sr
    x = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.default_rng(2).standard_normal(len(t))
    x = np.clip(x, -1, 1)
    blob = encode_mono(x, sr, N=512, num_bands=24, target_kbps=128)
    y = decode_mono(blob)
    assert len(y) == len(x)
    n = min(len(x), len(y))
    err = x[:n] - y[:n]
    snr = 10 * np.log10(np.mean(x[:n] ** 2) / np.mean(err ** 2))
    assert snr > 15  # sanity floor; real number is much higher at 128kbps


def test_silence_does_not_crash():
    sr = 8000
    x = np.zeros(sr)
    blob = encode_mono(x, sr, N=256, num_bands=16, target_kbps=64)
    y = decode_mono(blob)
    assert len(y) == len(x)
    assert np.max(np.abs(y)) < 1e-6


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
