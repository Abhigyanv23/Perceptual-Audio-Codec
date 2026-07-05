"""
Uniform mid-tread quantizer, applied per critical band with its own scale
factor (the band's peak absolute MDCT coefficient). A band allocated `bits`
bits per line is represented with 2**bits signed integer levels spanning
[-scale, +scale].
"""
import numpy as np


def quantize_band(values, bits, scale):
    """values: 1D array (one band's MDCT lines). bits: int. scale: float (>0).
    Returns integer codes (numpy int32 array)."""
    if bits <= 0 or scale <= 0:
        return np.zeros(len(values), dtype=np.int32)
    levels = 2 ** bits
    step = (2.0 * scale) / levels
    codes = np.round(values / step).astype(np.int32)
    lo, hi = -(levels // 2), levels // 2 - 1
    return np.clip(codes, lo, hi)


def dequantize_band(codes, bits, scale):
    if bits <= 0 or scale <= 0:
        return np.zeros(len(codes), dtype=np.float64)
    levels = 2 ** bits
    step = (2.0 * scale) / levels
    return codes.astype(np.float64) * step
