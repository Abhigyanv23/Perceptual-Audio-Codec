"""Minimal MSB-first bit-level packer/unpacker used for the variable-width
quantized MDCT codes (scale factors and bit-allocation tables use fixed-width
numpy dtypes and are stored separately in the container -- see format.py)."""
import numpy as np


class BitWriter:
    def __init__(self):
        self._bytes = bytearray()
        self._cur = 0
        self._nbits = 0

    def write(self, value, nbits):
        """Write the low `nbits` bits of `value` (unsigned), MSB first."""
        if nbits == 0:
            return
        value &= (1 << nbits) - 1
        self._cur = (self._cur << nbits) | value
        self._nbits += nbits
        while self._nbits >= 8:
            self._nbits -= 8
            byte = (self._cur >> self._nbits) & 0xFF
            self._bytes.append(byte)
        self._cur &= (1 << self._nbits) - 1 if self._nbits else 0

    def getvalue(self):
        """Flush remaining bits (zero-padded) and return bytes."""
        if self._nbits > 0:
            byte = (self._cur << (8 - self._nbits)) & 0xFF
            return bytes(self._bytes) + bytes([byte])
        return bytes(self._bytes)


class BitReader:
    def __init__(self, data):
        self._data = data
        self._byte_pos = 0
        self._cur = 0
        self._nbits = 0

    def read(self, nbits):
        if nbits == 0:
            return 0
        while self._nbits < nbits:
            b = self._data[self._byte_pos]
            self._byte_pos += 1
            self._cur = (self._cur << 8) | b
            self._nbits += 8
        self._nbits -= nbits
        value = (self._cur >> self._nbits) & ((1 << nbits) - 1)
        self._cur &= (1 << self._nbits) - 1 if self._nbits else 0
        return value


def signed_to_unsigned(codes, bits):
    """Map signed integer codes in [-2^(bits-1), 2^(bits-1)-1] to unsigned
    [0, 2^bits - 1] for bit-packing (simple offset encoding)."""
    return (codes + (1 << (bits - 1))).astype(np.int64)


def unsigned_to_signed(u, bits):
    return u.astype(np.int64) - (1 << (bits - 1))
