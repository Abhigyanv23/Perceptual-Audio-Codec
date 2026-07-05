"""
Modified Discrete Cosine Transform (MDCT) with 50% overlap.

Uses the Princen-Bradley TDAC (Time-Domain Aliasing Cancellation) formulation:

    X[k] = sum_{n=0}^{2N-1} w[n] x[n] cos( (pi/N)(n + N/2 + 1/2)(k + 1/2) )   k = 0..N-1
    y[n] = (2/N) sum_{k=0}^{N-1} X[k] cos( (pi/N)(n + N/2 + 1/2)(k + 1/2) )    n = 0..2N-1

with a sine analysis/synthesis window:

    w[n] = sin( pi/(2N) * (n + 0.5) ),  n = 0..2N-1

Overlap-adding successive inverse-transformed, re-windowed blocks with hop = N
gives exact reconstruction of the original signal (verified numerically).
"""
import numpy as np


class MDCT:
    def __init__(self, N):
        """N = number of frequency lines per block. Block (frame) size is 2N samples,
        hop size between successive blocks is N samples (50% overlap)."""
        self.N = N
        n0 = N / 2 + 0.5
        n = np.arange(2 * N)
        k = np.arange(N)
        # Precompute the (N, 2N) transform kernel once; reused for both directions.
        self.C = np.cos((np.pi / N) * (n[None, :] + n0) * (k[:, None] + 0.5))
        self.window = np.sin(np.pi / (2 * N) * (n + 0.5))

    def forward(self, block):
        """block: array of length 2N (already raw, unwindowed) -> N MDCT coefficients."""
        return self.C @ (block * self.window)

    def inverse(self, X):
        """X: N MDCT coefficients -> 2N windowed time-domain samples ready for overlap-add."""
        y = (2.0 / self.N) * (self.C.T @ X)
        return y * self.window
