"""
Simplified psychoacoustic model.

Pipeline per frame:
  1. Compute MDCT-line energies (already have MDCT coefficients from mdct.py).
  2. Group lines into critical bands using the Bark scale.
  3. Compute band energies, spread them across bands using a Schroeder-style
     spreading function (models basilar-membrane masking spread).
  4. Subtract a fixed masking offset to get a masking threshold per band.
  5. Floor the result at the absolute threshold of hearing (quiet threshold).
  6. Report, per band: signal energy, masking threshold, and signal-to-mask
     ratio (SMR) -- used later for bit allocation.
"""
import numpy as np


def hz_to_bark(f):
    f = np.maximum(f, 1e-6)
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def absolute_threshold_db(f_hz):
    """Terhardt's approximation of the absolute threshold of hearing, in dB SPL.
    f_hz: frequency in Hz (array). Valid roughly for 20 Hz - 20 kHz; clipped outside."""
    f = np.clip(f_hz, 20, 20000) / 1000.0  # kHz
    thr = (3.64 * f ** -0.8
           - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2)
           + 1e-3 * f ** 4)
    return thr


class PsychoacousticModel:
    def __init__(self, N, sample_rate, num_bands=32, masking_offset_db=6.0):
        """
        N: number of MDCT lines per frame.
        sample_rate: audio sample rate in Hz.
        num_bands: number of (roughly Bark-spaced) critical bands to group lines into.
        masking_offset_db: fixed safety-margin subtracted from the spread excitation
            to get the actual masking threshold (a simplification of the
            tonality-dependent offset used in full MPEG psychoacoustic models).
        """
        self.N = N
        self.sr = sample_rate
        self.num_bands = num_bands
        self.masking_offset_db = masking_offset_db

        # Center frequency of each MDCT line (line k covers freq k * sr / (2N) roughly).
        self.line_freqs = (np.arange(N) + 0.5) * (sample_rate / 2.0) / N
        self.line_barks = hz_to_bark(self.line_freqs)

        # Split the Bark range into num_bands equal-width bands.
        bark_max = self.line_barks[-1]
        edges = np.linspace(0, bark_max + 1e-6, num_bands + 1)
        self.band_of_line = np.clip(np.digitize(self.line_barks, edges) - 1, 0, num_bands - 1)

        # Band center in Bark, used for the spreading function.
        self.band_bark_centers = 0.5 * (edges[:-1] + edges[1:])

        # Precompute band sizes (number of MDCT lines per band).
        self.band_sizes = np.array([(self.band_of_line == b).sum() for b in range(num_bands)])
        self.band_sizes = np.maximum(self.band_sizes, 1)  # avoid empty bands

        # Precompute the Bark-domain spreading matrix S[i, j]: how much band j's
        # energy contributes (in the linear/energy domain, via dB) to band i.
        dz = self.band_bark_centers[:, None] - self.band_bark_centers[None, :]
        spread_db = 15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1 + (dz + 0.474) ** 2)
        self.spread_lin = 10 ** (spread_db / 10.0)

        # Absolute threshold of hearing per band (evaluated at band's mean line freq).
        band_freqs = np.array([
            self.line_freqs[self.band_of_line == b].mean() if (self.band_of_line == b).any()
            else self.line_freqs[-1]
            for b in range(num_bands)
        ])
        self.band_quiet_thresh_db = absolute_threshold_db(band_freqs)

    def band_energies(self, mdct_coeffs):
        """Sum of squared MDCT coefficients per band."""
        energies = np.zeros(self.num_bands)
        for b in range(self.num_bands):
            mask = self.band_of_line == b
            energies[b] = np.sum(mdct_coeffs[mask] ** 2)
        return energies

    def masking_thresholds(self, mdct_coeffs, ref_energy=1e-12):
        """Returns (band_thresh_lin, band_energy_lin): per-band masking threshold
        and signal energy, both in linear (not dB) units, floored so they are
        never zero (to keep downstream log/ratio math well-defined)."""
        energies = self.band_energies(mdct_coeffs)
        energies = np.maximum(energies, ref_energy)

        # Spread excitation across bands (energy domain matmul with spreading matrix).
        spread_energy = self.spread_lin @ energies

        # Convert to dB (relative to a reference), apply masking offset, then
        # convert back, and floor with the absolute threshold of hearing.
        spread_db = 10 * np.log10(spread_energy / ref_energy)
        masked_db = spread_db - self.masking_offset_db
        thresh_db = np.maximum(masked_db, self.band_quiet_thresh_db)
        thresh_lin = ref_energy * 10 ** (thresh_db / 10.0)

        return thresh_lin, energies
