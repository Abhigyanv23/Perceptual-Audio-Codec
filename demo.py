#!/usr/bin/env python3
"""Demo: generate a test signal, run it through the codec at a few bitrates,
report SNR and compression ratio, and plot a waveform/spectrum comparison."""
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from pac.codec import encode_wav, decode_wav

OUT_DIR = 'demo_out'
os.makedirs(OUT_DIR, exist_ok=True)


def make_test_signal(sr=44100, dur=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    x = (0.30 * np.sin(2 * np.pi * 440 * t)
         + 0.20 * np.sin(2 * np.pi * 1000 * t)
         + 0.10 * np.sin(2 * np.pi * 3000 * t)
         + 0.05 * np.sin(2 * np.pi * 8000 * t))
    x += 0.01 * rng.standard_normal(len(t))
    return sr, np.clip(x, -1, 1)


def snr_db(ref, test):
    n = min(len(ref), len(test))
    err = ref[:n] - test[:n]
    return 10 * np.log10(np.mean(ref[:n] ** 2) / max(np.mean(err ** 2), 1e-20))


def main():
    sr, x = make_test_signal()
    in_path = os.path.join(OUT_DIR, 'input.wav')
    wavfile.write(in_path, sr, (x * 32767).astype(np.int16))

    bitrates = [32, 64, 96, 128, 192, 256]
    results = []
    for kbps in bitrates:
        pac_path = os.path.join(OUT_DIR, f'encoded_{kbps}kbps.pac')
        wav_path = os.path.join(OUT_DIR, f'decoded_{kbps}kbps.wav')
        encode_wav(in_path, pac_path, target_kbps=kbps)
        _, y = decode_wav(pac_path, wav_path)
        y = y.astype(np.float64) if y.ndim == 1 else y[:, 0].astype(np.float64)
        y /= 32768.0 if np.max(np.abs(y)) > 2 else 1.0  # decode_wav returns float already; guard just in case
        ratio = os.path.getsize(in_path) / os.path.getsize(pac_path)
        s = snr_db(x, y)
        results.append((kbps, ratio, s))
        print(f"{kbps:>4} kbps  |  compression {ratio:5.2f}x  |  SNR {s:6.2f} dB")

    # Plot SNR vs bitrate
    kbps_list = [r[0] for r in results]
    snr_list = [r[2] for r in results]
    ratio_list = [r[1] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(kbps_list, snr_list, 'o-')
    axes[0].set_xlabel('Target bitrate (kbps)')
    axes[0].set_ylabel('SNR (dB)')
    axes[0].set_title('Quality vs bitrate')
    axes[0].grid(alpha=0.3)

    axes[1].plot(kbps_list, ratio_list, 'o-', color='darkorange')
    axes[1].set_xlabel('Target bitrate (kbps)')
    axes[1].set_ylabel('Compression ratio (x)')
    axes[1].set_title('Compression vs bitrate')
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'bitrate_vs_quality.png'), dpi=130)
    print(f"\nSaved plot to {OUT_DIR}/bitrate_vs_quality.png")

    # Waveform snippet comparison at 128 kbps
    _, y128 = decode_wav(os.path.join(OUT_DIR, 'encoded_128kbps.pac'),
                          os.path.join(OUT_DIR, '_tmp.wav'))
    if y128.ndim > 1:
        y128 = y128[:, 0]
    n0, n1 = 1000, 1300
    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    ax2.plot(x[n0:n1], label='original', lw=1.5)
    ax2.plot(y128[n0:n1], label='decoded (128 kbps)', lw=1.2, ls='--')
    ax2.set_title('Waveform: original vs decoded')
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, 'waveform_comparison.png'), dpi=130)
    print(f"Saved plot to {OUT_DIR}/waveform_comparison.png")


if __name__ == '__main__':
    main()
