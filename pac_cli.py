#!/usr/bin/env python3
"""Command-line interface for the perceptual audio codec.

Usage:
    python pac_cli.py encode input.wav output.pac [--kbps 128] [--N 1024] [--bands 32]
    python pac_cli.py decode output.pac reconstructed.wav
"""
import argparse
import os
import time

from pac.codec import encode_wav, decode_wav


def main():
    parser = argparse.ArgumentParser(description="Perceptual Audio Codec (MDCT + psychoacoustic masking)")
    sub = parser.add_subparsers(dest='cmd', required=True)

    enc = sub.add_parser('encode', help='Encode a .wav file to .pac')
    enc.add_argument('input', help='Input .wav path')
    enc.add_argument('output', help='Output .pac path')
    enc.add_argument('--kbps', type=float, default=128.0, help='Target bitrate (kbps), default 128')
    enc.add_argument('--N', type=int, default=1024, help='MDCT half-block size, default 1024')
    enc.add_argument('--bands', type=int, default=32, help='Number of critical bands, default 32')
    enc.add_argument('--mask-offset', type=float, default=6.0,
                      help='Masking safety margin in dB, default 6.0')

    dec = sub.add_parser('decode', help='Decode a .pac file back to .wav')
    dec.add_argument('input', help='Input .pac path')
    dec.add_argument('output', help='Output .wav path')

    args = parser.parse_args()

    if args.cmd == 'encode':
        t0 = time.time()
        encode_wav(args.input, args.output, N=args.N, num_bands=args.bands,
                   target_kbps=args.kbps, masking_offset_db=args.mask_offset)
        dt = time.time() - t0
        in_sz = os.path.getsize(args.input)
        out_sz = os.path.getsize(args.output)
        print(f"Encoded {args.input} -> {args.output}")
        print(f"  {in_sz:,} bytes -> {out_sz:,} bytes  (ratio {in_sz/out_sz:.2f}x)")
        print(f"  encode time: {dt:.2f}s")

    elif args.cmd == 'decode':
        t0 = time.time()
        decode_wav(args.input, args.output)
        dt = time.time() - t0
        print(f"Decoded {args.input} -> {args.output}")
        print(f"  decode time: {dt:.2f}s")


if __name__ == '__main__':
    main()
