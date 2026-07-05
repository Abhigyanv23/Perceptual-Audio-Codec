# Perceptual Audio Codec

A lossy audio codec built from first principles — the same core ideas behind MP3 and AAC:
MDCT transform coding driven by a psychoacoustic masking model.

**[Try the live demo →](https://abhigyanv23.github.io/Perceptual-Audio-Codec/)**
Drop in your own audio and watch the bit allocator decide what you can't hear — runs entirely
in your browser, no upload, no server.

```
WAV in --> [MDCT] --> [psychoacoustic model] --> [bit allocation] --> [quantize] --> bitstream
bitstream --> [unpack] --> [dequantize] --> [inverse MDCT + overlap-add] --> WAV out
```

## What's in this repo

This project has two halves, both implementing the same pipeline:

| | [`pac/`](./pac) | [`docs/`](./docs) |
|---|---|---|
| Language | Python | JavaScript |
| Purpose | Reference implementation | Interactive live demo |
| Output | Real `.pac` bitstream file you can decode later | In-browser visualization, no file format |
| Extras | CLI, unit tests, bitrate/quality plots | Waveform + bit-allocation heatmap, playable A/B audio |
| Run it | `python pac_cli.py encode …` | [Live on GitHub Pages](https://abhigyanv23.github.io/Perceptual-Audio-Codec/) |

They're numerically equivalent — the JS port was verified against the Python version's MDCT
perfect-reconstruction property and SNR behavior before anything else was built on top of it.

## How it works

**1. MDCT** — Audio is split into overlapping windows (50% overlap, sine window) and transformed
with the Modified Discrete Cosine Transform. Thanks to Time-Domain Aliasing Cancellation (TDAC),
overlap-adding the inverse-transformed blocks reconstructs the original signal exactly.

**2. Psychoacoustic model** — MDCT lines are grouped into critical bands spaced along the
**Bark scale** (how the cochlea actually resolves frequency). Band energies are spread across
neighbors with a Schroeder-style spreading function (a loud tone masks nearby frequencies), then
floored against the absolute threshold of human hearing.

**3. Bit allocation** — A greedy water-filling loop hands the next bit to whichever band
currently has the worst noise-to-mask ratio, until the bitrate budget runs out. Same idea used
in MPEG Layer I/II.

**4. Quantization** — Each band gets a uniform quantizer scaled to its own peak coefficient, at
whatever bit-width the allocator decided.

**5. Reconstruction** — Dequantize, inverse MDCT, overlap-add back into audio.

## Quickstart

### Python reference implementation
```bash
cd pac
pip install -r requirements.txt

python pac_cli.py encode input.wav output.pac --kbps 128
python pac_cli.py decode output.pac reconstructed.wav

python tests/test_codec.py   # unit tests
python demo.py                # bitrate-vs-quality plots -> demo_out/
```
See [`pac/README.md`](./pac/README.md) for the full API, file format, and tunable parameters.

### Web demo, locally
```bash
cd docs
python -m http.server 8080
# open http://localhost:8080
```
See [`docs/README.md`](./docs/README.md) for deployment details.

## Measured results

On the bundled synthetic test signal:

| Bitrate | Compression | SNR |
|--:|--:|--:|
| 32 kbps  | 10.7x | 18.5 dB |
| 64 kbps  | 7.2x  | 22.9 dB |
| 96 kbps  | 5.4x  | 27.8 dB |
| 128 kbps | 4.3x  | 31.1 dB |
| 192 kbps | 3.1x  | 37.9 dB |
| 256 kbps | 2.4x  | 47.1 dB |

## Known simplifications vs. a production codec (MP3/AAC/Opus)

- **No entropy coding** (Huffman/arithmetic) on top of the quantized codes.
- **No block switching** — a single fixed MDCT size, so transients can cause pre-echo.
- **Simplified masking offset** — one fixed dB margin for all bands, rather than a
  tonality-dependent offset.
- **No stereo joint coding** (mid/side, intensity stereo) — channels are encoded independently.
- **Web demo processes the first 15 seconds** of any upload, for in-browser responsiveness
  (the JS MDCT is a straightforward matrix implementation rather than a fast FFT-based one).

These are natural next steps if you want to push this further.

## Layout

```
pac/                    Python reference implementation
  mdct.py                MDCT/IMDCT transform
  psychoacoustic.py       Bark scale, spreading function, masking thresholds
  bitalloc.py             Greedy bit allocation
  quantize.py             Uniform quantizer
  bitstream.py            Bit-level packing
  format.py               Binary .pac file format
  encoder.py / decoder.py Mono channel encode/decode pipeline
  codec.py                WAV-level encode_wav / decode_wav API
  pac_cli.py              Command-line tool
  demo.py                 Bitrate/quality demo + plots
  tests/test_codec.py     Unit tests

docs/                    Live browser demo (GitHub Pages)
  index.html              Page structure and copy
  style.css               Styling
  js/codec-worker.js       DSP core, ported to JS, runs in a Web Worker
  js/app.js                UI: file handling, Web Audio decode, canvas rendering, playback
```
