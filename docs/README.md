# Perceptual Audio Codec — live browser demo

A static, dependency-free web page that runs the codec's full DSP pipeline
(MDCT → psychoacoustic masking → bit allocation → quantization) for real,
client-side, on audio you drop in. No server, no build step, no upload —
everything happens in your browser tab, in a Web Worker so the UI stays
responsive.

This is a JavaScript port of the Python reference implementation in the
parent repo. The math (MDCT constants, Bark-scale spreading, greedy bit
allocation) is the same; only the language changed. It's been numerically
verified against the Python version (perfect MDCT reconstruction to
float32 precision, matching SNR behavior).

## Run it locally

Browsers block Web Workers from loading over the `file://` protocol, so you
need a tiny local server rather than double-clicking `index.html`:

```bash
cd web
python3 -m http.server 8080
# then open http://localhost:8080
```

(Any static file server works — `npx serve`, VS Code's "Live Server"
extension, etc.)

## Deploy to GitHub Pages

1. Push this `web/` folder (or its contents) to your repo.
2. Repo Settings → **Pages** → set the source to the branch/folder containing
   `index.html`.
3. GitHub gives you a `https://<username>.github.io/<repo>/` URL — done.

No build step, no `package.json`, no dependencies to install — it's plain
HTML/CSS/JS.

## What's actually happening in the browser

1. Your audio file is decoded with the Web Audio API's `decodeAudioData` and
   downmixed to mono.
2. The first 15 seconds are handed to a Web Worker (`js/codec-worker.js`),
   which runs the same MDCT → masking → bit-allocation → quantization
   pipeline as the Python version, entirely on the CPU, no ML/inference.
3. Results come back: a reconstructed waveform (playable + downloadable as
   a WAV), an SNR number, a compression ratio, and a heatmap of exactly how
   many bits the allocator spent on each critical band, frame by frame —
   the direct visual output of the masking model's decisions.

## Files

```
web/
  index.html          page structure and copy
  style.css            all styling
  js/
    codec-worker.js    DSP core: MDCT, psychoacoustic model, bit allocation,
                        quantization — runs off the main thread
    app.js              UI: file handling, Web Audio decode, worker
                        orchestration, canvas rendering, playback
```

## Known limitations (stated in the page itself, too)

- Processes the first 15 seconds of any uploaded file, for responsiveness —
  the DSP here is a straightforward O(N²) matrix-based MDCT (simple to
  verify correct; a production codec would use a fast FFT-based MDCT to
  process full-length audio quickly).
- No entropy coding, block switching, or tonality-dependent masking offset —
  see the main repo README for the full list; this demo intentionally keeps
  the same simplifications as the Python reference so the two stay
  comparable.
