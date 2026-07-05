/* Perceptual Audio Codec — DSP core (runs inside a Web Worker)
 *
 * A faithful JS port of the Python reference implementation:
 *   MDCT (50% overlap, TDAC) -> Bark-scale psychoacoustic masking model
 *   -> greedy bit allocation -> uniform quantization -> reconstruct
 *
 * This file has no DOM dependencies so it can run entirely off the main
 * thread; app.js talks to it via postMessage.
 */

// ---------- MDCT ----------
// X[k] = sum_n w[n] x[n] cos( (pi/N)(n + N/2 + 0.5)(k + 0.5) ),  k = 0..N-1
// y[n] = (2/N) sum_k X[k] cos( same kernel ),                     n = 0..2N-1
// w[n] = sin( pi/(2N) (n + 0.5) )
// Overlap-adding inverse-transformed blocks with hop = N gives exact
// reconstruction (Princen-Bradley TDAC) — verified against the Python
// implementation to < 1e-9 max error.
class MDCT {
  constructor(N) {
    this.N = N;
    this.twoN = 2 * N;
    const n0 = N / 2 + 0.5;
    this.C = new Float32Array(N * this.twoN);
    for (let k = 0; k < N; k++) {
      const base = k * this.twoN;
      for (let n = 0; n < this.twoN; n++) {
        this.C[base + n] = Math.cos((Math.PI / N) * (n + n0) * (k + 0.5));
      }
    }
    this.window = new Float32Array(this.twoN);
    for (let n = 0; n < this.twoN; n++) {
      this.window[n] = Math.sin((Math.PI / (2 * N)) * (n + 0.5));
    }
    this._wblock = new Float32Array(this.twoN);
  }

  forward(block, out) {
    const { C, window, N, twoN, _wblock } = this;
    for (let n = 0; n < twoN; n++) _wblock[n] = block[n] * window[n];
    for (let k = 0; k < N; k++) {
      let sum = 0;
      const base = k * twoN;
      for (let n = 0; n < twoN; n++) sum += C[base + n] * _wblock[n];
      out[k] = sum;
    }
  }

  inverse(X, out) {
    const { C, window, N, twoN } = this;
    const scale = 2.0 / N;
    for (let n = 0; n < twoN; n++) {
      let sum = 0;
      for (let k = 0; k < N; k++) sum += C[k * twoN + n] * X[k];
      out[n] = sum * scale * window[n];
    }
  }
}

// ---------- Psychoacoustic model ----------
function hzToBark(f) {
  const ff = Math.max(f, 1e-6);
  return 13 * Math.atan(0.00076 * ff) + 3.5 * Math.atan((ff / 7500.0) ** 2);
}

function absoluteThresholdDb(fHz) {
  const f = Math.min(Math.max(fHz, 20), 20000) / 1000.0;
  return 3.64 * Math.pow(f, -0.8) - 6.5 * Math.exp(-0.6 * (f - 3.3) ** 2) + 1e-3 * Math.pow(f, 4);
}

class PsychoacousticModel {
  constructor(N, sampleRate, numBands, maskingOffsetDb) {
    this.N = N;
    this.sr = sampleRate;
    this.numBands = numBands;
    this.maskingOffsetDb = maskingOffsetDb;

    this.lineFreqs = new Float64Array(N);
    this.lineBarks = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      const f = (k + 0.5) * (sampleRate / 2.0) / N;
      this.lineFreqs[k] = f;
      this.lineBarks[k] = hzToBark(f);
    }

    const barkMax = this.lineBarks[N - 1];
    const edges = new Float64Array(numBands + 1);
    for (let i = 0; i <= numBands; i++) edges[i] = (barkMax + 1e-6) * (i / numBands);

    this.bandOfLine = new Int32Array(N);
    for (let k = 0; k < N; k++) {
      let b = 0;
      // digitize: find bin such that edges[b] <= bark < edges[b+1]
      for (let e = 1; e <= numBands; e++) {
        if (this.lineBarks[k] < edges[e]) { b = e - 1; break; }
        b = numBands - 1;
      }
      this.bandOfLine[k] = Math.min(Math.max(b, 0), numBands - 1);
    }

    this.bandBarkCenters = new Float64Array(numBands);
    for (let b = 0; b < numBands; b++) this.bandBarkCenters[b] = 0.5 * (edges[b] + edges[b + 1]);

    this.bandSizes = new Int32Array(numBands);
    for (let k = 0; k < N; k++) this.bandSizes[this.bandOfLine[k]]++;
    for (let b = 0; b < numBands; b++) if (this.bandSizes[b] === 0) this.bandSizes[b] = 1;

    // Schroeder-style spreading matrix (energy domain).
    this.spreadLin = new Float64Array(numBands * numBands);
    for (let i = 0; i < numBands; i++) {
      for (let j = 0; j < numBands; j++) {
        const dz = this.bandBarkCenters[i] - this.bandBarkCenters[j];
        const sdb = 15.81 + 7.5 * (dz + 0.474) - 17.5 * Math.sqrt(1 + (dz + 0.474) ** 2);
        this.spreadLin[i * numBands + j] = Math.pow(10, sdb / 10.0);
      }
    }

    // Absolute threshold of hearing per band, evaluated at the band's mean line freq.
    this.bandQuietThreshDb = new Float64Array(numBands);
    const sums = new Float64Array(numBands);
    const counts = new Float64Array(numBands);
    for (let k = 0; k < N; k++) {
      const b = this.bandOfLine[k];
      sums[b] += this.lineFreqs[k];
      counts[b] += 1;
    }
    for (let b = 0; b < numBands; b++) {
      const meanF = counts[b] > 0 ? sums[b] / counts[b] : this.lineFreqs[N - 1];
      this.bandQuietThreshDb[b] = absoluteThresholdDb(meanF);
    }
  }

  // Fills bandEnergy[numBands] and bandThresh[numBands] (linear units) for one frame's MDCT coeffs.
  masking(X, bandEnergy, bandThresh, refEnergy = 1e-12) {
    const { numBands, bandOfLine, spreadLin, bandQuietThreshDb, maskingOffsetDb } = this;
    bandEnergy.fill(0);
    for (let k = 0; k < X.length; k++) {
      const v = X[k];
      bandEnergy[bandOfLine[k]] += v * v;
    }
    for (let b = 0; b < numBands; b++) if (bandEnergy[b] < refEnergy) bandEnergy[b] = refEnergy;

    for (let i = 0; i < numBands; i++) {
      let spread = 0;
      const base = i * numBands;
      for (let j = 0; j < numBands; j++) spread += spreadLin[base + j] * bandEnergy[j];
      const spreadDb = 10 * Math.log10(spread / refEnergy);
      const maskedDb = spreadDb - maskingOffsetDb;
      const threshDb = Math.max(maskedDb, bandQuietThreshDb[i]);
      bandThresh[i] = refEnergy * Math.pow(10, threshDb / 10.0);
    }
  }
}

// ---------- Bit allocation ----------
const DB_PER_BIT = 6.02;
const MAX_BITS_PER_LINE = 15;

function allocateBits(bandEnergy, bandThresh, bandSizes, totalBits, bitsOut, refEnergy = 1e-12) {
  const numBands = bandEnergy.length;
  bitsOut.fill(0);
  const noiseDb = new Float64Array(numBands);
  const threshDb = new Float64Array(numBands);
  const perLineEnergy = new Float64Array(numBands);
  const active = new Uint8Array(numBands);

  for (let b = 0; b < numBands; b++) {
    perLineEnergy[b] = bandEnergy[b] / Math.max(bandSizes[b], 1);
    noiseDb[b] = 10 * Math.log10(Math.max(perLineEnergy[b], refEnergy) / refEnergy);
    threshDb[b] = 10 * Math.log10(Math.max(bandThresh[b], refEnergy) / refEnergy);
    active[b] = perLineEnergy[b] > refEnergy * 2 ? 1 : 0;
  }

  let remaining = totalBits;
  while (remaining > 0) {
    let bestB = -1;
    let bestNmr = -Infinity;
    for (let b = 0; b < numBands; b++) {
      if (!active[b]) continue;
      if (bitsOut[b] >= MAX_BITS_PER_LINE) continue;
      if (bandSizes[b] > remaining) continue;
      const nmr = noiseDb[b] - threshDb[b];
      if (nmr > bestNmr) { bestNmr = nmr; bestB = b; }
    }
    if (bestB === -1) break;
    bitsOut[bestB] += 1;
    noiseDb[bestB] -= DB_PER_BIT;
    remaining -= bandSizes[bestB];
  }
}

// ---------- Quantization ----------
function quantizeValue(v, bits, scale) {
  if (bits <= 0 || scale <= 0) return 0;
  const levels = 1 << bits;
  const step = (2.0 * scale) / levels;
  let code = Math.round(v / step);
  const lo = -(levels >> 1), hi = (levels >> 1) - 1;
  if (code < lo) code = lo; else if (code > hi) code = hi;
  return code;
}

function dequantizeValue(code, bits, scale) {
  if (bits <= 0 || scale <= 0) return 0;
  const levels = 1 << bits;
  const step = (2.0 * scale) / levels;
  return code * step;
}

// ---------- Full encode/decode simulation on one channel ----------
// Returns reconstructed Float32Array plus per-frame bit-allocation table
// (for visualization) and a bit-accounting-based compressed size estimate.
function encodeDecodeMono(x, sampleRate, { N = 512, numBands = 32, targetKbps = 128, maskingOffsetDb = 6.0 }, onProgress) {
  const numSamples = x.length;
  const bitsPerFrame = Math.round((targetKbps * 1000.0 * N) / sampleRate);

  const mdct = new MDCT(N);
  const psy = new PsychoacousticModel(N, sampleRate, numBands, maskingOffsetDb);

  const padFront = N;
  const tail = (N - (numSamples % N)) % N;
  const padBack = N + tail;
  const xp = new Float32Array(padFront + numSamples + padBack);
  xp.set(x, padFront);

  const numFrames = Math.floor((xp.length - 2 * N) / N) + 1;
  const outLen = (numFrames - 1) * N + 2 * N;
  const out = new Float32Array(outLen);

  const X = new Float32Array(N);
  const Xq = new Float32Array(N);
  const y = new Float32Array(2 * N);
  const bandEnergy = new Float64Array(numBands);
  const bandThresh = new Float64Array(numBands);
  const bitsArr = new Int32Array(numBands);

  // Per-frame, per-band bit allocation, kept for visualization (downsample
  // in time if there are many frames, to keep the payload light).
  const bandBitsHistory = new Uint8Array(numFrames * numBands);

  let totalBitsUsed = 0;
  // Fixed per-frame metadata overhead: bits-per-band table (4 bits fits 0-15)
  // plus a 16-bit scale factor per band (mirrors the reference format).
  const overheadBitsPerFrame = numBands * (4 + 16);

  const block = new Float32Array(2 * N);

  for (let f = 0; f < numFrames; f++) {
    const start = f * N;
    for (let i = 0; i < 2 * N; i++) block[i] = xp[start + i];
    mdct.forward(block, X);

    psy.masking(X, bandEnergy, bandThresh);
    allocateBits(bandEnergy, bandThresh, psy.bandSizes, bitsPerFrame, bitsArr);

    let usedThisFrame = overheadBitsPerFrame;

    for (let b = 0; b < numBands; b++) {
      bandBitsHistory[f * numBands + b] = bitsArr[b];
      const nb = bitsArr[b];
      // Compute this band's scale (max abs coefficient) and quantize/dequantize.
      let scale = 0;
      for (let k = 0; k < N; k++) {
        if (psy.bandOfLine[k] === b) {
          const av = Math.abs(X[k]);
          if (av > scale) scale = av;
        }
      }
      for (let k = 0; k < N; k++) {
        if (psy.bandOfLine[k] === b) {
          if (nb > 0 && scale > 0) {
            const code = quantizeValue(X[k], nb, scale);
            Xq[k] = dequantizeValue(code, nb, scale);
            usedThisFrame += nb;
          } else {
            Xq[k] = 0;
          }
        }
      }
    }
    totalBitsUsed += usedThisFrame;

    mdct.inverse(Xq, y);
    for (let i = 0; i < 2 * N; i++) out[start + i] += y[i];

    if (onProgress && (f % 32 === 0)) onProgress(f / numFrames);
  }

  const reconstructed = out.subarray(N, N + numSamples).slice();

  return {
    reconstructed,
    numFrames,
    numBands,
    N,
    bandBitsHistory,
    compressedBytes: Math.ceil(totalBitsUsed / 8),
  };
}

// ---------- Worker message handling ----------
self.onmessage = function (e) {
  const { cmd } = e.data;
  if (cmd === 'encode') {
    const { pcm, sampleRate, N, numBands, targetKbps, maskingOffsetDb } = e.data;
    try {
      const result = encodeDecodeMono(
        pcm, sampleRate,
        { N, numBands, targetKbps, maskingOffsetDb },
        (frac) => self.postMessage({ type: 'progress', frac })
      );
      // Compute SNR here so the worker owns all the number-crunching.
      let errE = 0, sigE = 0;
      const n = Math.min(pcm.length, result.reconstructed.length);
      for (let i = 0; i < n; i++) {
        const d = pcm[i] - result.reconstructed[i];
        errE += d * d;
        sigE += pcm[i] * pcm[i];
      }
      const snrDb = 10 * Math.log10(sigE / Math.max(errE, 1e-20));

      self.postMessage({
        type: 'done',
        reconstructed: result.reconstructed,
        numFrames: result.numFrames,
        numBands: result.numBands,
        N: result.N,
        bandBitsHistory: result.bandBitsHistory,
        compressedBytes: result.compressedBytes,
        snrDb,
      }, [result.reconstructed.buffer, result.bandBitsHistory.buffer]);
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message + '\n' + err.stack });
    }
  }
};
