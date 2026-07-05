/* Perceptual Audio Codec — UI layer (main thread)
 * Talks to js/codec-worker.js for all the actual DSP. This file only:
 *  - accepts a dropped/picked audio file and decodes it with the Web Audio API
 *  - trims to a preview window and hands the mono PCM to the worker
 *  - renders results: A/B players, waveform overlay, bit-allocation heatmap
 */

const PREVIEW_SECONDS = 15;

const css = getComputedStyle(document.documentElement);
const COLOR_AMBER = css.getPropertyValue('--amber').trim() || '#F2A93B';
const COLOR_TEAL = css.getPropertyValue('--teal').trim() || '#3A5A63';
const COLOR_CYAN = css.getPropertyValue('--cyan').trim() || '#6FE7DD';
const COLOR_ERROR = css.getPropertyValue('--error').trim() || '#E85D75';
const COLOR_BG = '#0D1215';

const dropzone = document.getElementById('dropzone');
const dropzoneContent = document.getElementById('dropzoneContent');
const dropzoneFile = document.getElementById('dropzoneFile');
const fileInput = document.getElementById('fileInput');
const fileNameEl = document.getElementById('fileName');
const fileChangeBtn = document.getElementById('fileChangeBtn');

const controls = document.getElementById('controls');
const kbpsSlider = document.getElementById('kbpsSlider');
const kbpsValue = document.getElementById('kbpsValue');
const marginSlider = document.getElementById('marginSlider');
const marginValue = document.getElementById('marginValue');
const encodeBtn = document.getElementById('encodeBtn');
const encodeBtnLabel = document.getElementById('encodeBtnLabel');

const progressWrap = document.getElementById('progressWrap');
const progressFill = document.getElementById('progressFill');
const progressLabel = document.getElementById('progressLabel');

const results = document.getElementById('results');
const statRatio = document.getElementById('statRatio');
const statSnr = document.getElementById('statSnr');
const statSize = document.getElementById('statSize');
const statTime = document.getElementById('statTime');

const audioOriginal = document.getElementById('audioOriginal');
const audioDecoded = document.getElementById('audioDecoded');
const downloadLink = document.getElementById('downloadLink');

const waveCanvas = document.getElementById('waveCanvas');
const errorCanvas = document.getElementById('errorCanvas');
const errorAmpLabel = document.getElementById('errorAmpLabel');
const heatCanvas = document.getElementById('heatCanvas');

let audioCtx = null;
let currentPcm = null;      // Float32Array, mono, trimmed to preview window
let currentSampleRate = 44100;
let currentFile = null;

function getAudioContext() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

// ---------- file intake ----------

function showFile(file) {
  currentFile = file;
  dropzoneContent.hidden = true;
  dropzoneFile.hidden = false;
  fileNameEl.textContent = file.name;
  controls.hidden = false;
  results.hidden = true;
  progressWrap.hidden = true;

  audioOriginal.src = URL.createObjectURL(file);

  decodeFile(file).catch((err) => {
    alert('Could not decode that audio file: ' + err.message);
  });
}

async function decodeFile(file) {
  const ctx = getAudioContext();
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));

  const sr = audioBuffer.sampleRate;
  const maxSamples = Math.min(audioBuffer.length, Math.floor(PREVIEW_SECONDS * sr));

  // Downmix to mono by averaging channels.
  const mono = new Float32Array(maxSamples);
  const nCh = audioBuffer.numberOfChannels;
  for (let c = 0; c < nCh; c++) {
    const chData = audioBuffer.getChannelData(c);
    for (let i = 0; i < maxSamples; i++) mono[i] += chData[i] / nCh;
  }

  currentPcm = mono;
  currentSampleRate = sr;
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) showFile(fileInput.files[0]);
});
fileChangeBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.value = '';
  fileInput.click();
});

['dragenter', 'dragover'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
});
['dragleave', 'drop'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
});
dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) showFile(file);
});

// ---------- controls ----------

kbpsSlider.addEventListener('input', () => {
  kbpsValue.textContent = `${kbpsSlider.value} kbps`;
});
marginSlider.addEventListener('input', () => {
  marginValue.textContent = `${parseFloat(marginSlider.value).toFixed(1)} dB`;
});

// ---------- encode ----------

encodeBtn.addEventListener('click', runEncode);

function runEncode() {
  if (!currentPcm) return;

  encodeBtn.disabled = true;
  encodeBtnLabel.textContent = 'running…';
  progressWrap.hidden = false;
  results.hidden = true;
  progressFill.style.width = '0%';
  progressLabel.textContent = 'transforming…';

  const kbps = parseFloat(kbpsSlider.value);
  const marginDb = parseFloat(marginSlider.value);
  const N = 512;
  const numBands = 32;

  const worker = new Worker('js/codec-worker.js');
  const t0 = performance.now();

  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'progress') {
      progressFill.style.width = `${Math.round(msg.frac * 100)}%`;
    } else if (msg.type === 'done') {
      const dt = (performance.now() - t0) / 1000;
      onEncodeDone(msg, dt);
      worker.terminate();
    } else if (msg.type === 'error') {
      alert('Encoding failed: ' + msg.message);
      resetEncodeButton();
      worker.terminate();
    }
  };

  // Copy the PCM so the worker can take ownership of the buffer (transfer).
  const pcmCopy = currentPcm.slice();
  worker.postMessage({
    cmd: 'encode',
    pcm: pcmCopy,
    sampleRate: currentSampleRate,
    N, numBands,
    targetKbps: kbps,
    maskingOffsetDb: marginDb,
  }, [pcmCopy.buffer]);
}

function resetEncodeButton() {
  encodeBtn.disabled = false;
  encodeBtnLabel.textContent = 'run the codec';
  progressWrap.hidden = true;
}

function onEncodeDone(msg, encodeTimeSec) {
  resetEncodeButton();
  results.hidden = false;

  const originalBytes = currentPcm.length * 4; // rough: float32 PCM baseline for ratio context
  const wavBytesOriginal = 44 + currentPcm.length * 2; // as 16-bit PCM wav, for an honest apples-to-apples ratio
  const ratio = wavBytesOriginal / msg.compressedBytes;

  statRatio.textContent = `${ratio.toFixed(2)}x`;
  statSnr.textContent = `${msg.snrDb.toFixed(1)} dB`;
  statSize.textContent = formatBytes(msg.compressedBytes);
  statTime.textContent = `${encodeTimeSec.toFixed(2)}s`;

  // Decoded audio player + download link
  const wavBlob = floatToWavBlob(msg.reconstructed, currentSampleRate);
  const url = URL.createObjectURL(wavBlob);
  audioDecoded.src = url;
  downloadLink.href = url;

  drawWaveform(currentPcm, msg.reconstructed);
  drawErrorWave(currentPcm, msg.reconstructed);
  drawHeatmap(msg.bandBitsHistory, msg.numFrames, msg.numBands);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// ---------- WAV encoding (16-bit PCM, for playback + download) ----------

function floatToWavBlob(float32, sampleRate) {
  const numSamples = float32.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

// ---------- canvas rendering ----------

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width || canvas.clientWidth || 600;
  const h = 160;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, w, h, dpr, pixelW: canvas.width, pixelH: canvas.height };
}

function drawWaveform(original, decoded) {
  const { ctx, w, h } = setupCanvas(waveCanvas);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = COLOR_BG;
  ctx.fillRect(0, 0, w, h);

  const n = Math.min(original.length, decoded.length);
  const mid = h / 2;

  function plot(data, color) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x++) {
      const idx = Math.floor(x * n / w);
      const step = Math.max(1, Math.floor(n / w));
      let min = 1, max = -1;
      for (let k = idx; k < Math.min(idx + step, n); k++) {
        if (data[k] < min) min = data[k];
        if (data[k] > max) max = data[k];
      }
      if (min > max) { min = 0; max = 0; }
      const y1 = mid - max * mid * 0.92;
      const y2 = mid - min * mid * 0.92;
      ctx.moveTo(x, y1);
      ctx.lineTo(x, y2);
    }
    ctx.stroke();
  }

  // Draw original normally, then switch to a 'lighten' blend for the decoded
  // trace: where the two agree (the common case at reasonable bitrates)
  // per-channel max blends cyan+amber into a warm cream. Wherever they
  // genuinely diverge, a pure cyan or pure amber sliver shows through —
  // painting amber over cyan at high opacity (the old approach) hid this
  // entirely since amber simply covered cyan pixel-for-pixel.
  ctx.globalCompositeOperation = 'source-over';
  plot(original, COLOR_CYAN);
  ctx.globalCompositeOperation = 'lighten';
  plot(decoded, COLOR_AMBER);
  ctx.globalCompositeOperation = 'source-over';
}

function drawErrorWave(original, decoded) {
  const { ctx, w, h } = setupCanvas(errorCanvas);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = COLOR_BG;
  ctx.fillRect(0, 0, w, h);

  const n = Math.min(original.length, decoded.length);
  const err = new Float32Array(n);
  let maxAbs = 1e-9;
  for (let i = 0; i < n; i++) {
    err[i] = original[i] - decoded[i];
    const a = Math.abs(err[i]);
    if (a > maxAbs) maxAbs = a;
  }

  // Auto-scale so the loudest error sample uses ~85% of the plot height —
  // this is exactly why the difference was invisible before: at a working
  // bitrate the raw error is a small fraction of the signal's own scale.
  const amp = 0.85 / maxAbs;
  errorAmpLabel.textContent = `difference, amplified ${amp.toFixed(1)}×`;

  const mid = h / 2;
  const step = Math.max(1, Math.floor(n / w));
  ctx.beginPath();
  ctx.strokeStyle = COLOR_ERROR;
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x++) {
    const idx = Math.floor(x * n / w);
    let min = 1, max = -1;
    for (let k = idx; k < Math.min(idx + step, n); k++) {
      const v = err[k] * amp;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (min > max) { min = 0; max = 0; }
    const y1 = mid - Math.max(-1, Math.min(1, max)) * mid * 0.92;
    const y2 = mid - Math.max(-1, Math.min(1, min)) * mid * 0.92;
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y2);
  }
  ctx.stroke();

  // Zero line for reference.
  ctx.strokeStyle = 'rgba(232,230,223,0.15)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(w, mid);
  ctx.stroke();
}

function hexToRgb(hex) {
  hex = hex.replace('#', '');
  if (hex.length === 3) hex = hex.split('').map((c) => c + c).join('');
  const num = parseInt(hex, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

function drawHeatmap(bandBitsHistory, numFrames, numBands) {
  const { ctx, w, h, pixelW, pixelH } = setupCanvas(heatCanvas);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = COLOR_BG;
  ctx.fillRect(0, 0, w, h);

  const teal = hexToRgb(COLOR_TEAL);
  const amber = hexToRgb(COLOR_AMBER);
  const maxBits = 15;

  // Build the image at the canvas's actual physical pixel resolution (not
  // the CSS size) since putImageData writes raw pixels and ignores the
  // ctx.scale() transform used elsewhere for retina-quality strokes.
  const imgW = Math.max(1, pixelW);
  const imgH = Math.max(1, pixelH);
  const img = ctx.createImageData(imgW, imgH);

  for (let py = 0; py < imgH; py++) {
    // flip vertically: band 0 (low freq) at bottom
    const bandF = numBands - 1 - (py / imgH) * numBands;
    const b = Math.min(numBands - 1, Math.max(0, Math.floor(bandF)));
    for (let px = 0; px < imgW; px++) {
      const f = Math.min(numFrames - 1, Math.floor((px / imgW) * numFrames));
      const bits = bandBitsHistory[f * numBands + b];
      const t = Math.min(1, bits / maxBits);
      const r = teal[0] + (amber[0] - teal[0]) * t;
      const g = teal[1] + (amber[1] - teal[1]) * t;
      const bl = teal[2] + (amber[2] - teal[2]) * t;
      const idx = (py * imgW + px) * 4;
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = bl;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

window.addEventListener('resize', () => {
  // Re-render at new size if we have results showing (cheap enough to just skip debouncing here).
});
