// Code Portrait: photo -> animated portrait of . - > < with orange details.
// Everything runs in the browser. MediaPipe finds the face, pose and person mask.

const SET = '.-><';
const ASPECT = 1.7; // cell height / cell width
const COLORS = {
  dark: { bg: '#141413', fg: '#ecebe6', muted: '#8f8e88', accent: '#f76707' },
  light: { bg: '#f4f3ef', fg: '#1a1a1a', muted: '#9a9992', accent: '#e8590c' }
};
const MODEL = {
  face: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  pose: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  seg: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite'
};

const $ = id => document.getElementById(id);
const cv = $('cv'), ctx = cv.getContext('2d');
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const state = { cols: 80, theme: 'dark', cut: true, img: null, det: null, grid: null, feats: [null, null, null], chars: null };

/* ---------- custom characters ---------- */
const DEFAULT_CHARS = '.-><';
function inkOf(ch) {
  const c = document.createElement('canvas'); c.width = c.height = 48;
  const x = c.getContext('2d', { willReadFrequently: true });
  x.font = '36px "JetBrains Mono", ui-monospace, Menlo, monospace'; x.textAlign = 'center'; x.textBaseline = 'middle';
  x.fillText(ch, 24, 24);
  const d = x.getImageData(0, 0, 48, 48).data; let s = 0;
  for (let i = 3; i < d.length; i += 4) s += d[i];
  return s;
}
// Returns sorted characters (light to dense), null for the default set, or an error message.
function parseChars(text) {
  const list = [...new Set(Array.from(text).filter(c => c.trim()))];
  if (list.length < 3) return { error: 'Add at least 3 different characters.' };
  if (list.length > 10) return { error: 'Use 10 characters at most.' };
  if (list.length === 4 && [...DEFAULT_CHARS].every(c => list.includes(c))) return { chars: null };
  return { chars: list.map(c => [c, inkOf(c)]).sort((a, b) => a[1] - b[1]).map(a => a[0]) };
}
// Accent shapes are drawn with . - > <; map those roles onto the custom set.
function mapGlyph(c) {
  const C = state.chars; if (!C) return c; const N = C.length;
  if (c === '.') return C[0];
  if (c === '-') return C[Math.floor((N - 1) / 2)];
  if (c === '<') return N >= 4 ? C[N - 2] : C[N - 1];
  return C[N - 1];
}

/* ---------- detection ---------- */
let models = null;
async function loadModels() {
  if (models) return models;
  if (window.__cpMockDetect) return (models = { mock: true });
  const mp = await import('./mp/vision_bundle.mjs');
  const files = await mp.FilesetResolver.forVisionTasks(new URL('./mp/wasm', location.href).href);
  const make = async (Cls, opts) => {
    try { return await Cls.createFromOptions(files, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'GPU' } }); }
    catch { return await Cls.createFromOptions(files, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'CPU' } }); }
  };
  const [face, pose, seg] = await Promise.all([
    make(mp.FaceLandmarker, { baseOptions: { modelAssetPath: MODEL.face }, runningMode: 'IMAGE', numFaces: 1 }),
    make(mp.PoseLandmarker, { baseOptions: { modelAssetPath: MODEL.pose }, runningMode: 'IMAGE', numPoses: 1 }),
    make(mp.ImageSegmenter, { baseOptions: { modelAssetPath: MODEL.seg }, runningMode: 'IMAGE', outputConfidenceMasks: true, outputCategoryMask: false })
  ]);
  return (models = { face, pose, seg });
}

async function detect(canvas) {
  const m = await loadModels();
  if (m.mock) return window.__cpMockDetect(canvas);
  const W = canvas.width, H = canvas.height;
  const face = m.face.detect(canvas).faceLandmarks?.[0] || null;
  const pose = m.pose.detect(canvas).landmarks?.[0] || null;
  let mask = null;
  const res = m.seg.segment(canvas);
  const masks = res.confidenceMasks || [];
  if (masks.length) {
    const mk = masks[masks.length === 1 ? 0 : 0];
    const raw = mk.getAsFloat32Array(), mw = mk.width, mh = mk.height;
    let data = Float32Array.from(raw);
    // Make sure the mask means "person": corners should be background.
    const corner = (x, y) => data[y * mw + x];
    const c = (corner(0, 0) + corner(mw - 1, 0) + corner(0, mh - 1) + corner(mw - 1, mh - 1)) / 4;
    if (masks.length > 1 || c > 0.5) data = data.map(v => 1 - v);
    mask = { data, w: mw, h: mh };
    masks.forEach(x => x.close && x.close());
  }
  return { face, pose, mask, W, H };
}

/* ---------- photo -> grid ---------- */
function prepImage(img) {
  const max = 1200, s = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement('canvas');
  c.width = Math.round(img.naturalWidth * s); c.height = Math.round(img.naturalHeight * s);
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  return c;
}

function maskAt(det, px, py) {
  if (!det.mask || !state.cut) return 1;
  const m = det.mask, x = Math.min(m.w - 1, Math.max(0, Math.round(px / det.W * m.w))), y = Math.min(m.h - 1, Math.max(0, Math.round(py / det.H * m.h)));
  return m.data[y * m.w + x];
}

function cropBox(det) {
  const { W, H } = det;
  if (!det.mask || !state.cut) return { x: 0, y: 0, w: W, h: H };
  let x0 = W, y0 = H, x1 = 0, y1 = 0;
  const step = Math.max(1, Math.round(Math.max(W, H) / 300));
  for (let y = 0; y < H; y += step) for (let x = 0; x < W; x += step) if (maskAt(det, x, y) > 0.5) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (x1 <= x0 || y1 <= y0) return { x: 0, y: 0, w: W, h: H };
  const px = (x1 - x0) * 0.08, py = (y1 - y0) * 0.06, top = (y1 - y0) * 0.08;
  x0 = Math.max(0, x0 - px); y0 = Math.max(0, y0 - top); x1 = Math.min(W, x1 + px); y1 = Math.min(H, y1 + py);
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

function buildGrid(canvas, det) {
  const COLS = state.cols, box = cropBox(det);
  const ROWS = Math.max(8, Math.round(COLS * box.h / box.w / ASPECT));
  const sc = document.createElement('canvas'); sc.width = COLS; sc.height = ROWS;
  const sx = sc.getContext('2d', { willReadFrequently: true });
  sx.imageSmoothingQuality = 'high';
  sx.drawImage(canvas, box.x, box.y, box.w, box.h, 0, 0, COLS, ROWS);
  const px = sx.getImageData(0, 0, COLS, ROWS).data;
  const n = COLS * ROWS, g = new Float32Array(n), mk = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    g[i] = (0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2]) / 255;
    const cx = i % COLS, cy = (i / COLS) | 0;
    mk[i] = maskAt(det, box.x + (cx + 0.5) / COLS * box.w, box.y + (cy + 0.5) / ROWS * box.h);
  }
  // autocontrast on the person
  const vals = []; for (let i = 0; i < n; i++) if (mk[i] > 0.5) vals.push(g[i]);
  const src = vals.length > 50 ? vals : Array.from(g);
  src.sort((a, b) => a - b);
  const lo = src[Math.floor(src.length * 0.02)], hi = src[Math.min(src.length - 1, Math.floor(src.length * 0.995))];
  for (let i = 0; i < n; i++) g[i] = Math.min(1, Math.max(0, (g[i] - lo) / Math.max(0.05, hi - lo)));
  if (state.theme === 'light') for (let i = 0; i < n; i++) g[i] = 1 - g[i];
  // local contrast: box blur
  const r = Math.max(2, Math.round(COLS * 0.05)), blur = new Float32Array(n);
  for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
    let s = 0, c = 0;
    for (let yy = Math.max(0, y - Math.round(r / ASPECT)); yy <= Math.min(ROWS - 1, y + Math.round(r / ASPECT)); yy++)
      for (let xx = Math.max(0, x - r); xx <= Math.min(COLS - 1, x + r); xx++) { s += g[yy * COLS + xx]; c++; }
    blur[y * COLS + x] = s / c;
  }
  const rows = [], C = state.chars;
  for (let y = 0; y < ROWS; y++) {
    const row = [];
    for (let x = 0; x < COLS; x++) {
      const i = y * COLS + x;
      let v = Math.pow(Math.min(1, Math.max(0, g[i] + (g[i] - blur[i]) * 1.2)), 1.6);
      if (mk[i] < 0.5) v = 0;
      const gx = (x < COLS - 1 ? g[i + 1] : g[i]) - (x > 0 ? g[i - 1] : g[i]);
      if (C) row.push(v < 0.2 ? ' ' : C[Math.min(C.length - 1, Math.floor((v - 0.2) / 0.8 * C.length))]);
      else row.push(v < 0.2 ? ' ' : v < 0.38 ? '.' : v < 0.6 ? '-' : (gx < 0 ? '>' : '<'));
    }
    rows.push(row);
  }
  return { rows, COLS, ROWS, box };
}

/* ---------- orange details ---------- */
const FEATURES = [
  { id: 'sunglasses', label: 'Sunglasses', re: /sun ?glass|shades/, need: 'face' },
  { id: 'glasses', label: 'Glasses', re: /glass|specs|spectacle/, need: 'face' },
  { id: 'watch', label: 'Watch', re: /watch|wrist|bracelet/, need: 'pose' },
  { id: 'heart', label: 'Heart on the chest', re: /heart|srdce|srdíčko|\blove\b/, need: 'body' },
  { id: 'dot', label: 'Dot on the chest', re: /\bdot\b|circle|logo|badge|\bpin\b/, need: 'body' },
  { id: 'hat', label: 'Hat', re: /\bhat\b|\bcap\b|beanie/, need: 'face' },
  { id: 'earrings', label: 'Earrings', re: /earring|\bears?\b/, need: 'face' },
  { id: 'mustache', label: 'Mustache', re: /mo?ustache|tache/, need: 'face' },
  { id: 'bowtie', label: 'Bow tie', re: /bow ?tie|\btie\b/, need: 'face' },
  { id: 'hair', label: 'Hair', re: /\bhair/, need: 'face' },
  { id: 'lips', label: 'Lips', re: /lips?\b|mouth|smile/, need: 'face' },
  { id: 'eyes', label: 'Eyes', re: /\beyes?\b/, need: 'face' },
  { id: 'shirt', label: 'Shirt', re: /shirt|\btee\b|\btop\b|sweater|hoodie|jacket|clothes/, need: 'body' },
  { id: 'face', label: 'Face', re: /\bface\b|skin/, need: 'face' }
];
const parse = text => { const t = text.toLowerCase().trim(); if (!t) return null; return FEATURES.find(f => f.re.test(t)) || { id: null, text: t }; };

function featureCells(f, text, det, grid) {
  const { COLS, ROWS, box, rows } = grid;
  const toG = p => [((p.x * det.W) - box.x) / box.w * COLS - 0.5, ((p.y * det.H) - box.y) / box.h * ROWS - 0.5];
  const F = det.face, P = det.pose;
  const cells = new Map();
  const put = (x, y, c) => { x = Math.round(x); y = Math.round(y); if (x >= 0 && x < COLS && y >= 0 && y < ROWS) cells.set(x + ',' + y, [x, y, mapGlyph(c)]); };
  const keep = (x, y) => { x = Math.round(x); y = Math.round(y); if (x >= 0 && x < COLS && y >= 0 && y < ROWS && rows[y][x] !== ' ') cells.set(x + ',' + y, [x, y, rows[y][x]]); };
  const lensBox = (cx, cy, hw, hh, fill) => {
    cx = Math.round(cx); cy = Math.round(cy);
    for (let x = cx - hw; x <= cx + hw; x++) { const e = x === cx - hw || x === cx + hw; put(x, cy - hh, e ? '.' : '-'); put(x, cy + hh, e ? '.' : '-'); }
    for (let y = cy - hh + 1; y < cy + hh; y++) { put(cx - hw, y, '>'); put(cx + hw, y, '<'); if (fill) for (let x = cx - hw + 1; x < cx + hw; x++) put(x, y, '-'); }
  };
  const disc = (cx, cy, r) => {
    for (let y = Math.floor(cy - r / ASPECT) - 1; y <= cy + r / ASPECT + 1; y++)
      for (let x = Math.floor(cx - r) - 1; x <= cx + r + 1; x++) {
        const d = Math.hypot(x - cx, (y - cy) * ASPECT);
        if (d <= r) put(x, y, d > r - 1.1 ? '.' : ((x + y) % 2 ? '>' : '<'));
      }
  };
  const ellipseKeep = (cx, cy, rx, ry) => { for (let y = Math.floor(cy - ry); y <= cy + ry; y++) for (let x = Math.floor(cx - rx); x <= cx + rx; x++) if (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1) keep(x, y); };
  const L = i => toG(F[i]);
  const vis = i => P && P[i] && (P[i].visibility ?? 1) > 0.5 && P[i].x > 0 && P[i].x < 1 && P[i].y > 0 && P[i].y < 1;

  let fw = 0, fh = 0, top, chin, left, right;
  if (F) { top = L(10); chin = L(152); left = L(234); right = L(454); fw = Math.abs(right[0] - left[0]); fh = Math.abs(chin[1] - top[1]); }
  const shoulders = vis(11) && vis(12) ? [toG(P[11]), toG(P[12])] : null;
  const chest = () => {
    if (shoulders) {
      const [a, b] = shoulders, sw = Math.abs(a[0] - b[0]);
      const hips = vis(23) && vis(24) ? (toG(P[23])[1] + toG(P[24])[1]) / 2 : null;
      const sy = (a[1] + b[1]) / 2;
      return [(a[0] + b[0]) / 2, hips ? sy + (hips - sy) * 0.3 : sy + sw * 0.35 / ASPECT, sw];
    }
    if (F) return [(left[0] + right[0]) / 2, chin[1] + fh * 0.9, fw * 2];
    return null;
  };

  switch (f.id) {
    case 'glasses': case 'sunglasses': {
      const a = toG({ x: (F[33].x + F[133].x) / 2, y: (F[33].y + F[133].y) / 2 }), b = toG({ x: (F[362].x + F[263].x) / 2, y: (F[362].y + F[263].y) / 2 });
      const [l, r] = a[0] < b[0] ? [a, b] : [b, a];
      const d = r[0] - l[0], hw = Math.max(1, Math.round(d * 0.32)), hh = Math.max(1, Math.round(hw / ASPECT * 0.9)), y = Math.round((l[1] + r[1]) / 2);
      lensBox(l[0], y, hw, hh, f.id === 'sunglasses'); lensBox(r[0], y, hw, hh, f.id === 'sunglasses');
      for (let x = Math.round(l[0]) + hw + 1; x < Math.round(r[0]) - hw; x++) put(x, y - (hh > 1 ? 1 : 0), '-');
      const ex = Math.min(left[0], right[0]), fx = Math.max(left[0], right[0]);
      for (let x = Math.round(ex); x < Math.round(l[0]) - hw; x++) put(x, y - (hh > 1 ? 1 : 0), '-');
      for (let x = Math.round(r[0]) + hw + 1; x <= Math.round(fx); x++) put(x, y - (hh > 1 ? 1 : 0), '-');
      break;
    }
    case 'hat': {
      const x0 = Math.min(left[0], right[0]), x1 = Math.max(left[0], right[0]), brim = Math.round(top[1] - fh * 0.08);
      const ext = Math.max(1, Math.round(fw * 0.12));
      for (let x = Math.round(x0) - ext; x <= Math.round(x1) + ext; x++) put(x, brim, '-');
      const ch = Math.max(2, Math.round(fh * 0.45)), cx = (x0 + x1) / 2, hw = Math.max(2, Math.round(fw * 0.42));
      for (let x = Math.round(cx) - hw; x <= Math.round(cx) + hw; x++) put(x, brim - ch, x === Math.round(cx) - hw || x === Math.round(cx) + hw ? '.' : '-');
      for (let y = brim - ch + 1; y < brim; y++) { put(cx - hw, y, '>'); put(cx + hw, y, '<'); }
      break;
    }
    case 'earrings': {
      for (const [lm, out] of [[234, -1], [454, 1]]) {
        const p = L(lm), o = Math.sign((p[0] - (left[0] + right[0]) / 2)) || out;
        const x = p[0] + o * 0.5, y = p[1] + fh * 0.22;
        put(x, y, '.'); put(x, y + 1, '<'); put(x - 1, y + 1, '>'); put(x, y + 2, '.');
      }
      break;
    }
    case 'mustache': {
      const ml = L(61), mr = L(291), y = Math.round((L(2)[1] + L(0)[1]) / 2);
      const x0 = Math.round(Math.min(ml[0], mr[0])), x1 = Math.round(Math.max(ml[0], mr[0]));
      for (let x = x0 + 1; x < x1; x++) put(x, y, '-');
      put(x0, y, '<'); put(x1, y, '>'); put(x0 - 1, y + 1, '.'); put(x1 + 1, y + 1, '.');
      break;
    }
    case 'bowtie': {
      const cx = Math.round(chin[0]), y = Math.round(chin[1] + Math.max(1, fh * 0.28)), hw = Math.max(2, Math.round(fw * 0.2));
      put(cx, y, '.');
      for (let x = cx - hw; x <= cx + hw; x++) if (x !== cx) put(x, y, x === cx - hw ? '>' : x === cx + hw ? '<' : '-');
      for (const dy of [-1, 1]) { put(cx - hw, y + dy, '>'); put(cx + hw, y + dy, '<'); put(cx - hw + 1, y + dy, '-'); put(cx + hw - 1, y + dy, '-'); }
      break;
    }
    case 'watch': {
      const want = /\bright\b/.test(text) ? 16 : /\bleft\b/.test(text) ? 15 : null;
      const pick = want && vis(want) ? want : [15, 16].filter(vis).sort((a, b) => (P[b].visibility ?? 1) - (P[a].visibility ?? 1))[0];
      if (!pick) return { cells: [], note: "Couldn't find a wrist in this photo" };
      const [x, y] = toG(P[pick]);
      const sw = shoulders ? Math.abs(shoulders[0][0] - shoulders[1][0]) : COLS * 0.4;
      const hw = Math.max(2, Math.round(sw * 0.06)), hh = Math.max(1, Math.round(hw / ASPECT));
      lensBox(x, y, hw, hh, false); put(x, y, '.');
      break;
    }
    case 'heart': {
      const c = chest(); if (!c) return { cells: [], note: "Couldn't find a chest in this photo" };
      // which side, as you look at the photo (default: right)
      const side = /\bleft\b|vlevo|levé|leve/.test(text) ? -1 : /\bcenter\b|\bmiddle\b|uprostřed/.test(text) ? 0 : 1;
      const HEARTS = [
        ['## ##', '#####', ' ### ', '  #  '],
        [' ## ## ', '#######', ' ##### ', '  ###  ', '   #   '],
        [' ###   ### ', '##### #####', '###########', ' ######### ', '  #######  ', '   #####   ', '    ###    ', '     #     ']
      ];
      const want = c[2] * 0.14, shape = want < 7 ? HEARTS[0] : want < 10 ? HEARTS[1] : HEARTS[2];
      const hx = Math.round(c[0] + side * c[2] * 0.22), hy = Math.round(c[1] - 1), w = shape[0].length;
      if (hy >= ROWS - 1) return { cells: [], note: 'The chest is out of frame' };
      shape.forEach((row, dy) => [...row].forEach((m, dx) => {
        if (m !== '#') return;
        const x = hx - (w >> 1) + dx, y = hy - 1 + dy;
        if (x < 0 || x >= COLS || y < 0 || y >= ROWS) return;
        cells.set(x + ',' + y, [x, y, rows[y][x] !== ' ' ? rows[y][x] : mapGlyph('.')]);
      }));
      break;
    }
    case 'dot': {
      const c = chest(); if (!c) return { cells: [], note: "Couldn't find a chest in this photo" };
      if (c[1] >= ROWS - 1) return { cells: [], note: 'The chest is out of frame' };
      disc(c[0], c[1], Math.max(1.6, c[2] * 0.08));
      break;
    }
    case 'hair': {
      const x0 = Math.min(left[0], right[0]) - fw * 0.15, x1 = Math.max(left[0], right[0]) + fw * 0.15, yb = top[1] + fh * 0.12;
      for (let y = 0; y <= yb; y++) for (let x = Math.floor(x0); x <= x1; x++) keep(x, y);
      break;
    }
    case 'lips': { const a = L(61), b = L(291), u = L(0), d = L(17); ellipseKeep((a[0] + b[0]) / 2, (u[1] + d[1]) / 2, Math.abs(b[0] - a[0]) / 2 + 0.6, Math.abs(d[1] - u[1]) / 2 + 0.6); break; }
    case 'eyes': {
      for (const [i, j] of [[33, 133], [362, 263]]) { const a = L(i), b = L(j); ellipseKeep((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, Math.abs(b[0] - a[0]) / 2 + 0.8, 1.2); }
      break;
    }
    case 'face': ellipseKeep((left[0] + right[0]) / 2, (top[1] + chin[1]) / 2, fw / 2, fh / 2); break;
    case 'shirt': {
      const y0 = shoulders ? Math.min(shoulders[0][1], shoulders[1][1]) - 1 : (F ? chin[1] + fh * 0.35 : ROWS * 0.6);
      const ex = F ? [Math.min(left[0], right[0]), Math.max(left[0], right[0]), chin[1]] : null;
      for (let y = Math.max(0, Math.floor(y0)); y < ROWS; y++) for (let x = 0; x < COLS; x++) {
        if (ex && y <= ex[2] + 1 && x >= ex[0] && x <= ex[1]) continue;
        keep(x, y);
      }
      break;
    }
  }
  return { cells: [...cells.values()] };
}

function resolveFeatures() {
  const det = state.det, grid = state.grid, out = [];
  for (let i = 0; i < 3; i++) {
    const text = $('f' + i).value, chip = $('c' + i), f = parse(text);
    chip.className = 'chip'; chip.textContent = '';
    if (!f) continue;
    if (!f.id) { chip.className = 'chip no'; chip.textContent = "Not sure what that is. Try one of the ideas below."; continue; }
    if (f.need === 'face' && !det.face) { chip.className = 'chip no'; chip.textContent = `${f.label}: couldn't find a face`; continue; }
    if (f.need === 'pose' && !det.pose) { chip.className = 'chip no'; chip.textContent = `${f.label}: couldn't find a body`; continue; }
    if (f.need === 'body' && !det.pose && !det.face) { chip.className = 'chip no'; chip.textContent = `${f.label}: couldn't find a person`; continue; }
    const r = featureCells(f, text.toLowerCase(), det, grid);
    if (!r.cells.length) { chip.className = 'chip no'; chip.textContent = `${f.label}: ${r.note || 'nothing to draw here'}`; continue; }
    chip.className = 'chip ok'; chip.textContent = `${f.label} ✓`;
    out.push(...r.cells);
  }
  return out;
}

/* ---------- animation ---------- */
let W = 0, H = 0, cw = 0, chh = 0, base = [], acc = [], raf = 0;
const rnd = (a, b) => a + Math.random() * (b - a);
const ease = k => 1 - Math.pow(1 - k, 3);
const glyph = () => { const S = state.chars || SET; return S[(Math.random() * S.length) | 0]; };

function layout() {
  if (!state.grid) return;
  const dpr = window.devicePixelRatio || 1, { COLS, ROWS } = state.grid;
  W = cv.clientWidth; cw = W / COLS; chh = cw * ASPECT; H = chh * ROWS;
  cv.style.height = H + 'px'; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
}
const font = (size, bold) => (bold ? '700 ' : '') + size + 'px "JetBrains Mono", ui-monospace, Menlo, monospace';
function somewhere() { const { COLS, ROWS } = state.grid, a = rnd(0, Math.PI * 2), r = rnd(0.15, 0.9); return [COLS / 2 + Math.cos(a) * COLS * r, ROWS / 2 + Math.sin(a) * ROWS * r]; }

function playBase(brk) {
  const now = performance.now(), { rows, ROWS } = state.grid, lead = brk ? 550 : 0, old = new Map(base.map(p => [p.t.x + ',' + p.t.y, p]));
  base = [];
  rows.forEach((row, y) => [...row].forEach((c, x) => {
    if (c === ' ') return;
    const [sx, sy] = somewhere();
    base.push({ t: { x, y, ch: c }, sx, sy, brk: brk && old.has(x + ',' + y) ? now : null, start: now + lead + (y / ROWS) * 900 + rnd(0, 600), dur: rnd(1100, 1800), g: glyph() });
  }));
  playAccents(cellsCache, lead + 2300);
}
let cellsCache = [];
function playAccents(cells, delay = 0) {
  const now = performance.now();
  acc.forEach(p => { if (p.dir === 'in') { p.dir = 'out'; p.from = [p.t.x, p.t.y]; [p.ex, p.ey] = somewhere(); p.start = now + rnd(0, 150); p.dur = rnd(500, 800); } });
  cells.forEach(([x, y, c]) => { const [sx, sy] = somewhere(); acc.push({ t: { x, y, ch: c }, dir: 'in', sx, sy, start: now + delay + rnd(0, 400), dur: rnd(700, 1100), g: glyph() }); });
  kick();
}
function draw(now) {
  const col = COLORS[state.theme];
  ctx.clearRect(0, 0, W, H);
  let busy = false;
  const hidden = new Set(acc.filter(p => p.dir === 'in' && (reduce || now - p.start >= p.dur)).map(p => p.t.x + ',' + p.t.y));
  ctx.font = font(cw * 1.6, false);
  for (const p of base) {
    const k = reduce ? 1 : Math.min(1, Math.max(0, (now - p.start) / p.dur));
    if (k < 1) busy = true;
    if (k >= 1 && hidden.has(p.t.x + ',' + p.t.y)) continue;
    if (k <= 0 && p.brk !== null && !reduce) {
      const b = ease(Math.min(1, (now - p.brk) / 500));
      if (Math.random() < 0.12) p.g = glyph();
      ctx.globalAlpha = 1 - 0.8 * b; ctx.fillStyle = b < 0.3 ? col.fg : col.accent;
      ctx.fillText(b < 0.2 ? p.t.ch : p.g, (p.t.x + (p.sx - p.t.x) * b + 0.5) * cw, (p.t.y + (p.sy - p.t.y) * b + 0.5) * chh);
      continue;
    }
    if (k <= 0) continue;
    const e = ease(k);
    if (k < 0.9 && Math.random() < 0.12) p.g = glyph();
    ctx.globalAlpha = 0.2 + 0.8 * e;
    ctx.fillStyle = k >= 1 ? col.fg : (p.t.ch === '.' ? col.muted : col.accent);
    ctx.fillText(k < 0.9 ? p.g : p.t.ch, (p.sx + (p.t.x - p.sx) * e + 0.5) * cw, (p.sy + (p.t.y - p.sy) * e + 0.5) * chh);
  }
  ctx.font = font(cw * 1.6, true); ctx.fillStyle = col.accent;
  acc = acc.filter(p => !(p.dir === 'out' && now - p.start > p.dur));
  for (const p of acc) {
    const k = reduce ? 1 : Math.min(1, Math.max(0, (now - p.start) / p.dur));
    if (k < 1) busy = true;
    if (k <= 0 && p.dir === 'in') continue;
    const e = ease(k); let x, y, a, g = p.t.ch;
    if (p.dir === 'in') { x = p.sx + (p.t.x - p.sx) * e; y = p.sy + (p.t.y - p.sy) * e; a = 0.2 + 0.8 * e; if (k < 0.85) { if (Math.random() < 0.15) p.g = glyph(); g = p.g; } }
    else { x = p.from[0] + (p.ex - p.from[0]) * e; y = p.from[1] + (p.ey - p.from[1]) * e; a = 1 - e; }
    ctx.globalAlpha = a; ctx.fillText(g, (x + 0.5) * cw, (y + 0.5) * chh);
  }
  ctx.globalAlpha = 1;
  raf = busy ? requestAnimationFrame(draw) : 0;
}
function kick() { if (!raf && state.grid) raf = requestAnimationFrame(draw); }

/* ---------- final frame for export ---------- */
function finalCells() {
  const { rows } = state.grid, orange = new Map(cellsCache.map(c => [c[0] + ',' + c[1], c[2]])), out = [];
  rows.forEach((row, y) => [...row].forEach((c, x) => { const k = x + ',' + y; if (orange.has(k)) return; if (c !== ' ') out.push([x, y, c, false]); }));
  orange.forEach((c, k) => { const [x, y] = k.split(',').map(Number); out.push([x, y, c, true]); });
  return out;
}
// Story-sized (1080x1920) image for Instagram, shared through the phone's share sheet.
function storyBlob() {
  const col = COLORS[state.theme], W = 1080, H = 1920, cells = finalCells();
  // fit the drawn area, not the empty margins around it
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [cx, cy] of cells) { x0 = Math.min(x0, cx); x1 = Math.max(x1, cx); y0 = Math.min(y0, cy); y1 = Math.max(y1, cy); }
  const COLS = x1 - x0 + 1, ROWS = y1 - y0 + 1;
  const unit = Math.min(940 / COLS, 1250 / (ROWS * ASPECT));
  const w = COLS * unit, h = ROWS * unit * ASPECT, ox = (W - w) / 2 - x0 * unit, oy = 300 + (1250 - h) / 2 - y0 * unit * ASPECT;
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const x = c.getContext('2d'); x.fillStyle = col.bg; x.fillRect(0, 0, W, H);
  x.textAlign = 'center'; x.textBaseline = 'middle';
  for (const [cx, cy, ch, o] of cells) { x.font = font(unit * 1.6, o); x.fillStyle = o ? col.accent : col.fg; x.fillText(ch, ox + (cx + 0.5) * unit, oy + (cy + 0.5) * unit * ASPECT); }
  x.fillStyle = col.accent; x.beginPath(); x.arc(W / 2, 1660, 9, 0, Math.PI * 2); x.fill();
  x.fillStyle = col.muted; x.font = '500 30px Inter, -apple-system, "Helvetica Neue", Arial, sans-serif';
  x.fillText('code-portrait.sevcovicp.workers.dev', W / 2, 1712);
  return new Promise(r => c.toBlob(r, 'image/png'));
}
async function shareInstagram() {
  try {
    const blob = await storyBlob(), file = new File([blob], 'code-portrait.png', { type: 'image/png' });
    if (navigator.canShare && navigator.canShare({ files: [file] })) await navigator.share({ files: [file] });
    else { save(blob, 'code-portrait-story.png'); toast('Saved. Add it to your Instagram story from your photos.'); }
  } catch (e) { if (e.name !== 'AbortError') toast("Couldn't open sharing. Try Download PNG instead."); }
}
function setupShare() {
  try {
    const probe = new File([new Blob(['x'], { type: 'image/png' })], 'p.png', { type: 'image/png' });
    const can = !!(navigator.canShare && navigator.canShare({ files: [probe] }));
    const phone = matchMedia('(pointer: coarse)').matches;
    if (can && phone) { $('share').hidden = false; $('sharehint').hidden = false; $('dl').classList.remove('primary'); }
  } catch {}
}

/* ---------- MP4 video of the animation ---------- */
function storyLayout(cells, W, H) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [cx, cy] of cells) { x0 = Math.min(x0, cx); x1 = Math.max(x1, cx); y0 = Math.min(y0, cy); y1 = Math.max(y1, cy); }
  const COLS = x1 - x0 + 1, ROWS = y1 - y0 + 1;
  const unit = Math.min(940 / COLS, 1250 / (ROWS * ASPECT));
  const w = COLS * unit, h = ROWS * unit * ASPECT;
  return { unit, ox: (W - w) / 2 - x0 * unit, oy: 300 + (1250 - h) / 2 - y0 * unit * ASPECT, x0, x1, y0, y1 };
}
function seeded(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

// Builds a deterministic version of the on-screen animation and returns drawFrame(ctx, ms).
function videoScene(W, H) {
  const col = COLORS[state.theme], rows = state.grid.rows, S = state.chars || SET;
  const cells = finalCells(), L = storyLayout(cells, W, H), R = seeded(7);
  const spanX = L.x1 - L.x0 + 1, spanY = L.y1 - L.y0 + 1, cx0 = (L.x0 + L.x1) / 2, cy0 = (L.y0 + L.y1) / 2;
  const from = () => { const a = R() * Math.PI * 2, r = 0.15 + R() * 0.75; return [cx0 + Math.cos(a) * spanX * r, cy0 + Math.sin(a) * spanY * r]; };
  const base = [], acc = [];
  rows.forEach((row, y) => [...row].forEach((c, x) => { if (c === ' ') return; const [sx, sy] = from(); base.push({ x, y, c, sx, sy, start: 200 + ((y - L.y0) / spanY) * 900 + R() * 600, dur: 1100 + R() * 700, seed: (R() * 1e6) | 0 }); }));
  cellsCache.forEach(([x, y, c]) => { const [sx, sy] = from(); acc.push({ x, y, c, sx, sy, start: 2600 + R() * 400, dur: 700 + R() * 400, seed: (R() * 1e6) | 0 }); });
  const end = Math.max(...base.map(p => p.start + p.dur), ...acc.map(p => p.start + p.dur), 0);
  const ease = k => 1 - Math.pow(1 - k, 3);
  const scr = (p, t) => S[(p.seed + Math.floor(t / 90)) % S.length];
  const px = (x, y) => [L.ox + (x + 0.5) * L.unit, L.oy + (y + 0.5) * L.unit * ASPECT];
  function draw(x, t) {
    x.globalAlpha = 1; x.fillStyle = col.bg; x.fillRect(0, 0, W, H);
    x.textAlign = 'center'; x.textBaseline = 'middle';
    const hidden = new Set(acc.filter(p => t >= p.start + p.dur).map(p => p.x + ',' + p.y));
    x.font = font(L.unit * 1.6, false);
    for (const p of base) {
      const k = Math.min(1, Math.max(0, (t - p.start) / p.dur)); if (k <= 0) continue;
      if (k >= 1 && hidden.has(p.x + ',' + p.y)) continue;
      const e = ease(k), [X, Y] = px(p.sx + (p.x - p.sx) * e, p.sy + (p.y - p.sy) * e);
      x.globalAlpha = 0.2 + 0.8 * e; x.fillStyle = k >= 1 ? col.fg : (p.c === '.' ? col.muted : col.accent);
      x.fillText(k < 0.9 ? scr(p, t) : p.c, X, Y);
    }
    x.font = font(L.unit * 1.6, true); x.fillStyle = col.accent;
    for (const p of acc) {
      const k = Math.min(1, Math.max(0, (t - p.start) / p.dur)); if (k <= 0) continue;
      const e = ease(k), [X, Y] = px(p.sx + (p.x - p.sx) * e, p.sy + (p.y - p.sy) * e);
      x.globalAlpha = 0.2 + 0.8 * e; x.fillText(k < 0.85 ? scr(p, t) : p.c, X, Y);
    }
    x.globalAlpha = 1; x.fillStyle = col.accent; x.beginPath(); x.arc(W / 2, 1660, 9, 0, Math.PI * 2); x.fill();
    x.fillStyle = col.muted; x.font = '500 30px Inter, -apple-system, "Helvetica Neue", Arial, sans-serif';
    x.fillText('code-portrait.sevcovicp.workers.dev', W / 2, 1712);
  }
  return { draw, duration: end + 1800 };
}

async function renderVideo(onProgress) {
  const W = 1080, H = 1920, FPS = 30, scene = videoScene(W, H), frames = Math.ceil(scene.duration / 1000 * FPS);
  const c = document.createElement('canvas'); c.width = W; c.height = H; const x = c.getContext('2d');
  // Preferred: WebCodecs + MP4 muxer (renders frame by frame, works offline, real .mp4)
  if ('VideoEncoder' in window) {
    let config = null;
    for (const codec of ['avc1.640028', 'avc1.4d0028', 'avc1.42003e']) {
      const cfg = { codec, width: W, height: H, bitrate: 8_000_000, framerate: FPS };
      try { if ((await VideoEncoder.isConfigSupported(cfg)).supported) { config = cfg; break; } } catch {}
    }
    if (config) {
      const { Muxer, ArrayBufferTarget } = await import('./lib/mp4-muxer.mjs');
      const muxer = new Muxer({ target: new ArrayBufferTarget(), video: { codec: 'avc', width: W, height: H, frameRate: FPS }, fastStart: 'in-memory' });
      let failed = null;
      const enc = new VideoEncoder({ output: (chunk, meta) => muxer.addVideoChunk(chunk, meta), error: e => { failed = e; } });
      enc.configure(config);
      for (let i = 0; i < frames; i++) {
        if (failed) throw failed;
        scene.draw(x, i * 1000 / FPS);
        const vf = new VideoFrame(c, { timestamp: Math.round(i * 1e6 / FPS), duration: Math.round(1e6 / FPS) });
        enc.encode(vf, { keyFrame: i % (FPS * 2) === 0 }); vf.close();
        if (enc.encodeQueueSize > 8) await new Promise(r => setTimeout(r, 0));
        if (i % 10 === 0) { onProgress(i / frames); await new Promise(r => setTimeout(r, 0)); }
      }
      await enc.flush(); muxer.finalize();
      return { blob: new Blob([muxer.target.buffer], { type: 'video/mp4' }), ext: 'mp4' };
    }
  }
  // Fallback: record the canvas in real time
  const mime = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'].find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m));
  if (!mime) throw new Error('no-video');
  const stream = c.captureStream(FPS), rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8_000_000 }), parts = [];
  rec.ondataavailable = e => e.data.size && parts.push(e.data);
  const done = new Promise(r => (rec.onstop = r));
  scene.draw(x, 0); rec.start();
  const t0 = performance.now();
  await new Promise(resolve => { const tick = () => { const t = performance.now() - t0; scene.draw(x, t); onProgress(Math.min(1, t / scene.duration)); if (t < scene.duration) requestAnimationFrame(tick); else resolve(); }; requestAnimationFrame(tick); });
  rec.stop(); await done;
  return { blob: new Blob(parts, { type: mime.split(';')[0] }), ext: mime.includes('mp4') ? 'mp4' : 'webm' };
}

let rendering = false;
// Sharing needs a fresh tap. If rendering took too long, the browser refuses,
// so the button turns into "Tap to share" and the next tap shares the file.
let pending = null;
async function shareOrSave(file, btn, label) {
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share({ files: [file] }); return; }
    catch (e) {
      if (e.name === 'AbortError') return;
      if (e.name === 'NotAllowedError') { pending = { file, btn, label, target: btn.getAttribute('popovertarget') }; btn.removeAttribute('popovertarget'); btn.textContent = 'Tap to share'; return 'pending'; }
    }
  }
  save(file, file.name);
}
document.addEventListener('click', async e => {
  if (!pending || !e.target.closest('button') || e.target.closest('button') !== pending.btn) return;
  e.preventDefault(); e.stopPropagation();
  const p = pending; pending = null;
  p.btn.innerHTML = p.label; if (p.target) setTimeout(() => p.btn.setAttribute('popovertarget', p.target), 0);
  try { await navigator.share({ files: [p.file] }); } catch (err) { if (err.name !== 'AbortError') save(p.file, p.file.name); }
}, true);

async function exportVideo(btn = $('dl'), forShare = !$('share').hidden) {
  if (rendering || !state.grid) return;
  rendering = true; const label = btn.innerHTML; btn.disabled = true; let kept = false;
  try {
    const { blob, ext } = await renderVideo(p => { btn.textContent = `Rendering ${Math.round(p * 100)}%`; });
    const file = new File([blob], 'code-portrait.' + ext, { type: blob.type });
    btn.innerHTML = label; btn.disabled = false;
    if (forShare) kept = (await shareOrSave(file, btn, label)) === 'pending';
    else save(blob, file.name);
    if (ext !== 'mp4') toast('Your browser can only make a .webm video. Chrome or Safari make .mp4.');
  } catch (e) { console.error(e); toast("Couldn't make the video in this browser. Try Chrome or Safari."); }
  finally { if (!kept) btn.innerHTML = label; btn.disabled = false; rendering = false; }
}

async function copyText() {
  const { rows } = state.grid, grid = rows.map(r => [...r]);
  cellsCache.forEach(([x, y, c]) => { grid[y][x] = c; });
  const text = grid.map(r => r.join('').replace(/\s+$/, '')).join('\n').replace(/^\n+|\n+$/g, '');
  try { await navigator.clipboard.writeText(text); toast('Copied. Best pasted somewhere with a monospace font, like a code editor.'); }
  catch { toast("Couldn't copy. Your browser blocked clipboard access."); }
}

function exportPNG() {
  const { COLS, ROWS } = state.grid, col = COLORS[state.theme], unit = 12, pad = unit * 3, scale = 2;
  const w = COLS * unit + pad * 2, h = ROWS * unit * ASPECT + pad * 2;
  const c = document.createElement('canvas'); c.width = w * scale; c.height = h * scale;
  const x = c.getContext('2d'); x.scale(scale, scale); x.fillStyle = col.bg; x.fillRect(0, 0, w, h);
  x.textAlign = 'center'; x.textBaseline = 'middle';
  for (const [cx, cy, ch, o] of finalCells()) { x.font = font(unit * 1.6, o); x.fillStyle = o ? col.accent : col.fg; x.fillText(ch, pad + (cx + 0.5) * unit, pad + (cy + 0.5) * unit * ASPECT); }
  c.toBlob(b => save(b, 'code-portrait.png'), 'image/png');
}
function exportSVG() {
  const { COLS, ROWS } = state.grid, col = COLORS[state.theme], unit = 12, pad = unit * 3;
  const w = COLS * unit + pad * 2, h = Math.round(ROWS * unit * ASPECT + pad * 2);
  const esc = c => c.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const cells = finalCells(), t = o => cells.filter(c => c[3] === o).map(([cx, cy, ch]) => `<text x="${(pad + (cx + 0.5) * unit).toFixed(1)}" y="${(pad + (cy + 0.5) * unit * ASPECT).toFixed(1)}">${esc(ch)}</text>`).join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}"><rect width="100%" height="100%" fill="${col.bg}"/><g font-family="JetBrains Mono, Menlo, Consolas, monospace" font-size="${unit * 1.6}" text-anchor="middle" dominant-baseline="central"><g fill="${col.fg}">${t(false)}</g><g fill="${col.accent}" font-weight="700">${t(true)}</g></g></svg>`;
  save(new Blob([svg], { type: 'image/svg+xml' }), 'code-portrait.svg');
}
// Messaging apps use proportional fonts, so . - > < drift out of line there.
// Braille dot characters all share one width (even the blank one), so a braille
// version keeps its shape in iMessage, WhatsApp and Messenger.
function brailleText(widthChars = 20) {
  const det = state.det, box = cropBox(det), DW = widthChars * 2;
  const DH = Math.max(8, Math.round(DW * box.h / box.w / 4) * 4);
  const sc = document.createElement('canvas'); sc.width = DW; sc.height = DH;
  const sx = sc.getContext('2d', { willReadFrequently: true });
  sx.imageSmoothingQuality = 'high';
  sx.drawImage(state.img, box.x, box.y, box.w, box.h, 0, 0, DW, DH);
  const px = sx.getImageData(0, 0, DW, DH).data, n = DW * DH, g = new Float32Array(n), mk = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    g[i] = (0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2]) / 255;
    mk[i] = maskAt(det, box.x + ((i % DW) + 0.5) / DW * box.w, box.y + (((i / DW) | 0) + 0.5) / DH * box.h);
  }
  const vals = []; for (let i = 0; i < n; i++) if (mk[i] > 0.5) vals.push(g[i]);
  const src = (vals.length > 50 ? vals : Array.from(g)).sort((a, b) => a - b);
  const lo = src[Math.floor(src.length * 0.02)], hi = src[Math.min(src.length - 1, Math.floor(src.length * 0.995))];
  const BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
  const on = (x, y) => {
    const i = y * DW + x; if (mk[i] < 0.5) return false;
    const v = Math.min(1, Math.max(0, (g[i] - lo) / Math.max(0.05, hi - lo)));
    return v > 0.2 + 0.6 * (BAYER[y % 4][x % 4] + 0.5) / 16;
  };
  const BITS = [[0, 0, 0x01], [0, 1, 0x02], [0, 2, 0x04], [1, 0, 0x08], [1, 1, 0x10], [1, 2, 0x20], [0, 3, 0x40], [1, 3, 0x80]];
  const lines = [];
  for (let y = 0; y < DH; y += 4) {
    let line = '';
    for (let x = 0; x < DW; x += 2) { let b = 0; for (const [dx, dy, bit] of BITS) if (on(x + dx, y + dy)) b |= bit; line += String.fromCharCode(0x2800 + b); }
    lines.push(line);
  }
  while (lines.length && /^⠀+$/.test(lines[0])) lines.shift();
  while (lines.length && /^⠀+$/.test(lines[lines.length - 1])) lines.pop();
  return lines.join('\n');
}
async function copyForMessages() {
  try { await navigator.clipboard.writeText(brailleText(20)); toast('Copied. Paste it into any chat, it keeps its shape.'); }
  catch { toast("Couldn't copy. Your browser blocked clipboard access."); }
}

function save(blob, name) { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); toast(`Saved ${name}`); }
let tt; function toast(m) { $('toast').textContent = m; clearTimeout(tt); tt = setTimeout(() => ($('toast').textContent = ''), 4000); }

/* ---------- example portrait ---------- */
function exampleDet(W, H) {
  const S = W / 1145, n = (x, y, v = 0.95) => ({ x: x * S / W, y: y * S / H, z: 0, visibility: v });
  const face = Array.from({ length: 478 }, () => n(570, 420));
  Object.entries({ 10: [590, 240], 152: [560, 560], 234: [462, 400], 454: [712, 390], 33: [475, 372], 133: [530, 372], 362: [575, 370], 263: [625, 368], 61: [510, 470], 291: [600, 468], 0: [555, 455], 2: [555, 435], 17: [555, 500] }).forEach(([i, [x, y]]) => (face[i] = n(x, y)));
  const pose = Array.from({ length: 33 }, () => n(0.5, 0.5, 0));
  Object.entries({ 11: [790, 700, 0.95], 12: [330, 690, 0.95], 15: [300, 925, 0.9], 16: [840, 980, 0.3], 23: [700, 1300, 0.1], 24: [400, 1300, 0.1] }).forEach(([i, [x, y, v]]) => (pose[i] = n(x, y, v)));
  const mw = 256, mh = 256, data = new Float32Array(mw * mh);
  for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) {
    const X = x / mw * 1145, Y = y / mh * 1145;
    data[y * mw + x] = (((X - 590) / 150) ** 2 + ((Y - 400) / 210) ** 2 <= 1) || (Y > 590 && X > 170 && X < 900) ? 1 : 0;
  }
  return { face, pose, mask: { data, w: mw, h: mh }, W, H };
}
function loadExample() {
  const img = new Image();
  img.onload = () => { state.img = prepImage(img); state.det = exampleDet(state.img.width, state.img.height); state.example = true; $('backwrap').hidden = true; $('caption').textContent = 'Example: Petr Ševcovic. Click the portrait to replay.'; rebuild(); };
  img.src = 'petr.jpg';
}

/* ---------- flow ---------- */
function status(msg) { const s = $('status'); s.textContent = msg || ''; s.classList.toggle('on', !!msg); }
function rebuild() {
  if (!state.det) return;
  state.grid = buildGrid(state.img, state.det);
  $('stage').classList.toggle('light', state.theme === 'light');
  layout();
  cellsCache = resolveFeatures();
  acc = [];
  playBase(false);
}
async function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file), img = new Image();
  img.onload = async () => {
    URL.revokeObjectURL(url);
    state.img = prepImage(img);
    state.example = false; $('backwrap').hidden = false; $('caption').textContent = 'Click the portrait to replay.';
    state.grid = null; base = []; acc = []; ctx.clearRect(0, 0, cv.width, cv.height);
    cv.style.height = '320px';
    status(models ? 'Finding you in the photo…' : 'Loading face detection (first time only)…');
    try { state.det = await detect(state.img); }
    catch (e) { console.error(e); state.det = { face: null, pose: null, mask: null, W: state.img.width, H: state.img.height }; toast("Face detection didn't load, so orange details are off."); }
    status('');
    rebuild();
  };
  img.src = url;
}

$('file').addEventListener('change', e => handleFile(e.target.files[0]));
$('pick').addEventListener('click', () => $('file').click());
$('back').addEventListener('click', () => { $('file').value = ''; loadExample(); });
let dragN = 0;
addEventListener('dragenter', e => { if ([...(e.dataTransfer?.types || [])].includes('Files')) { e.preventDefault(); dragN++; document.body.classList.add('over'); } });
addEventListener('dragover', e => { if (dragN) e.preventDefault(); });
addEventListener('dragleave', () => { if (dragN && --dragN === 0) document.body.classList.remove('over'); });
addEventListener('drop', e => { if (!dragN) return; e.preventDefault(); dragN = 0; document.body.classList.remove('over'); handleFile(e.dataTransfer.files[0]); });
addEventListener('paste', e => { const f = [...(e.clipboardData?.files || [])].find(f => f.type.startsWith('image/')); if (f) handleFile(f); });
$('stage').addEventListener('click', () => { if (state.grid) playBase(true); });

let ft;
for (let i = 0; i < 3; i++) $('f' + i).addEventListener('input', () => {
  clearTimeout(ft);
  ft = setTimeout(() => { if (!state.grid) return; cellsCache = resolveFeatures(); playAccents(cellsCache); }, 450);
});
document.querySelectorAll('.ideas button').forEach(b => b.addEventListener('click', () => {
  const empty = [0, 1, 2].map(i => $('f' + i)).find(f => !f.value.trim()) || $('f2');
  empty.value = b.textContent; empty.dispatchEvent(new Event('input'));
}));
document.querySelectorAll('.seg').forEach(seg => seg.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  seg.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b));
  const k = seg.dataset.opt; state[k] = k === 'cols' ? +b.dataset.v : k === 'cut' ? b.dataset.v === '1' : b.dataset.v;
  rebuild();
}));
function setupMenu(menu, btn) {
  const place = () => {
    const r = btn.getBoundingClientRect(), mh = menu.offsetHeight || 150, below = r.bottom + 6 + mh < innerHeight;
    menu.style.left = Math.max(12, Math.min(r.left, innerWidth - menu.offsetWidth - 12)) + 'px';
    menu.style.top = (below ? r.bottom + 6 : r.top - 6 - mh) + 'px';
  };
  const close = () => { if (menu.hidePopover) menu.hidePopover(); else menu.style.display = 'none'; };
  if (menu.showPopover) menu.addEventListener('toggle', e => { if (e.newState === 'open') { place(); menu.querySelector('button').focus(); } });
  else { btn.addEventListener('click', () => { menu.style.display = menu.style.display === 'block' ? 'none' : 'block'; place(); }); menu.style.display = 'none'; }
  menu.addEventListener('keydown', e => {
    const items = [...menu.querySelectorAll('button')], i = items.indexOf(document.activeElement);
    if (e.key === 'ArrowDown') { e.preventDefault(); items[(i + 1) % items.length].focus(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
  });
  menu.querySelectorAll('[data-dl]').forEach(b => b.addEventListener('click', () => {
    close();
    if (!state.grid) return;
    ({ png: exportPNG, svg: exportSVG, mp4: () => exportVideo(), text: copyText, msg: copyForMessages, shareimg: shareInstagram, sharevid: () => exportVideo($('share'), true) })[b.dataset.dl]();
  }));
}
setupMenu($('dlmenu'), $('dl'));
setupMenu($('cpmenu'), $('cp'));
setupMenu($('shmenu'), $('share'));
setupShare();
let ct;
$('chars').addEventListener('input', () => {
  clearTimeout(ct);
  ct = setTimeout(() => {
    const r = parseChars($('chars').value), msg = $('charsmsg');
    if (r.error) { msg.className = 'chip no'; msg.textContent = r.error; return; }
    state.chars = r.chars;
    msg.className = 'chip'; msg.textContent = r.chars ? 'Light to dense: ' + r.chars.join(' ') : '';
    rebuild();
  }, 500);
});
$('charsreset').addEventListener('click', () => { $('chars').value = '. - > <'; $('chars').dispatchEvent(new Event('input')); });
let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { layout(); kick(); }, 120); });
loadExample();
