// 음원 fingerprint — tools/afp.py와 같은 규격(Philips 방식, Haitsma & Kalker, ISMIR 2002).
// 5kHz 신호, 프레임 2048·간격 64, 300~2000Hz 로그 간격 33대역, 대역·프레임 에너지 차이의 부호로 프레임마다 32비트.
export const FS = 5000, FRAME = 2048, HOP = 64, NB = 33, BER_T = 0.35;

const LOG2 = Math.log2(FRAME);
const WIN = Float64Array.from({ length: FRAME }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (FRAME - 1))); // numpy.hanning과 같은 식
const BANDS = (() => {
  const lo = 300, hi = 2000, edges = [];
  for (let b = 0; b <= NB; b++) edges.push(lo * Math.pow(hi / lo, b / NB));
  const out = [];
  for (let b = 0; b < NB; b++) {
    const idx = [];
    for (let k = 0; k <= FRAME / 2; k++) { const f = (k * FS) / FRAME; if (f >= edges[b] && f < edges[b + 1]) idx.push(k); }
    out.push(idx);
  }
  return out;
})();

function fftPower(re) {
  // 제자리 기수-2 FFT, 결과는 0..FRAME/2 의 |X|^2
  const n = FRAME, im = new Float64Array(n);
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
  const p = new Float64Array(n / 2 + 1);
  for (let k = 0; k <= n / 2; k++) p[k] = re[k] * re[k] + im[k] * im[k];
  return p;
}

// y: 5kHz 모노 Float32Array/Float64Array → Uint8Array 배열(프레임마다 32비트)
export function fingerprint(y) {
  const nF = 1 + Math.floor((y.length - FRAME) / HOP);
  if (nF < 2) return [];
  let prev = null;
  const out = [];
  const buf = new Float64Array(FRAME);
  for (let f = 0; f < nF; f++) {
    for (let i = 0; i < FRAME; i++) buf[i] = y[f * HOP + i] * WIN[i];
    const p = fftPower(buf);
    const E = BANDS.map((idx) => idx.reduce((s, k) => s + p[k], 0));
    const d = new Float64Array(NB - 1);
    for (let m = 0; m < NB - 1; m++) d[m] = E[m] - E[m + 1];
    if (prev) {
      const bits = new Uint8Array(32);
      for (let m = 0; m < 32; m++) bits[m] = d[m] - prev[m] > 0 ? 1 : 0;
      out.push(bits);
    }
    prev = d;
  }
  return out;
}

export function ber(a, b, maxShift = 8) {
  let best = 1;
  for (let s = -maxShift; s <= maxShift; s++) {
    const x = s >= 0 ? a.slice(s) : a, y = s >= 0 ? b : b.slice(-s);
    const m = Math.min(x.length, y.length);
    if (m < 64) continue;
    let diff = 0;
    for (let i = 0; i < m; i++) for (let k = 0; k < 32; k++) diff += x[i][k] !== y[i][k];
    best = Math.min(best, diff / (32 * m));
  }
  return best;
}

// 원장 기록(afp_philips 16진수, afp_frames) → 비트 배열
export function fromLedger(hex, frames) {
  const out = [];
  for (let i = 0; i < frames; i++) {
    const bits = new Uint8Array(32);
    for (let byte = 0; byte < 4; byte++) {
      const v = parseInt(hex.substr((i * 4 + byte) * 2, 2), 16);
      for (let k = 0; k < 8; k++) bits[byte * 8 + k] = (v >> (7 - k)) & 1;
    }
    out.push(bits);
  }
  return out;
}

// 브라우저: 파일 → 5kHz 모노 신호
export async function decodeTo5k(blob) {
  const ab = await blob.arrayBuffer();
  const tmp = new AudioContext();
  const audio = await tmp.decodeAudioData(ab);
  await tmp.close();
  // 5,000Hz OfflineAudioContext는 Web Audio 규격의 필수 지원 범위(8,000~96,000Hz) 밖이라 브라우저에 따라 거절될 수 있다(조사 15 지적 8).
  // 8,000Hz로 풀고(저역 통과 포함) JS 선형 보간으로 5kHz로 내린다 — 대역이 300~2,000Hz라 보간 영향은 BER 0.003 수준(11-15 Node 시험).
  const R = 8000;
  const len = Math.ceil(audio.duration * R);
  const off = new OfflineAudioContext(1, len, R);
  const src = off.createBufferSource();
  src.buffer = audio;
  src.connect(off.destination);
  src.start();
  const r = await off.startRendering();
  return stretch(r.getChannelData(0), R / FS);  // 8kHz → 5kHz (f = 1.6배 간격으로 표본 추출)
}

// 속도 되돌려 보기(실험 25의 브라우저판): 5kHz 신호를 선형 보간으로 f배 늘이거나 줄여(속도·음 높이 함께) 지문을 만든다.
// 속도만 바뀐 질의는 음 높이가 최대 5% 어긋난 채 남지만, 대역 폭이 5.9%라 판정에는 견딘다(실험 24).
export const TEMPO_FACTORS = [0.95, 0.96, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03, 1.04, 1.05];
export function stretch(y, f) {
  const n = Math.floor(y.length / f), out = new Float64Array(n);
  for (let i = 0; i < n; i++) { const x = i * f, j = Math.floor(x), a = x - j; out[i] = (y[j] ?? 0) * (1 - a) + (y[j + 1] ?? y[j] ?? 0) * a; }
  return out;
}
