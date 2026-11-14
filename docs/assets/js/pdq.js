// PDQ 256비트 이미지 해시 — Meta ThreatExchange pdq/cpp(hashing/pdqhashing.cpp, downscaling/downscaling.cpp)의 순서를 그대로 옮김.
// 회색조 → Jarosz 상자 필터 2회 → 64x64 솎아내기 → 16x16 DCT(직류 제외) → 중앙값 비교, 품질은 64x64 기울기 합.
// 비트 순서는 pdqhash(Python) compute()의 반환 벡터와 같게 맞춘다(실험 21 원장 기록과 비교하기 위해).

function box1D(inv, outv, off, len, stride, win) {
  const half = Math.floor((win + 2) / 2);
  const p1 = half - 1, p2 = win - half + 1, p3 = len - win, p4 = half - 1;
  let li = off, ri = off, oi = off, sum = 0, cur = 0;
  for (let i = 0; i < p1; i++) { sum += inv[ri]; cur++; ri += stride; }
  for (let i = 0; i < p2; i++) { sum += inv[ri]; cur++; outv[oi] = sum / cur; ri += stride; oi += stride; }
  for (let i = 0; i < p3; i++) { sum += inv[ri]; sum -= inv[li]; outv[oi] = sum / cur; li += stride; ri += stride; oi += stride; }
  for (let i = 0; i < p4; i++) { sum -= inv[li]; cur--; outv[oi] = sum / cur; li += stride; oi += stride; }
}

const windowSize = (oldDim, newDim) => Math.floor((oldDim + 2 * newDim - 1) / (2 * newDim));

const DCT = (() => {
  const m = [], s = Math.sqrt(2 / 64);
  for (let i = 0; i < 16; i++) { const r = new Float64Array(64); for (let j = 0; j < 64; j++) r[j] = s * Math.cos((Math.PI / 2 / 64) * (i + 1) * (2 * j + 1)); m.push(r); }
  return m;
})();

// luma: Float32Array(rows*cols) → {bits: Uint8Array(256) (pdqhash 벡터 순서), quality}
export function pdqFromLuma(luma, rows, cols) {
  let a = Float32Array.from(luma), b = new Float32Array(luma.length);
  const wr = windowSize(cols, 64), wc = windowSize(rows, 64);
  for (let rep = 0; rep < 2; rep++) {
    for (let i = 0; i < rows; i++) box1D(a, b, i * cols, cols, 1, wr);   // 행 방향
    for (let j = 0; j < cols; j++) box1D(b, a, j, rows, cols, wc);        // 열 방향
  }
  const B = Array.from({ length: 64 }, () => new Float64Array(64));
  for (let i = 0; i < 64; i++) {
    const ii = Math.floor(((i + 0.5) * rows) / 64);
    for (let j = 0; j < 64; j++) B[i][j] = a[ii * cols + Math.floor(((j + 0.5) * cols) / 64)];
  }
  let g = 0;
  for (let i = 0; i < 63; i++) for (let j = 0; j < 64; j++) g += Math.abs(Math.trunc(((B[i][j] - B[i + 1][j]) * 100) / 255));
  for (let i = 0; i < 64; i++) for (let j = 0; j < 63; j++) g += Math.abs(Math.trunc(((B[i][j] - B[i][j + 1]) * 100) / 255));
  const quality = Math.min(100, Math.floor(g / 90));
  const T = Array.from({ length: 16 }, () => new Float64Array(64));
  for (let i = 0; i < 16; i++) for (let j = 0; j < 64; j++) { let s = 0; for (let k = 0; k < 64; k++) s += DCT[i][k] * B[k][j]; T[i][j] = s; }
  const D = new Float64Array(256);
  for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) { let s = 0; for (let k = 0; k < 64; k++) s += T[i][k] * DCT[j][k]; D[i * 16 + j] = s; }
  const sorted = Float64Array.from(D).sort();
  const med = sorted[127];  // torben 중앙값(짝수 개면 아래쪽 중앙값)
  const bits = new Uint8Array(256);
  for (let k = 0; k < 256; k++) bits[255 - k] = D[k] > med ? 1 : 0;  // pdqhash.hash_to_vector의 [::-1] 순서
  return { bits, quality };
}

export function lumaFromRGBA(data, rows, cols) {
  const y = new Float32Array(rows * cols);
  for (let p = 0, q = 0; q < y.length; p += 4, q++) y[q] = data[p] * 0.299 + data[p + 1] * 0.587 + data[p + 2] * 0.114;
  return y;
}
