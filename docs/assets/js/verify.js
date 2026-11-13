// 브라우저 안에서 C2PA 증명서를 읽는다. 파일은 서버로 보내지 않는다.
import { createC2pa, Reader, Context } from "https://cdn.jsdelivr.net/npm/@contentauth/c2pa-web@0.17.0/+esm";

const WASM = "https://cdn.jsdelivr.net/npm/@contentauth/c2pa-web@0.17.0/dist/resources/c2pa_bg.wasm";
const LAB_ROOT = new URL("../trust/lab-root.pem", import.meta.url).href;

const result = document.getElementById("result");
const fileInput = document.getElementById("file");
const drop = fileInput.closest(".drop");
const trustLab = document.getElementById("trustLab");
let c2paPromise = null;
let lastBlob = null;
let lastName = "";

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function c2pa() {
  c2paPromise ??= createC2pa({ wasmSrc: WASM });
  return c2paPromise;
}

async function sha256(blob) {
  const buf = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// lab-phash: tools/fphash.py 와 같은 규격. 회색조 → 32x32 구역 평균 → DCT-II → 8x8 → 중앙값 비교
const N = 32;
const PHASH_T = 15; // 실험 이미지 76장(2,850쌍)에서 쌍당 오탐 1% 이하가 되는 가장 큰 거리. 개발일지 참고
const TILE_T = 5; // 조각 125개와 비교할 때: 질의 하나를 원장 전체와 비교해 오탐 1% 이하가 되는 거리(실험 8). 비교 횟수가 많아 훨씬 엄격해야 함
async function labPhash(blob) {
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const cv = new OffscreenCanvas(bmp.width, bmp.height);
  const cx = cv.getContext("2d", { willReadFrequently: true });
  cx.drawImage(bmp, 0, 0);
  const { data, width: w, height: h } = cx.getImageData(0, 0, bmp.width, bmp.height);
  const s = Array.from({ length: N }, () => new Float64Array(N));
  for (let i = 0; i < N; i++) {
    const r0 = Math.floor((i * h) / N), r1 = Math.max(Math.floor(((i + 1) * h) / N), r0 + 1);
    for (let j = 0; j < N; j++) {
      const c0 = Math.floor((j * w) / N), c1 = Math.max(Math.floor(((j + 1) * w) / N), c0 + 1);
      let sum = 0;
      for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) {
        const k = (r * w + c) * 4;
        sum += (299 * data[k] + 587 * data[k + 1] + 114 * data[k + 2]) / 1000;
      }
      s[i][j] = sum / ((r1 - r0) * (c1 - c0));
    }
  }
  const D = Array.from({ length: N }, (_, k) => Array.from({ length: N }, (_, n) => Math.cos((Math.PI * (2 * n + 1) * k) / (2 * N))));
  const coef = [];
  for (let u = 0; u < 8; u++) for (let v = 0; v < 8; v++) {
    let acc = 0;
    for (let i = 0; i < N; i++) { let row = 0; for (let j = 0; j < N; j++) row += s[i][j] * D[v][j]; acc += D[u][i] * row; }
    coef.push(acc);
  }
  const sorted = [...coef].sort((a, b) => a - b);
  const med = (sorted[31] + sorted[32]) / 2;
  let bits = 0n;
  for (const c of coef) bits = (bits << 1n) | (c > med ? 1n : 0n);
  return bits.toString(16).padStart(16, "0");
}
const hamming = (a, b) => { let x = BigInt("0x" + a) ^ BigInt("0x" + b), n = 0; while (x) { n += Number(x & 1n); x >>= 1n; } return n; };

let ledgerPromise = null;
function loadLedger() {
  ledgerPromise ??= fetch(new URL("../ledger.jsonl", import.meta.url)).then((r) => r.text())
    .then((t) => t.split("\n").filter(Boolean).map((l) => JSON.parse(l)));
  return ledgerPromise;
}

async function ledgerSection(hash, blob) {
  const [entries, ph] = await Promise.all([loadLedger(), labPhash(blob)]);
  const exact = entries.find((e) => e.record.sha256 && e.record.sha256 === hash);
  const near = entries.filter((e) => e.record.lab_phash).map((e) => ({ e, d: hamming(ph, e.record.lab_phash) })).sort((a, b) => a.d - b.d)[0];
  // 전체 해시로 못 찾으면 발행 때 등록한 조각 해시와 비교(잘라 쓴 그림 찾기, 실험 5)
  let tile = null;
  if (!exact && !(near && near.d <= PHASH_T)) {
    for (const e of entries.filter((x) => x.record.type === "tiles")) {
      for (const t of e.record.tiles) {
        const d = hamming(ph, t);
        if (!tile || d < tile.d) tile = { e, d };
      }
    }
  }
  let body;
  if (exact) {
    body = `<p class="state ok"><span class="dot"></span>원장에 같은 파일이 있습니다 (SHA-256 일치, 기록 #${exact.index})</p>`;
  } else if (near && near.d <= PHASH_T) {
    body = `<p class="state warn"><span class="dot"></span>파일은 다르지만 내용이 같은 그림을 원장에서 찾았습니다 (지각 해시 거리 ${near.d} ≤ ${PHASH_T})</p>
      <p>원래 발행본: <a href="${esc(near.e.record.file)}">${esc(near.e.record.caption)}</a><br>
      <span class="muted small">원본의 증명서 ${esc(near.e.record.c2pa_manifest)}</span></p>
      <p class="muted small">이 파일 자체가 인증된 것은 아닙니다. 내용이 같아 보이는 원본의 증명서를 찾아 온 것이라 신뢰 수준이 낮습니다(C2PA 2.5 규격 16.4.1.10).</p>`;
  } else if (tile && tile.d <= TILE_T) {
    const main = entries.find((e) => e.record.sha256 === tile.e.record.of_sha256);
    body = `<p class="state warn"><span class="dot"></span>그림 전체와는 다르지만 발행된 그림의 일부(조각)와 일치합니다 (거리 ${tile.d} ≤ ${TILE_T}) — 잘라서 쓴 것으로 보입니다</p>
      <p>원래 발행본: <a href="${esc(tile.e.record.file)}">${esc(main ? main.record.caption : tile.e.record.file)}</a><br>
      <span class="muted small">원본의 증명서 ${esc(main ? main.record.c2pa_manifest : "-")}</span></p>
      <p class="muted small">이 파일 자체가 인증된 것은 아닙니다. 일부가 같아 보이는 원본의 증명서를 찾아 온 것이라 신뢰 수준이 낮습니다.</p>`;
  } else {
    body = `<p class="state bad"><span class="dot"></span>원장에서 찾지 못했습니다</p>
      <p class="muted small">가장 가까운 기록과의 지각 해시 거리 ${near ? near.d : "-"} (문턱 ${PHASH_T})</p>`;
  }
  return `<h2 class="sub">원장 조회</h2>${body}<dl><dt>지각 해시</dt><dd class="mono">${ph}</dd></dl>`;
}

function context() {
  return trustLab.checked
    ? new Context({ verify: { verifyTrust: true }, trust: { trustAnchors: LAB_ROOT } })
    : new Context({ verify: { verifyTrust: true } });
}

function verdict(store) {
  const codes = (store.validation_status || []).map((s) => s.code);
  const state = store.validation_state;
  if (state === "Invalid" || codes.some((c) => c.includes("mismatch") || c.includes("invalid")))
    return ["bad", "서명 검증 실패 — 서명 뒤에 파일이 바뀌었거나 증명서가 손상됨"];
  if (state === "Trusted") return ["ok", "신뢰된 서명자 — 신뢰 목록에 있는 인증서로 서명됨"];
  if (codes.includes("signingCredential.untrusted"))
    return ["warn", "서명은 유효하지만 신뢰 목록에 없는 서명자"];
  return ["ok", `서명 유효 (${esc(state)})`];
}

async function show(blob, name) {
  lastBlob = blob;
  lastName = name;
  const url = URL.createObjectURL(blob);
  result.innerHTML = `<img class="thumb" src="${url}" alt=""><p class="muted">읽는 중…</p>`;
  const hash = await sha256(blob);
  let reader;
  try {
    reader = await Reader.fromBlob(await c2pa(), blob.type || "image/png", blob, context());
  } catch (e) {
    reader = null;
  }
  if (!reader) {
    result.innerHTML = `<img class="thumb" src="${url}" alt="">
      <p class="state warn"><span class="dot"></span>C2PA 증명서가 없습니다</p>
      <p>증명서가 없다는 것은 "가짜"라는 뜻이 아니라 출처를 이 파일만으로는 알 수 없다는 뜻입니다. 다시 저장하거나 SNS를 거치면 증명서가 지워지는 경우가 많습니다.</p>
      <dl><dt>파일</dt><dd>${esc(name)}</dd><dt>SHA-256</dt><dd class="mono">${hash}</dd></dl>`;
    result.insertAdjacentHTML("beforeend", await ledgerSection(hash, blob));
    return;
  }
  const store = await reader.manifestStore();
  await reader.free();
  const m = store.manifests[store.active_manifest];
  const sig = m.signature_info || {};
  const actions = (m.assertions || []).find((a) => a.label.startsWith("c2pa.actions"));
  const acts = actions ? actions.data.actions.map((a) => a.action.replace("c2pa.", "")).join(" → ") : "";
  const [cls, text] = verdict(store);
  const codes = (store.validation_status || []).map((s) => `<li class="mono">${esc(s.code)}</li>`).join("");
  result.innerHTML = `<img class="thumb" src="${url}" alt="">
    <p class="state ${cls}"><span class="dot"></span>${text}</p>
    <dl>
      <dt>제목</dt><dd>${esc(m.title)}</dd>
      <dt>서명자</dt><dd>${esc(sig.common_name)}<br><span class="muted">${esc(sig.issuer)}</span></dd>
      <dt>서명 시각</dt><dd class="mono">${esc(sig.time)}</dd>
      <dt>기록된 행동</dt><dd>${esc(acts)}</dd>
      <dt>도구</dt><dd>${esc((m.claim_generator_info || []).map((g) => `${g.name} ${g.version || ""}`).join(", "))}</dd>
      <dt>SHA-256</dt><dd class="mono">${hash}</dd>
      <dt>검증 코드</dt><dd><ul class="plain">${codes || "<li>없음</li>"}</ul></dd>
    </dl>
    <p class="muted small">서명은 누가 서명했는지와 서명 뒤 바뀌지 않았다는 것만 보증합니다. 그림 내용이 사실이라는 보증은 아닙니다.</p>`;
  result.insertAdjacentHTML("beforeend", await ledgerSection(hash, blob));
}

fileInput.addEventListener("change", () => fileInput.files[0] && show(fileInput.files[0], fileInput.files[0].name));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const f = e.dataTransfer.files[0];
  if (f) show(f, f.name);
});
document.getElementById("useSample").addEventListener("click", async () => {
  const path = document.getElementById("sample").value;
  const res = await fetch(path);
  show(await res.blob(), path.split("/").pop());
});
trustLab.addEventListener("change", () => lastBlob && show(lastBlob, lastName));
