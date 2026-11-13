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
const PHASH_T = 5; // 원장의 서로 다른 그림 14장 × 쌍 오탐 0.07% ≤ 질의 오탐 1% (실험 9). 원장이 커지면 다시 낮춰야 함
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
  let body, detail;
  if (exact) {
    body = `<p class="state ok"><span class="dot"></span>연구실 원장에 같은 파일이 기록되어 있습니다</p>`;
    detail = `<dt>근거</dt><dd>SHA-256 일치, 기록 #${exact.index}</dd>`;
  } else if (near && near.d <= PHASH_T) {
    body = `<p class="state warn"><span class="dot"></span>이 파일 자체는 인증되지 않았습니다. 다만 발행된 그림과 닮아 보여, 그 원본의 증명서를 참고로 보여 드립니다</p>
      <p>닮아 보이는 원본: <a href="${esc(near.e.record.file)}">${esc(near.e.record.caption)}</a></p>
      <p class="muted small">닮음은 기계가 판단한 것이라 틀릴 수 있습니다. 규격은 이렇게 찾아 온 증명서를 낮은 신뢰로 따로 표시하라고 합니다(C2PA 2.5 규격 16.4.1.10).</p>`;
    detail = `<dt>근거</dt><dd>지각 해시 거리 ${near.d} (문턱 ${PHASH_T} 이하면 닮음으로 봄)</dd><dt>원본의 증명서</dt><dd class="mono">${esc(near.e.record.c2pa_manifest)}</dd>`;
  } else if (tile && tile.d <= TILE_T) {
    const main = entries.find((e) => e.record.sha256 === tile.e.record.of_sha256);
    body = `<p class="state warn"><span class="dot"></span>이 파일 자체는 인증되지 않았습니다. 다만 발행된 그림의 일부와 닮아 보여(잘라 쓴 것으로 보임), 그 원본의 증명서를 참고로 보여 드립니다</p>
      <p>닮아 보이는 원본: <a href="${esc(tile.e.record.file)}">${esc(main ? main.record.caption : tile.e.record.file)}</a></p>
      <p class="muted small">닮음은 기계가 판단한 것이라 틀릴 수 있습니다. 찾아 온 증명서는 낮은 신뢰입니다.</p>`;
    detail = `<dt>근거</dt><dd>조각 해시 거리 ${tile.d} (문턱 ${TILE_T} 이하면 닮음으로 봄)</dd><dt>원본의 증명서</dt><dd class="mono">${esc(main ? main.record.c2pa_manifest : "-")}</dd>`;
  } else {
    body = `<p class="state none"><span class="dot"></span>이 연구실 원장에는 없는 그림입니다</p>
      <p class="muted small">다른 곳에서 만든 그림일 수 있으며, 가짜라는 뜻은 아닙니다.</p>`;
    detail = `<dt>근거</dt><dd>가장 가까운 기록과의 지각 해시 거리 ${near ? near.d : "-"} (문턱 ${PHASH_T})</dd>`;
  }
  return `<h2 class="sub">원장 조회</h2>${body}<details><summary>자세히</summary><dl>${detail}<dt>이 파일의 지각 해시</dt><dd class="mono">${ph}</dd></dl></details>`;
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
    return ["bad", "서명 뒤에 파일이 바뀌었습니다. 증명서는 바뀌기 전의 파일을 보증한 것이라 지금 이 파일에는 적용되지 않습니다. 어디가 바뀌었는지는 이 검증기로는 알 수 없습니다"];
  if (state === "Trusted")
    return ["ok", `서명자 확인됨. 서명한 뒤 파일이 바뀌지 않았고, 서명자가 이 검증기의 신뢰 목록에 있습니다${trustLab.checked ? "(연구실 자체 목록 — C2PA 공식 목록 아님)" : ""}. 내용이 사실이라는 뜻은 아닙니다`];
  if (codes.includes("signingCredential.untrusted"))
    return ["warn", "서명은 손상 없이 유효합니다. 다만 서명한 사람(기관)이 이 검증기가 아는 명단에 없어 신원은 확인하지 못했습니다. 명단에 없다는 것이 가짜라는 뜻은 아닙니다"];
  return ["ok", "서명은 손상 없이 유효합니다. 내용이 사실이라는 뜻은 아닙니다"];
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
      <p class="state none"><span class="dot"></span>출처 정보가 없습니다 (C2PA 증명서 없음)</p>
      <p>가짜라는 뜻이 아니라, 이 파일만으로는 누가 만들었는지 알 수 없다는 뜻입니다. 다시 저장하거나 메신저·SNS를 거치면 증명서가 지워지는 경우가 많습니다.</p>
      <dl><dt>파일</dt><dd>${esc(name)}</dd><dt>SHA-256</dt><dd class="mono">${hash}</dd></dl>`;
    result.insertAdjacentHTML("beforeend", await ledgerSection(hash, blob));
    return;
  }
  const store = await reader.manifestStore();
  await reader.free();
  const m = store.manifests[store.active_manifest];
  const sig = m.signature_info || {};
  const actions = (m.assertions || []).find((a) => a.label.startsWith("c2pa.actions"));
  const ACT = { created: "만듦", opened: "엶", edited: "편집", published: "발행", converted: "형식 변환", resized: "크기 변경", cropped: "자름" };
  const acts = actions ? actions.data.actions.map((a) => { const k = a.action.replace("c2pa.", ""); return ACT[k] || k; }).join(" → ") : "";
  const [cls, text] = verdict(store);
  const codes = (store.validation_status || []).map((s) => `<li class="mono">${esc(s.code)}</li>`).join("");
  result.innerHTML = `<img class="thumb" src="${url}" alt="">
    <p class="state ${cls}"><span class="dot"></span>${text}</p>
    ${cls === "bad" ? `<p class="muted small">아래 정보는 유효하지 않은 증명서에서 읽은 것이므로 서명자의 것으로 볼 수 없습니다(C2PA 규격 15.3).</p>` : ""}
    <dl>
      <dt>서명자</dt><dd>${esc(sig.common_name)}<br><span class="muted">${esc(sig.issuer)}</span></dd>
      <dt>서명 시각</dt><dd class="mono">${esc(sig.time)}</dd>
      <dt>서명자가 적은 작업 기록</dt><dd>${esc(acts)}</dd>
    </dl>
    <details><summary>자세히</summary><dl>
      <dt>제목</dt><dd>${esc(m.title)}</dd>
      <dt>서명 도구</dt><dd>${esc((m.claim_generator_info || []).map((g) => `${g.name} ${g.version || ""}`).join(", "))}</dd>
      <dt>검증 상태</dt><dd class="mono">${esc(store.validation_state)}</dd>
      <dt>SHA-256</dt><dd class="mono">${hash}</dd>
      <dt>검증 코드</dt><dd><ul class="plain">${codes || "<li>없음</li>"}</ul></dd>
    </dl></details>
    <p class="muted small">서명은 누가 서명했는지와 서명 뒤 바뀌지 않았다는 것만 보증합니다. 그림 내용이 사실이라는 보증은 아닙니다. 작업 기록은 서명자가 적은 것이고 검증기가 직접 본 것이 아닙니다.</p>`;
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
