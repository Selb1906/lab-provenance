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
