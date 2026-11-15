// 문서 텍스트 지문 — tools/dfp.py와 같은 규격의 '포함도'(질의 shingle 중 원본에도 있는 비율, bottom-k 표본 추정).
// 단어 = [0-9a-z가-힣]+ (소문자), shingle = 연속 5단어, shingle 해시 = SHA-256 앞 8바이트 >> 3.
export const K = 5, CON_T = 0.352; // CON_T: 실험 22의 query 단위 1% threshold

export function words(text) {
  return text.toLowerCase().match(/[0-9a-z가-힣]+/g) || [];
}

export async function shingleHashes(text) {
  const w = words(text), enc = new TextEncoder(), out = new Set();
  for (let i = 0; i + K <= w.length; i++) {
    const d = new Uint8Array(await crypto.subtle.digest("SHA-256", enc.encode(w.slice(i, i + K).join(" "))));
    let v = 0n;
    for (let k = 0; k < 8; k++) v = (v << 8n) | BigInt(d[k]);
    out.add(v >> 3n);
  }
  return out;
}

// refBottom: 원장의 bottom-k 표본(10진수 문자열 배열, 오름차순)
export function containment(qSet, refBottom) {
  if (!refBottom.length || !qSet.size) return 0;
  const ref = new Set(refBottom.map((s) => BigInt(s)));
  const hi = BigInt(refBottom[refBottom.length - 1]);
  let n = 0, hit = 0;
  for (const v of qSet) if (v <= hi) { n++; if (ref.has(v)) hit++; }
  return n ? hit / n : 0;
}

// 브라우저: PDF → 본문 텍스트(pdf.js)
export async function pdfText(blob) {
  const pdfjs = await import("https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.min.mjs");
  pdfjs.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.worker.min.mjs";
  // cMap: 한글 등 CJK 폰트 문자 대응표. 없으면 한글 PDF에서 글자를 하나도 못 뽑는다(11-15 Node 시험에서 확인)
  const base = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/";
  const doc = await pdfjs.getDocument({ data: new Uint8Array(await blob.arrayBuffer()), cMapUrl: base + "cmaps/", cMapPacked: true,
    standardFontDataUrl: base + "standard_fonts/" }).promise;
  const parts = [];
  for (let p = 1; p <= doc.numPages; p++) {
    const c = await (await doc.getPage(p)).getTextContent();
    parts.push(c.items.map((it) => it.str + (it.hasEOL ? "\n" : " ")).join(""));
  }
  return parts.join("\n");
}

// docx 본문(tools/dfp.py docx_text와 같은 규칙): word/document.xml의 <w:t> 글자를 문단(<w:p>)마다 줄바꿈으로
export function docxTextFromXml(xml) {
  const paras = xml.match(/<w:p[ >][\s\S]*?<\/w:p>/g) || [];
  const un = (s) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'").replace(/&amp;/g, "&");
  return paras.map((p) => un((p.match(/<w:t(?: [^>]*)?>[\s\S]*?<\/w:t>/g) || []).map((x) => x.replace(/^<w:t(?: [^>]*)?>|<\/w:t>$/g, "")).join(""))).join(String.fromCharCode(10));
}
export async function docxText(blob) {
  const JSZip = (await import("https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm")).default;
  const zip = await JSZip.loadAsync(await blob.arrayBuffer());
  return docxTextFromXml(await zip.file("word/document.xml").async("string"));
}
