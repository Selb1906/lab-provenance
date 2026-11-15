# lab-provenance

연구실 홈페이지에 콘텐츠 출처증명(C2PA)을 붙이고, C2PA가 혼자서는 왜 부족한지와 fingerprint·watermark·원장(ledger)·포렌식이 각각 무엇을 메우는지를 실험으로 확인한 산학협력 과제 저장소입니다.

- 사이트: https://selb1906.github.io/lab-provenance/ (GitHub Pages, `docs/`)
- 전체 기록: [`notes/개발일지.md`](notes/개발일지.md) — 모든 결정의 "무엇을 — 왜", 실험 설계·예측·결과, 시행착오
- 발표 요약: [`notes/발표요약_1117.md`](notes/발표요약_1117.md) · 구상의 흐름: [`notes/구상과_전개.md`](notes/구상과_전개.md) · 조사 보고서: [`notes/research/`](notes/research/)

## 사이트에서 되는 것

| 페이지 | 내용 |
|---|---|
| 검증하기 | 파일을 올리면 **브라우저 안에서** C2PA 증명서를 읽고(그림·음원·영상·docx), 증명서가 없으면 지문으로 연구실 원장을 찾음(그림·음원·영상·PDF·docx). 파일은 서버로 보내지 않음 |
| 자동 발견 | 순찰기가 가짜 재게시 페이지를 돌며 연구실 콘텐츠의 사본을 찾고 근거 등급(원장 대조 / 워터마크 단서)을 보고 |
| 한 장의 여정 | 그림 하나가 메신저·크롭·스크린숏·조작·유출을 거칠 때 각 기술이 무엇을 답하는지 |
| 출처증명 | 공격마다 살아남는 기술 표, 원장과 비트코인 시각 고정 |

## 도구 (`tools/`)

| 파일 | 하는 일 |
|---|---|
| `build_site.py` | 사이트 생성. `--resign`이면 그림을 TrustMark → 그림별 block DCT key로 심고 C2PA 서명, 원장에 기록 |
| `make_certs.py`, `sign.py`, `sign_media.py` | 연구실 3단 인증서 체인 생성, C2PA 서명(그림·음원·영상) |
| `ledger.py`, `anchor.py` | 해시체인 원장(추가·검증), OpenTimestamps로 비트코인 시각 고정(stamp·upgrade·verify) |
| `fphash.py` | 브라우저와 같은 규격의 perceptual hash(lab-phash)와 조각 해시 |
| `wm_blockdct.py` | block DCT watermark(연구 중인 비디오 워터마크 core_AX v3의 이미지판), `derive_key`로 그림별 key |
| `align.py` | 원본 후보 고르기(ORB + 늘림 되돌려 보기) → SIFT registration → 유효 블록 watermark 검정 |
| `monitor.py` | 순찰기: 그림·음원·영상·문서, robots.txt 준수, 발견 증거(해시·헤더·시각) 보존, 원장 크기에 따른 2단계(SSCD 후보) |
| `afp.py`, `vfp.py`, `dfp.py` | 음원 지문(Philips 방식), 영상 지문(vPDQ 방식), 문서 지문(MinHash 포함도) |
| `media_lookup.py` | 음원·영상 파일을 원장과 대조(속도 되돌려 보기·좌우 반전 포함) |
| `attacks.py`, `ai_attacks.py` | WAVES 규약 공격 8종, AI 공격(FFDNet denoiser, SD-VAE 재생성) |
| `forensics.py`, `leak_trace.py` | ELA·EXIF·JPEG 품질 추정, 받는 사람별 TrustMark(유출 추적) |
| `make_samples.py`, `make_mirror.py`, `journey.py` | 검증 페이지 샘플, 가짜 재게시 이미지, 여정 페이지 |

브라우저 쪽 같은 규격 구현: `docs/assets/js/`(`verify.js`, `afp.js`, `pdq.js`, `dfp.js`) — 각각 Python 구현과 비트 단위로 대조함(개발일지).

## 실험 (`experiments/`, 결과 `experiments/results/`)

| 번호 | 질문 | 한 줄 결과 |
|---|---|---|
| 1 | perceptual hash 4종은 어떤 공격에 강한가 | 압축·잡음·흐림 강함, 회전·crop 약함 |
| 2 | watermark 4종 비교(+AI 공격) | block DCT는 신호 처리, TrustMark는 crop에 강함. AI denoiser엔 둘 다 약함 |
| 3 | PDQ·ORB로 회전·crop | ORB가 회전을 메움 |
| 5, 8 | 조각 해시로 crop | query 단위 오탐으로 재면 13~24% |
| 7 | 두 watermark 함께 | 큰 간섭 없음 |
| 9 | 원장 규모와 perceptual hash threshold | 15는 너무 느슨 → 사이트 2 |
| 10, 18, 18b | 공개 사본으로 watermark 위조 | 같은 key면 20/20 위조. 그림별 key만으로는 key 하나 기준 10%(18b), '원장 대조 → 정합' 순서까지 더하면 0/80 — 주역은 정합 단계 |
| 11, 11b, 19 | crop된 그림의 원본 찾기 | 빈틈은 대부분 시험 규약(정사각 crop 후 늘림) 탓. 실제 crop은 특징점 82~100% |
| 12, 12b | 원본에 맞춰 되돌린 뒤 watermark | crop·회전 전부 회복, 무관한 이미지 오탐 0/71 |
| 13 | AI 공격 뒤 fingerprint | 95~100% |
| 14, 16 | SSCD 후보 생성기, 원장 키우기 | 상위 5개 적중 100% (16은 메모리 부족으로 미완) |
| 15 | SyncSeal(원본 없는 되돌리기) | 사진 회복, 흰 바탕 그림 20~80% |
| 17 | perceptual hash 밝기 정규화 | 미완(메모리 부족 중지) |
| 20, 24, 25 | 음원 지문 | MP3 32kbps 10/10, 속도 ±5%는 되돌려 보기로 10/10 |
| 21 | 영상 지문 | 재인코딩·축소·잘라 올리기 10/10 |
| 22 | 문서 지문 | 재저장·발췌·수정 34/34 |
| 23 | 영상에 block DCT(프레임 누적) | 미완(메모리 부족 중지, 수정판 준비) |

## 실행 환경

- Python 3.11 가상환경(노트북 C 드라이브 여유가 적어 `E:/envs/c2pa-site`), 모델·실험 산출물은 `E:/envs/c2pa-work`.
- **서명 개인키와 watermark 주 key는 저장소에 없습니다**(`E:/envs/c2pa-site-keys`). 공개 저장소에 key가 있으면 누구나 watermark를 위조할 수 있다는 것을 실험 10에서 확인했습니다.
- 사이트 다시 만들기: `python tools/build_site.py` (그림 재서명은 `--resign`, key 필요)
- 원장 검증: `python -c "import sys; sys.path.insert(0,'tools'); import ledger; print(ledger.verify('docs/assets/ledger.jsonl'))"` → `None`이면 해시체인 정상
- 비트코인 시각 고정 확인: `python tools/anchor.py verify`

## 공개 범위

공개 저장소이므로 개인키·모델 파일·비공개 자료(미공개 특허, 현장 조사, 개인정보)는 넣지 않습니다. 논문 그림은 오픈 액세스 라이선스 범위에서만 발행하고, 특허는 서지 정보만 적습니다. 실험에 쓴 외부 논문 PDF·그림은 노트북에서만 쓰고 올리지 않았습니다.
