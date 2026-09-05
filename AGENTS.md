# AGENTS.md

한국어 350M LM 프로젝트. 자세한 배경·일정·목표는 [기획서.md](기획서.md) 참고.

## 문서 위치 규칙

새 `.md` 문서는 전부 `docs/`에 만든다. 예외: `기획서.md`(기획 문서)와 이 `AGENTS.md`(도구 컨벤션 파일)만 루트에 둔다.

## 작업 방식

이 프로젝트는 사용자가 직접 자료를 찾고 방향을 정한다. 에이전트는 지시받은 작업만 수행하고, 다음 단계를 먼저 제안하거나 임의로 시작하지 않는다. 작업 끝나면 결과만 보고하고 사용자의 다음 지시를 기다린다.

## 디렉터리 구조

```
data/
  raw/
    pretrain/    FineWeb2 한국어(kor_Hang, 7/62 샤드, ~32GB) + 한국어 위키(20231101.ko, 전체)
    sft/         대화/instruction 데이터셋 6종 (beomi, developer-lunark, huggingface-krew,
                 junidude14, lemon-mint, mkd-chanwoo) — 이미 충분함, 현재 병목 아님
  interim/{pretrain,sft}/   정제 중간 산출물 (완료 후 정리 대상)
  processed/{pretrain,sft}/ 최종 산출물
  sources/                  외부 git 레포 (korean-people-persona, open-korean-instructions)
notebooks/    탐색용 ipynb
src/{data,tokenizer,model,training,eval,utils}/   단계별 소스 (대부분 아직 스켈레톤)
scripts/      CLI 진입점
configs/      토크나이저·모델·학습 설정
checkpoints/, logs/   학습 산출물 (gitignore)
docs/         dataset.md 등 문서
```

## Pretraining 파이프라인 (`src/data/fineWeb2_preprocess.py`)

FineWeb2/위키 원시 corpus를 정제·중복제거하는 스크립트. 실행:

```bash
python src/data/fineWeb2_preprocess.py --tasks <N>
```

파이프라인 순서 (한 번의 실행에 다 포함됨): FTFY(모지바케 복원) → 기호 라인 제거 → PII 제거 → 한글 비율 필터(65% 미만 제거) → Gopher 품질 필터 → Gopher 반복 탐지 → Exact Dedup → MinHash Fuzzy Dedup(LSH).

**환경 관련 주의사항 (Windows + Python 3.14 + datatrove 0.10.0 조합에서 필요했던 우회):**
- `importlib.metadata`를 명시적으로 import해야 하는 버그 있음 (안 하면 의존성 체크에서 에러)
- 한국어 형태소 분석은 Kiwi(`kiwipiepy`) 사용 — datatrove 기본 모델(`sbg`)이 kiwipiepy 0.22+에서 빠져서 `cong`으로 교체 필요, CSV 자산 파일도 cp949로 깨져서 임시 UTF-8 강제 필요 (`_prepare_korean_tokenizer()` 참고)
- xxhash는 문자열이 아니라 bytes만 받음 — exact/minhash dedup의 `content_getter`는 반드시 bytes 반환해야 함
- `MinhashDedupBuckets`는 `config`를 키워드 인자로 넘겨야 함 (위치 인자 순서가 `index_folder`와 꼬임)
- Gopher 계열 필터는 영어 기본값이라 한국어엔 `language=Languages.korean` + 평균 단어 길이(1~10, Kiwi 형태소 기준)·불용어(한국어 조사) 재설정 필요
- 입력 파일 수가 곧 최대 병렬도(파일 단위로만 샤딩됨) — `--tasks`를 파일 수보다 크게 줘도 의미 없음, 필요하면 원본을 여러 파일로 쪼개서 넣을 것
- 워커 하나당 Kiwi 모델을 개별 로드해서 메모리 소모가 큼 — tasks 수를 메모리 여유에 맞춰 조절할 것 (코어 수만 보고 정하면 OOM 위험)

## 현재 진행 상태 (2026-09-01 기준)

- Pretraining 원시 corpus 다운로드 완료 (FineWeb2 7샤드 + 위키)
- 1차 전체 전처리(FTFY/반복탐지 **미포함** 버전) 완료 → `data/processed/pretrain/` (17,820,098 문서, 27GB)
- FTFY + 반복탐지 넣은 버전으로 **재실행 예정** (아직 실행 안 함)
- Train/Val 분리, SentencePiece 토크나이저 학습, 토큰 인코딩(uint16) 전부 미착수
- 모델 구현·학습(스모크런/본선)·SFT 전부 미착수
