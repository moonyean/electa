# 연구노트 — Pretraining 데이터 파이프라인 (2026-08-30 ~ 2026-09-05)

## 1. 데이터셋 선정

### Pretraining
- **FineWeb2** (`HuggingFaceFW/fineweb-2`, config `kor_Hang`)
- **한국어 위키백과** (`wikimedia/wikipedia`, config `20231101.ko`)

**선정 이유**: 가장 유명하고 쉽게 접근할 수 있어서. 별도 크롤링/전처리 없이 이미 웹 필터링을 거친 대규모 한국어 코퍼스를 바로 받아 쓸 수 있고, 커뮤니티에서 검증된 표준 소스라 재현성과 신뢰도가 높다.

전체 62개 샤드 중 7개(약 32GB)만 확보 (10~12B 토큰 목표에 맞춰 후보군만 우선 확보, 전량 다운로드는 하지 않음).

### SFT (참고용, 아직 본격 작업 전)
- `huggingface-KREW/korean-role-playing` (gf-persona-data 등)
- `beomi/KoAlpaca-RealQA`, `lemon-mint/Korean-FineTome-100k`, `lemon-mint/smol-koreantalk`, `mkd-chanwoo/keural-rag-chatml-ko`, `developer-lunark/korean-character-roleplay-sft`, `junidude14/korean_roleplay_dataset_for_chat_game_2`

이미 확보된 양이 충분해 현재 병목이 아님. 정제 연습(`gf_persona_cleaning.ipynb`)만 진행, 본격 작업은 pretraining 이후로 미룸.

---

## 2. Pretraining 전처리 파이프라인

### 파이프라인 구성 (한 번의 실행에 포함)

```
FTFY (모지바케 복원)
    ↓
기호 라인 제거 (SymbolLinesFormatter)
    ↓
PII 제거 (이메일/IP)
    ↓
한글 비율 필터 (65% 미만 제거, KoreanRatioFilter)
    ↓
Gopher 품질 필터 (반복 문장·저품질 문서 제거)
    ↓
Gopher 반복 탐지 (문단/줄/n-gram 반복 문서 제거)
    ↓
Exact Dedup (완전 중복 문서 제거)
    ↓
MinHash Fuzzy Dedup (LSH 기반 유사 문서 제거)
```

구현: `src/data/fineWeb2_preprocess.py` (datatrove 라이브러리 기반). 한국어 형태소 분석은 Kiwi(`kiwipiepy`) 사용.

### 최종 산출물

- **`data/processed/pretrain/`**: 17,484,971 문서, 26GB (12개 파일)
- 7개 원본 샤드 → 정제 후 exact dedup으로 17,686,307 → 샤드 간 MinHash로 17,484,971 (교차 중복 약 20만 건 추가 제거)

### 처리 방식: 샤드 단위 분할 → 순차 처리

한 샤드(약 278만 문서)를 11조각으로 쪼개 11-way 병렬 처리 시 약 **2시간 10~20분** 소요 (실측 3회 평균, 편차 거의 없음). 7개 샤드를 이 방식으로 순차 처리.

---

## 3. 겪은 문제와 원인 (기술 부채 기록)

이 프로젝트의 가장 큰 시간 소모는 알고리즘이 아니라 **Windows 환경 + 장시간 백그라운드 실행의 인프라 문제**였다. 기록해둘 가치가 있는 이슈들:

| 문제 | 원인 | 해결 |
| --- | --- | --- |
| 세션 재시작 시 백그라운드 작업 사망 | 셸에 종속된 프로세스는 세션 종료 시 같이 죽음 | Windows 작업 스케줄러(Task Scheduler)로 완전히 분리해서 실행 |
| 스케줄러로 띄운 프로세스가 `PermissionError`로 죽음 | 스케줄러 기본 실행 컨텍스트의 보안 토큰이 멀티프로세싱 파이프 핸들 복제와 충돌 | `/IT`(Interactive Token) 옵션으로 대화형 세션 토큰 사용 |
| 세션 인터럽트 시 스케줄러 작업도 같이 죽음 | Windows 콘솔 Ctrl+C 브로드캐스트가 같은 콘솔에 물린 프로세스까지 전파 | `start "title" /min cmd /c ...`로 완전히 새 콘솔(프로세스 그룹) 생성 |
| Kiwi 한국어 형태소 분석기 로드 실패 | ① datatrove 내부 CSV가 UTF-8인데 Windows 기본 코드페이지(cp949)로 읽어서 깨짐 ② datatrove 기본 Kiwi 모델(`sbg`)이 kiwipiepy 0.22+에서 배포 모델에서 빠짐 | UTF-8 강제 후 캐시 워밍업, 모델을 `cong`으로 교체 |
| `GopherQualityFilter`가 한국어 문서를 거의 다 걸러냄 | 기본값이 영어 기준(평균 단어 길이 3~10자, 영어 stop word) | Kiwi 형태소 기준 실측 평균 길이(~1.7자)로 재조정, 한국어 조사로 stop word 교체 |
| exact/minhash dedup에서 `TypeError` | xxhash는 입력 타입 상관없이 bytes만 받는데 문자열을 넘김 | `content_getter`가 bytes를 반환하도록 수정 |
| 재개 스크립트에서 exact dedup 결과가 통째로 비어서 저장된 데이터까지 삭제됨 | 원본 CLI(`main()`)에는 있던 Windows UTF-8 강제 설정이, 함수를 직접 호출하는 재개 스크립트에는 빠져 있어서 gzip 내용을 cp949로 잘못 읽음(전부 0건 처리) | 모든 커스텀 스크립트에 동일한 UTF-8 강제 로직 적용 + exact dedup 결과가 비면 즉시 중단하고 이전 단계 산출물을 보존하는 안전장치 추가 |
| 디스크 부족으로 파이프라인 반복적으로 중단 | 각 단계(정제→exact→minhash)가 이전 단계만큼(때로는 그 이상) 디스크를 다시 잡아먹는데 정리를 안 함 | 다음 단계가 성공하면 이전 단계 중간 산출물을 자동 삭제하도록 수정, 대용량 원본은 하드링크로 복사 비용 없이 재사용 |
| 한 샤드를 통째로 한 워커에 맡기면 병렬도가 안 나옴 | datatrove reader는 **파일 단위**로만 작업을 나눔 (`tasks`를 파일 수보다 크게 줘도 무의미) | 샤드를 여러 조각으로 물리적으로 분할해서 파일 수 자체를 늘림 |

### 교훈

1. **작업을 잘게 쪼갤수록 복원력이 높아진다.** 큰 단위(샤드 전체)로 돌리다 죽으면 몇 시간~하루치가 통째로 날아가지만, 작은 단위(샤드를 11조각)로 쪼개서 순차 처리하면 죽어도 잃는 양이 조각 하나 분량으로 제한된다.
2. **`tasks`(작업 조각 수)와 `workers`(동시 실행 수)는 분리해서 생각해야 한다.** 조각은 잘게 쪼개되 동시 실행 개수는 메모리 안전선으로 따로 제한하는 것이 최적.
3. Windows에서 장시간 백그라운드 프로세스를 안전하게 돌리려면: 셸 세션과 완전히 분리(작업 스케줄러) + 새 콘솔/프로세스 그룹(인터럽트 격리) + 명시적 UTF-8 강제(인코딩 버그 회피) 세 가지가 다 필요하다.

---

## 4. 다음 단계 논의 — 토크나이저

기획서상 다음 단계는 SentencePiece 32K 학습. 알고리즘 선택지 비교:

| 방식 | 장점 | 단점 |
| --- | --- | --- |
| BPE | 단순·빠름·예측 가능, GPT/LLaMA 계열 검증됨 | 형태소 경계를 잘 못 맞춤, 유연성 없음 |
| Unigram (SentencePiece 기본) | 형태소 경계에 더 자연스러움(교착어 유리), subword regularization으로 일반화 도움 | 학습이 BPE보다 다소 복잡 |
| Byte-level BPE | OOV 없음, 어떤 유니코드도 안전 | 한글처럼 다바이트 문자는 초반 압축률 손해 |
| WordPiece | BPE와 실질적 차이 적음 | 선택할 이점 없음 |
| 형태소(Kiwi) 우선 분절 + BPE/Unigram 하이브리드 | 언어학적으로 가장 깔끔한 분절 | 추론 시에도 Kiwi 의존성 필요, 구현 복잡도 증가 |
| 순수 음절/문자 단위 | 구현 단순 | 시퀀스 길이 증가로 350M 규모엔 비효율 |
| Unigram + byte_fallback | Unigram의 장점 + OOV 안전성 동시 확보 | 거의 없음 |

**현재 추천**: Unigram + `byte_fallback=True`. 350M 규모·교착어 특성·구현 복잡도를 종합적으로 고려한 실용적 절충안. (최종 결정 대기 중)
