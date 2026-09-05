# 3B Local LLM × Subculture Persona Dataset 프로젝트 기획서

## 1. 프로젝트 개요

### 프로젝트 목표

한국어 서브컬처/캐릭터 데이터를 활용하여 **3B급 소형 언어 모델의 캐릭터 대화 및 Persona 표현 능력을 구축하고 평가**한다.

단순 캐릭터 챗봇 개발에 그치지 않고, 프로젝트 진행 과정에서 발견되는 문제를 기반으로 연구 질문을 도출하여 학·석사 연구로 확장하는 것을 목표로 한다.

전체 방향은 다음과 같다.

```text
데이터 수집
    ↓
데이터 정제
    ↓
Persona Dataset 구축
    ↓
3B Baseline 구축
    ↓
Fine-tuning / RAG / Memory
    ↓
Evaluation
    ↓
문제 발견
    ↓
Research Question
    ↓
방법론 제안
    ↓
실험 / Ablation
    ↓
논문화
```

---

# 2. 핵심 연구 관심사

## Main Question

> 3B급 소형 언어 모델이 제한된 컴퓨팅 환경에서도 캐릭터의 말투, 성격, 관계, 설정을 일관되게 유지할 수 있는가?

이를 다음 세부 질문으로 확장한다.

### RQ1. Persona Learning

구조화된 캐릭터 데이터가 일반적인 dialogue 데이터보다 Persona 표현 능력을 향상시키는가?

### RQ2. Long-term Persona Consistency

대화가 길어질수록 3B 모델에서 Persona Drift가 얼마나 발생하는가?

### RQ3. Persona Memory

외부 Persona Memory를 사용하면 장기 대화에서 캐릭터 일관성을 향상시킬 수 있는가?

### RQ4. Efficient Persona LLM

Quantization 등 경량화 이후에도 캐릭터의 말투와 Persona 특성이 유지되는가?

---

# 3. Dataset

## 3.1 Raw Dataset

가능한 데이터 구성 요소:

```text
Character Dialogue
Character Description
Character Personality
Character Relationship
World Setting
Character Speech Style
Character Emotion
Scene / Context
```

가능하면 단순한 `input-output` 데이터보다 구조화된 형태로 저장한다.

예:

```json
{
    "character": "Character_A",

    "persona": {
        "personality": [],
        "speech_style": [],
        "likes": [],
        "dislikes": []
    },

    "relationships": {
        "Character_B": "friend",
        "Character_C": "enemy"
    },

    "world_knowledge": [],

    "dialogue": []
}
```

---

# 4. Data Pipeline

전체 데이터 처리 과정:

```text
Raw Data
   ↓
Parsing
   ↓
Normalization
   ↓
Exact Dedup
   ↓
Fuzzy Dedup
   ↓
Language Filtering
   ↓
Quality Filtering
   ↓
Persona Extraction / Annotation
   ↓
Dataset Formatting
   ↓
Train / Validation / Test Split
```

---

# 5. 현재 알아둘 데이터 처리 개념

## Normalization

텍스트 표현을 일정하게 정규화한다.

예:

```text
Unicode normalization
공백 정리
특수문자 처리
개행 처리
```

---

## Exact Dedup

완전히 동일한 데이터를 제거한다.

```text
A == B
→ duplicate
```

---

## Fuzzy Dedup

완전히 동일하지 않아도 매우 유사한 데이터를 찾아 제거한다.

개념:

```text
비슷한 문장
→ Near Duplicate
→ 제거 후보
```

대표적인 구조:

```text
Text
 ↓
n-gram / Shingle
 ↓
MinHash
 ↓
LSH
 ↓
Candidate Pair
 ↓
Similarity Check
 ↓
Duplicate Decision
```

---

## n-gram

텍스트를 일정한 크기의 조각으로 분리하여 특징으로 사용한다.

```text
Text
 ↓
small text chunks
```

---

## Jaccard Similarity

두 집합이 얼마나 겹치는지 측정한다.

```text
J(A,B)

= Intersection / Union
```

즉:

```text
공통 특징
──────────
전체 특징
```

---

## MinHash

Jaccard Similarity를 빠르게 근사하기 위해 문서 특징을 작은 signature로 압축한다.

```text
큰 n-gram 집합
      ↓
   MinHash
      ↓
작은 Signature
```

---

## LSH

Locality-Sensitive Hashing.

비슷한 데이터가 높은 확률로 같은 bucket에 들어가도록 하여 similarity search를 빠르게 수행한다.

```text
MinHash Signature
       ↓
      LSH
       ↓

Bucket A
├─ Similar Document 1
├─ Similar Document 2
└─ Similar Document 3

Bucket B
├─ Different Document
└─ ...
```

핵심 개념:

```text
일반 Hash
→ 동일한 데이터 찾기

LSH
→ 비슷한 데이터 찾기
```

---

# 6. Tokenizer

## BPE

Byte Pair Encoding.

가장 자주 등장하는 인접 token pair를 하나의 token으로 합치는 과정을 반복한다.

```text
l o w
l o w e r

↓

lo w
lo w e r

↓

low
low e r
```

핵심:

```text
BPE
→ 가장 빈번한 인접 쌍을 합침
→ 반복
→ Vocabulary 생성
```

---

# 7. Model

기본 모델은 **약 3B parameter급 LLM**을 사용한다.

실험 후보:

```text
Base 3B

↓

Instruct 3B

↓

Persona Fine-tuned 3B
```

필요한 경우 다음 방법을 비교한다.

```text
Prompt Only

RAG

LoRA

QLoRA

Persona Memory
```

---

# 8. Baseline

연구를 위해 반드시 Baseline을 먼저 구축한다.

예:

| Model    | Persona Data | RAG | Memory | Quantization |
| -------- | ------------ | --- | ------ | ------------ |
| Baseline | X            | X   | X      | FP16         |
| Prompt   | X            | X   | X      | FP16         |
| LoRA     | O            | X   | X      | FP16         |
| RAG      | X            | O   | X      | FP16         |
| Memory   | O            | X   | O      | FP16         |
| Q4       | O            | X   | O      | INT4         |

새로운 방법을 만들기 전에 기존 방법들의 성능을 확인한다.

---

# 9. Persona Memory

장기 대화에서 Persona Drift를 줄이기 위한 후보 구조.

```text
              User
                ↓
         Conversation
                ↓
        Persona Memory
                │
       ┌────────┼─────────┐
       ↓        ↓         ↓
   Identity  Relation   Episode
    Memory    Memory     Memory
       │        │         │
       └────────┼─────────┘
                ↓
              3B LLM
                ↓
             Response
```

Memory 종류:

### Identity Memory

변하지 않아야 하는 캐릭터 정보.

```text
이름
성격
직업
세계관 설정
말투
```

### Relationship Memory

다른 캐릭터와의 관계.

```text
friend
enemy
family
senior
junior
etc.
```

### Episodic Memory

이전 대화에서 발생한 사건.

```text
User와 무엇을 이야기했는가?
무슨 약속을 했는가?
어떤 사건이 있었는가?
```

---

# 10. Evaluation

Evaluation은 **모델 또는 방법이 실제로 개선되었는지 객관적으로 측정하는 과정**이다.

Persona 모델에서는 단순 Accuracy만으로 평가하기 어렵다.

따라서 다음과 같은 평가 기준을 고려한다.

```text
Persona Consistency
Speech Style
Relationship Consistency
World Knowledge
Emotion Consistency
Long-term Consistency
```

예:

| Metric | 의미                       |
| ------ | ------------------------ |
| PCS    | Persona Consistency      |
| STS    | Speech Style             |
| RCS    | Relationship Consistency |
| WKS    | World Knowledge          |
| ECS    | Emotional Consistency    |
| LCS    | Long-term Consistency    |

평가 방법:

```text
Automatic Metric
       +
LLM-as-a-Judge
       +
Human Evaluation
```

가능하면 여러 평가 방법을 함께 사용한다.

---

# 11. Long-Conversation Evaluation

Persona Drift를 측정하기 위해 대화 길이를 변화시킨다.

```text
1 Turn
 ↓
5 Turns
 ↓
10 Turns
 ↓
20 Turns
 ↓
50 Turns
```

측정:

```text
Turn 증가

vs

Persona Consistency
Relationship Consistency
Speech Style
Hallucination
Memory Accuracy
```

---

# 12. Quantization Experiment

로컬 환경에서 사용하기 위해 모델 경량화를 실험한다.

예:

```text
FP16
 ↓
Q8
 ↓
Q6
 ↓
Q5
 ↓
Q4
```

평가:

| Precision | Persona | Quality | RAM/VRAM | tok/s | TTFT |
| --------- | ------: | ------: | -------: | ----: | ---: |
| FP16      |     TBD |     TBD |      TBD |   TBD |  TBD |
| Q8        |     TBD |     TBD |      TBD |   TBD |  TBD |
| Q5        |     TBD |     TBD |      TBD |   TBD |  TBD |
| Q4        |     TBD |     TBD |      TBD |   TBD |  TBD |

연구 질문:

> Quantization이 일반적인 language capability뿐 아니라 Persona/Style consistency에도 영향을 미치는가?

---

# 13. Ablation Study

새로운 방법을 제안했을 경우 어떤 구성 요소가 실제 성능 향상에 기여했는지 분석한다.

예:

```text
Proposed Method

Persona Memory
     +
Relationship Memory
     +
Episodic Memory
```

비교:

| Method            | Persona | Relation | Long-term |
| ----------------- | ------: | -------: | --------: |
| Baseline          |     TBD |      TBD |       TBD |
| + Persona Memory  |     TBD |      TBD |       TBD |
| + Relation Memory |     TBD |      TBD |       TBD |
| + Episodic Memory |     TBD |      TBD |       TBD |
| Full Method       |     TBD |      TBD |       TBD |

---

# 14. Research Workflow

연구 구현 과정은 다음 방식을 사용한다.

```text
Problem
 ↓
Paper Search
 ↓
Method 이해
 ↓
Official Implementation Search
 ↓
Reference Code 확인
 ↓
Project 적용 판단
 ↓
Coding Agent를 통한 Adaptation
 ↓
Code Review
 ↓
Small Test
 ↓
Experiment
 ↓
Evaluation
 ↓
Analysis
 ↓
Research Note
```

---

# 15. Coding Agent 사용 원칙

AI Coding Agent는 적극적으로 사용한다.

Agent가 담당할 수 있는 영역:

```text
Boilerplate
Refactoring
Dataset Loader
Config
Logging
API
Visualization
Unit Test
Existing Method Adaptation
```

다만 연구 핵심 부분은 직접 이해한다.

```text
Research Question
Method
Dataset
Training
Evaluation
Metric
Experiment Design
Result Analysis
```

기준:

> 모든 코드를 직접 작성할 필요는 없지만, 논문에서 사용하는 핵심 코드가 무엇을 하는지는 설명할 수 있어야 한다.

---

# 16. Reference Code 관리

인터넷/GitHub에서 유용한 코드를 발견하면 적극적으로 참고한다.

Workflow:

```text
코드 발견

"이거 괜찮네!"

↓

방법론 확인

↓

SOURCE / LICENSE 확인

↓

Reference 저장

↓

Agent에게 프로젝트에 맞게 수정 요청

↓

Diff 확인

↓

실험
```

권장 주석:

```python
# WHAT:
# MinHash + LSH 기반 fuzzy deduplication
#
# WHY:
# Persona dataset의 near-duplicate 제거
#
# SOURCE:
# Paper / Repository / Commit 기록
#
# LICENSE:
# 확인 후 기록
#
# NOTE:
# 현재 threshold = TBD
# 필요하면 threshold ablation 수행
```

---

# 17. 프로젝트 폴더 구조

```text
project/

├── README.md
│
├── src/
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── memory/
│   ├── inference/
│   └── evaluation/
│
├── configs/
│   ├── baseline.yaml
│   ├── lora.yaml
│   ├── memory.yaml
│   └── quantization.yaml
│
├── references/
│   ├── preprocessing/
│   ├── persona/
│   ├── memory/
│   └── quantization/
│
├── experiments/
│   ├── exp001_baseline/
│   ├── exp002_lora/
│   ├── exp003_memory/
│   └── exp004_quantization/
│
├── tests/
│
└── docs/
    ├── papers.md
    ├── methodology.md
    ├── dataset.md
    └── experiment_log.md
```

---

# 18. Experiment 기록

각 실험은 다음 형식으로 기록한다.

```text
Experiment:
EXP-001

Question:
Persona fine-tuning이 baseline보다 좋은가?

Hypothesis:
Persona dataset으로 LoRA fine-tuning하면
persona consistency가 증가할 것이다.

Model:
3B

Dataset:
TBD

Method:
LoRA

Baseline:
Prompt-only 3B

Metrics:
PCS
STS
RCS

Seed:
42 / 123 / 777

Result:
TBD

Observation:
TBD

Conclusion:
TBD

Next Experiment:
TBD
```

---

# 19. 연구 접근 방식

프로젝트 초반에는 모든 방법을 깊게 공부한 뒤 구현하려 하지 않는다.

기본 접근:

```text
"이 방법 괜찮네!"
        ↓
개념 이해
        ↓
일단 적용
        ↓
실험
        ↓
문제 발견
        ↓
중요한 부분만 Deep Dive
```

예:

```text
BPE
→ 자주 등장하는 인접 쌍을 반복적으로 합친다.

Fuzzy Dedup
→ 비슷한 데이터 중복 제거.

MinHash
→ Jaccard similarity를 효율적으로 근사.

LSH
→ 비슷한 데이터를 빠르게 후보군으로 찾는다.

LoRA
→ 일부 저랭크 파라미터를 학습해 효율적으로 fine-tuning.

Quantization
→ 낮은 precision을 사용해 모델 메모리/연산량 감소.

RAG
→ 외부 정보를 검색하여 context에 제공.

Persona Memory
→ 캐릭터 정보를 외부 memory에서 관리하고 필요할 때 제공.
```

연구 결과에 직접 영향을 주는 요소가 발견되면 해당 부분을 깊게 분석한다.

---

# 20. 최종 목표

### Stage 1 — Engineering

```text
Dataset Pipeline
        ↓
3B Model
        ↓
Persona Chat
        ↓
Evaluation Pipeline
```

### Stage 2 — Experiment

```text
Prompt
vs
LoRA
vs
RAG
vs
Memory
vs
Quantization
```

### Stage 3 — Research

실험 과정에서 문제를 발견한다.

```text
Observation
     ↓
"왜 이런 현상이 발생하지?"
     ↓
Research Question
     ↓
Hypothesis
```

### Stage 4 — Proposed Method

기존 방법의 문제를 해결하는 새로운 방법을 설계한다.

```text
Baseline
   ↓
Problem
   ↓
Proposed Method
   ↓
Experiment
   ↓
Ablation
```

### Stage 5 — Thesis / Paper

최종적으로 다음과 같은 형태의 연구를 목표로 한다.

> **Efficient Persona-Consistent Role-Playing with Small Language Models**

핵심 키워드:

```text
Small Language Model
3B LLM
Korean Subculture Dataset
Character Role-Playing
Persona Consistency
Long-Term Memory
Efficient AI
Local LLM
Quantization
```

---

# 한 줄 원칙

> **구현은 빠르게, 출처는 남기고, 중요한 부분은 깊게 파고, 개선을 주장하려면 실험으로 증명한다.**
