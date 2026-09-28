"""한국어 pretraining 데이터로 SentencePiece Unigram 토크나이저를 학습한다.

입력::

    data/processed/pretrain_split/train/*.jsonl.gz

출력::

    data/processed/tokenizer/unigram_32k.model
    data/processed/tokenizer/unigram_32k.vocab

학습용 말뭉치 생성은 샤드 단위 체크포인트를 지원한다. CPU 프로세스가
중단되면 마지막으로 완료된 샤드 다음부터 다시 시작할 수 있다.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import sys
import json
import os
import random
from pathlib import Path

import sentencepiece as spm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import file_identity, save_state

MAX_SENTENCE_BYTES = 4096


def open_jsonl(path: Path):
    """압축 여부에 맞춰 JSONL 파일을 UTF-8 텍스트로 연다."""
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def get_input_files(input_dir: Path, seed: int) -> list[Path]:
    """입력 JSONL 샤드를 고정된 순서로 섞어 반환한다."""
    files = sorted(input_dir.glob("*.jsonl.gz"))
    files.extend(sorted(input_dir.glob("*.jsonl")))
    if not files:
        raise FileNotFoundError(f"토크나이저 입력 파일이 없습니다: {input_dir.resolve()}")
    random.Random(seed).shuffle(files)
    return files


def write_checkpoint(checkpoint_path: Path, state: dict) -> None:
    """체크포인트를 임시 파일에 쓴 뒤 원자적으로 교체한다."""
    save_state(checkpoint_path, state)


def load_checkpoint(checkpoint_path: Path) -> dict | None:
    """기존 체크포인트가 있으면 읽는다."""
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sentence_chunks(text: str, max_bytes: int = MAX_SENTENCE_BYTES):
    """UTF-8 문자 경계를 유지하며 SentencePiece 입력 길이 이내로 분할한다."""
    if max_bytes < 4:
        raise ValueError("max_bytes는 4 이상이어야 합니다.")
    chunk, size = [], 0
    for char in text.replace("\n", " ").replace("\r", " "):
        width = len(char.encode("utf-8"))
        if size + width > max_bytes:
            yield "".join(chunk)
            chunk, size = [], 0
        chunk.append(char)
        size += width
    if chunk:
        yield "".join(chunk)


def sample_shard(path: Path, documents: int, characters: int, seed: int):
    """샤드 전체에서 난수 우선순위로 표본을 유지하며 메모리를 문자 예산으로 제한한다.

    샤드별 동일 예산의 층화 표본이다. 예산 경계 문서는 일부만 포함될 수 있다.
    """
    rng = random.Random(seed)
    heap = []
    used = eligible = 0
    with open_jsonl(path) as source:
        for line_number, line in enumerate(source):
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            eligible += 1
            if not documents or not characters:
                continue
            text = text.strip()[:characters]
            item = (-rng.random(), line_number, text)
            heapq.heappush(heap, item)
            used += len(text)
            if len(heap) > documents:
                used -= len(heapq.heappop(heap)[2])
            while used > characters:
                priority, number, value = heapq.heappop(heap)
                excess = used - characters
                used -= len(value)
                if len(value) > excess:
                    value = value[:-excess]
                    heapq.heappush(heap, (priority, number, value))
                    used += len(value)
    return [item[2] for item in sorted(heap, reverse=True)], eligible


def build_corpus(
    input_dir: Path,
    corpus_path: Path,
    checkpoint_path: Path,
    max_documents: int,
    max_characters: int,
    seed: int,
    rebuild: bool = False,
) -> tuple[int, int]:
    """모든 샤드에 예산을 나누고 완료된 샤드부터 재개한다."""
    if max_documents <= 0 or max_characters <= 0:
        raise ValueError("max_documents와 max_characters는 양수여야 합니다.")
    files = get_input_files(input_dir, seed)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "files": [file_identity(path) for path in files],
        "seed": seed, "max_documents": max_documents, "max_characters": max_characters,
        "corpus": str(corpus_path.resolve()), "max_sentence_bytes": MAX_SENTENCE_BYTES,
        "sampling": "stratified_priority_v1",
    }
    state = None if rebuild else load_checkpoint(checkpoint_path)
    if state is not None:
        if state.get("version") != 2 or state.get("settings") != settings:
            raise RuntimeError("corpus 입력/설정이 다릅니다. --rebuild-corpus가 필요합니다.")
        if not corpus_path.exists() or corpus_path.stat().st_size < state["corpus_bytes"]:
            raise RuntimeError("corpus가 없거나 체크포인트보다 짧습니다. 재생성이 필요합니다.")
        if state["completed"]:
            if file_identity(corpus_path) != state["corpus_identity"]:
                raise RuntimeError("완료된 corpus가 변경되었습니다.")
            return state["documents"], state["characters"]
        with corpus_path.open("r+b") as output:
            output.truncate(state["corpus_bytes"])
    else:
        if corpus_path.exists() and not rebuild:
            raise FileExistsError("체크포인트 없는 corpus를 덮어쓰지 않습니다. --rebuild-corpus를 사용하세요.")
        with corpus_path.open("wb"):
            pass
        state = {"version": 2, "settings": settings, "next_file_index": 0,
                 "documents": 0, "characters": 0, "lines": 0, "corpus_bytes": 0,
                 "completed": False, "shards": []}
        write_checkpoint(checkpoint_path, state)

    for index in range(state["next_file_index"], len(files)):
        doc_budget = max_documents // len(files) + (index < max_documents % len(files))
        char_budget = max_characters // len(files) + (index < max_characters % len(files))
        print(f"[{index + 1}/{len(files)}] 전체 샤드에서 표본 추출: {files[index].name}", flush=True)
        samples, eligible = sample_shard(files[index], doc_budget, char_budget, seed + index)
        count_chars = count_lines = 0
        with corpus_path.open("a", encoding="utf-8", newline="\n") as output:
            for text in samples:
                for chunk in sentence_chunks(text):
                    output.write(chunk + "\n")
                    count_lines += 1
                count_chars += len(text)
            output.flush()
            os.fsync(output.fileno())
        state["documents"] += len(samples)
        state["characters"] += count_chars
        state["lines"] += count_lines
        state["next_file_index"] = index + 1
        state["corpus_bytes"] = corpus_path.stat().st_size
        state["shards"].append({"file": str(files[index]), "eligible": eligible,
                                "sampled": len(samples), "characters": count_chars})
        write_checkpoint(checkpoint_path, state)
    if not state["documents"]:
        raise RuntimeError("학습에 사용할 문서가 없습니다.")
    state["completed"] = True
    state["corpus_identity"] = file_identity(corpus_path)
    write_checkpoint(checkpoint_path, state)
    return state["documents"], state["characters"]


def train_tokenizer(
    corpus_path: Path,
    model_prefix: Path,
    vocab_size: int,
    character_coverage: float,
    threads: int,
) -> None:
    """SentencePiece Unigram 모델을 임시 파일에 학습한 뒤 최종 파일로 교체한다."""
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    # with_suffix()를 사용하면 prefix에 이미 .tmp가 있을 때 경로가 겹친다.
    # SentencePiece가 실제로 생성하는 "prefix.model" 형태를 문자열로 명시한다.
    final_model = Path(f"{model_prefix}.model")
    final_vocab = Path(f"{model_prefix}.vocab")
    temporary_prefix = model_prefix.with_name(model_prefix.name + ".tmp")
    temporary_model = Path(f"{temporary_prefix}.model")
    temporary_vocab = Path(f"{temporary_prefix}.vocab")

    # 이전에 중단된 SentencePiece 임시 결과는 다시 학습하기 전에 제거한다.
    for path in (temporary_model, temporary_vocab):
        if path.exists():
            path.unlink()

    lines = 0
    with corpus_path.open("rb") as source:
        for line in source:
            if len(line.rstrip(b"\r\n")) > MAX_SENTENCE_BYTES:
                raise ValueError("corpus에 길이 제한 초과 줄이 있습니다. 새 corpus가 필요합니다.")
            lines += 1
    if not lines:
        raise ValueError("빈 corpus로 학습할 수 없습니다.")
    print(f"SentencePiece 입력: {lines:,}줄, 길이 초과 0줄", flush=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(temporary_prefix),
        model_type="unigram",
        max_sentence_length=MAX_SENTENCE_BYTES,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
        num_threads=threads,
    )
    temporary_model.replace(final_model)
    temporary_vocab.replace(final_vocab)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/pretrain_split/train"))
    parser.add_argument("--corpus", type=Path, default=Path("data/interim/tokenizer/unigram_corpus.txt"))
    parser.add_argument("--model-prefix", type=Path, default=Path("data/processed/tokenizer/unigram_32k"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="말뭉치 생성 체크포인트 경로")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--max-documents", type=int, default=2_000_000)
    parser.add_argument("--max-characters", type=int, default=500_000_000)
    parser.add_argument("--threads", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-corpus", action="store_true", help="말뭉치와 체크포인트를 지우고 새로 시작")
    args = parser.parse_args()

    checkpoint = args.checkpoint or args.corpus.with_suffix(args.corpus.suffix + ".checkpoint.json")
    final_model = Path(f"{args.model_prefix}.model")
    final_vocab = Path(f"{args.model_prefix}.vocab")
    if final_model.exists() or final_vocab.exists():
        raise FileExistsError(
            f"이미 최종 토크나이저 파일이 있습니다: {final_model} 또는 {final_vocab}"
        )

    documents, characters = build_corpus(
        input_dir=args.input,
        corpus_path=args.corpus,
        checkpoint_path=checkpoint,
        max_documents=args.max_documents,
        max_characters=args.max_characters,
        seed=args.seed,
        rebuild=args.rebuild_corpus,
    )
    print(f"학습용 문서 수: {documents:,}")
    print(f"학습용 문자 수: {characters:,}")
    print(f"SentencePiece Unigram 학습 시작 (CPU 스레드 {args.threads}개)")
    train_tokenizer(
        corpus_path=args.corpus,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        threads=args.threads,
    )
    print(f"생성 완료: {final_model}")
    print(f"생성 완료: {final_vocab}")


if __name__ == "__main__":
    main()
