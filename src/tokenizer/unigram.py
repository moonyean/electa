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
import json
import os
import random
from pathlib import Path

import sentencepiece as spm


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
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)
    temporary_path.replace(checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> dict | None:
    """기존 체크포인트가 있으면 읽는다."""
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_corpus(
    input_dir: Path,
    corpus_path: Path,
    checkpoint_path: Path,
    max_documents: int,
    max_characters: int,
    seed: int,
    rebuild: bool = False,
) -> tuple[int, int]:
    """샤드를 순회해 학습용 텍스트를 만들고 샤드마다 진행 상황을 저장한다."""
    if max_documents <= 0 or max_characters <= 0:
        raise ValueError("max_documents와 max_characters는 양수여야 합니다.")

    files = get_input_files(input_dir, seed)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if rebuild:
        for path in (corpus_path, checkpoint_path):
            if path.exists():
                path.unlink()

    state = load_checkpoint(checkpoint_path)
    file_names = [str(path.resolve()) for path in files]
    start_index = 0
    document_count = 0
    character_count = 0
    corpus_bytes = 0

    if state is not None:
        if state.get("files") != file_names or state.get("seed") != seed:
            raise RuntimeError(
                "기존 체크포인트의 입력 파일 또는 seed가 현재 실행과 다릅니다. "
                "--rebuild-corpus로 새로 시작하세요."
            )
        if state.get("completed", False):
            print(f"완료된 학습용 말뭉치를 재사용합니다: {corpus_path}")
            return int(state["documents"]), int(state["characters"])
        if not corpus_path.exists():
            raise FileNotFoundError("체크포인트는 있지만 학습용 말뭉치 파일이 없습니다.")

        # 이전 샤드까지 확실히 저장된 위치로 잘라서 중간 샤드의 부분 기록을 제거한다.
        corpus_bytes = int(state["corpus_bytes"])
        with corpus_path.open("r+b") as file:
            file.truncate(corpus_bytes)
        start_index = int(state["next_file_index"])
        document_count = int(state["documents"])
        character_count = int(state["characters"])
        print(f"체크포인트에서 재개: {start_index}/{len(files)}개 샤드 완료")

    mode = "a" if start_index > 0 else "w"
    with corpus_path.open(mode, encoding="utf-8", newline="\n") as output:
        for file_index in range(start_index, len(files)):
            path = files[file_index]
            print(f"[{file_index + 1}/{len(files)}] 처리 중: {path.name}", flush=True)
            with open_jsonl(path) as source:
                for line in source:
                    if document_count >= max_documents or character_count >= max_characters:
                        output.flush()
                        state = {
                            "version": 1,
                            "completed": True,
                            "files": file_names,
                            "seed": seed,
                            "next_file_index": file_index,
                            "documents": document_count,
                            "characters": character_count,
                            "corpus_bytes": corpus_path.stat().st_size,
                        }
                        write_checkpoint(checkpoint_path, state)
                        return document_count, character_count
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(record.get("text", "")).strip()
                    if not text:
                        continue
                    text = text[: max_characters - character_count]
                    output.write(text.replace("\n", " ") + "\n")
                    document_count += 1
                    character_count += len(text)

            # 샤드 하나가 끝날 때마다 안전한 복구 지점을 남긴다.
            output.flush()
            state = {
                "version": 1,
                "completed": False,
                "files": file_names,
                "seed": seed,
                "next_file_index": file_index + 1,
                "documents": document_count,
                "characters": character_count,
                "corpus_bytes": corpus_path.stat().st_size,
            }
            write_checkpoint(checkpoint_path, state)

    state["completed"] = True
    write_checkpoint(checkpoint_path, state)
    return document_count, character_count


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

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(temporary_prefix),
        model_type="unigram",
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
