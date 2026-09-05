"""학습된 SentencePiece 토크나이저로 전체 데이터를 토큰화하고 이진화한다.

입력::

    data/processed/pretrain_split/train/*.jsonl.gz
    data/processed/pretrain_split/val/*.jsonl.gz

출력::

    data/processed/tokenized/train/00000.bin
    data/processed/tokenized/train/00000.idx
    data/processed/tokenized/val/00000.bin
    data/processed/tokenized/val/00000.idx

`.bin`은 uint16 토큰 ID 배열이고, `.idx`는 각 문서의 시작 토큰 위치와
토큰 길이를 uint64 두 개로 저장한다. 샤드 하나가 끝날 때마다 체크포인트를
저장하므로 중단 후 완료된 샤드는 건너뛰고 이어서 처리할 수 있다.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import time
from array import array
from pathlib import Path

import sentencepiece as spm


def open_jsonl(path: Path):
    """압축 여부에 맞춰 JSONL 파일을 UTF-8 텍스트로 연다."""
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def get_shards(input_dir: Path) -> list[Path]:
    """입력 폴더의 JSONL 샤드를 정렬해서 반환한다."""
    shards = sorted(input_dir.glob("*.jsonl.gz"))
    shards.extend(sorted(input_dir.glob("*.jsonl")))
    if not shards:
        raise FileNotFoundError(f"입력 샤드가 없습니다: {input_dir.resolve()}")
    return shards


def save_json(path: Path, value: dict) -> None:
    """JSON 체크포인트를 임시 파일을 거쳐 안전하게 저장한다."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def tokenize_shard(
    tokenizer: spm.SentencePieceProcessor,
    input_path: Path,
    output_dir: Path,
    max_documents: int | None = None,
) -> dict:
    """하나의 JSONL 샤드를 .bin/.idx로 변환한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.name.removesuffix(".gz").removesuffix(".jsonl")
    final_bin = output_dir / f"{stem}.bin"
    final_idx = output_dir / f"{stem}.idx"
    partial_bin = output_dir / f"{stem}.bin.part"
    partial_idx = output_dir / f"{stem}.idx.part"

    # 중간 파일이 남아 있으면 현재 샤드부터 다시 완전히 처리한다.
    for path in (partial_bin, partial_idx):
        if path.exists():
            path.unlink()

    eos_id = tokenizer.eos_id()
    if eos_id < 0:
        raise RuntimeError("토크나이저에 EOS 토큰이 없습니다.")

    document_count = 0
    token_count = 0
    skipped_count = 0
    started = time.perf_counter()
    with (
        partial_bin.open("wb") as bin_file,
        partial_idx.open("wb") as idx_file,
        open_jsonl(input_path) as source,
    ):
        for line in source:
            if max_documents is not None and document_count >= max_documents:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_count += 1
                continue
            text = str(record.get("text", "")).strip()
            if not text:
                skipped_count += 1
                continue

            token_ids = tokenizer.encode(text, out_type=int)
            token_ids.append(eos_id)
            if max(token_ids, default=0) > 65535:
                raise ValueError("uint16 범위를 벗어난 토큰 ID가 발견되었습니다.")

            # .bin: 토큰 ID를 16비트 정수로 저장한다.
            token_array = array("H", token_ids)
            bin_file.write(token_array.tobytes())
            # .idx: 시작 위치와 문서 토큰 수를 little-endian uint64로 저장한다.
            idx_file.write(struct.pack("<QQ", token_count, len(token_ids)))
            token_count += len(token_ids)
            document_count += 1
            if document_count % 100_000 == 0:
                print(
                    f"  {input_path.name}: {document_count:,}개 문서, "
                    f"{token_count:,}개 토큰",
                    flush=True,
                )

    partial_bin.replace(final_bin)
    partial_idx.replace(final_idx)
    return {
        "input": str(input_path.resolve()),
        "bin": str(final_bin.resolve()),
        "idx": str(final_idx.resolve()),
        "documents": document_count,
        "tokens": token_count,
        "skipped": skipped_count,
        "seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/processed/pretrain_split"),
        help="train/val JSONL 폴더가 있는 입력 루트",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/processed/tokenizer/unigram_32k.model"),
        help="학습된 SentencePiece .model 파일",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/tokenized"),
        help=".bin/.idx 결과를 저장할 루트",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/interim/tokenizer/tokenize_checkpoint.json"),
    )
    parser.add_argument("--max-documents", type=int, default=None, help="테스트용 샤드별 최대 문서 수")
    parser.add_argument("--overwrite", action="store_true", help="기존 체크포인트와 결과를 무시하고 새로 시작")
    args = parser.parse_args()

    if not args.tokenizer.exists():
        raise FileNotFoundError(f"토크나이저 파일이 없습니다: {args.tokenizer.resolve()}")

    if args.overwrite and args.checkpoint.exists():
        args.checkpoint.unlink()

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    print(f"토크나이저: {args.tokenizer.resolve()}")
    print(f"어휘 크기: {tokenizer.vocab_size():,}")
    print(f"EOS ID: {tokenizer.eos_id()}")

    checkpoint = load_json(args.checkpoint) if args.checkpoint.exists() else {
        "version": 1,
        "tokenizer": str(args.tokenizer.resolve()),
        "splits": {},
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    all_stats = {}

    for split in ("train", "val"):
        input_dir = args.input_root / split
        output_dir = args.output_root / split
        shards = get_shards(input_dir)
        completed = checkpoint["splits"].setdefault(split, {})
        print(f"\n[{split}] {len(shards)}개 샤드 처리 시작")

        for shard in shards:
            key = str(shard.resolve())
            existing = completed.get(key)
            stem = shard.name.removesuffix(".gz").removesuffix(".jsonl")
            final_bin = output_dir / f"{stem}.bin"
            final_idx = output_dir / f"{stem}.idx"
            if existing and final_bin.exists() and final_idx.exists() and not args.overwrite:
                print(f"건너뜀: {shard.name} (체크포인트 완료)")
                all_stats[f"{split}/{stem}"] = existing
                continue

            print(f"처리 중: {split}/{shard.name}", flush=True)
            stats = tokenize_shard(
                tokenizer=tokenizer,
                input_path=shard,
                output_dir=output_dir,
                max_documents=args.max_documents,
            )
            completed[key] = stats
            all_stats[f"{split}/{stem}"] = stats
            save_json(args.checkpoint, checkpoint)
            print(
                f"완료: {stats['documents']:,}개 문서, "
                f"{stats['tokens']:,}개 토큰, {stats['seconds']:.1f}초",
                flush=True,
            )

    metadata = {
        "tokenizer": str(args.tokenizer.resolve()),
        "vocab_size": tokenizer.vocab_size(),
        "dtype": "uint16",
        "index_dtype": "<uint64,uint64",
        "eos_id": tokenizer.eos_id(),
        "shards": all_stats,
    }
    save_json(args.output_root / "metadata.json", metadata)
    checkpoint["completed"] = True
    save_json(args.checkpoint, checkpoint)
    print(f"\n전체 토큰화·이진화 완료: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
