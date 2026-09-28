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
import sys
import time
from array import array
from pathlib import Path

import sentencepiece as spm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import file_identity, save_state


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
    save_state(path, value)


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
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents는 양수여야 합니다.")
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
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                skipped_count += 1
                continue
            text = text.strip()

            token_ids = tokenizer.encode(text, out_type=int)
            token_ids.append(eos_id)
            if max(token_ids, default=0) > 65535:
                raise ValueError("uint16 범위를 벗어난 토큰 ID가 발견되었습니다.")

            # .bin: 토큰 ID를 16비트 정수로 저장한다.
            token_array = array("H", token_ids)
            if sys.byteorder != "little":
                token_array.byteswap()
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

    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max-documents는 양수여야 합니다.")

    if not args.tokenizer.exists():
        raise FileNotFoundError(f"토크나이저 파일이 없습니다: {args.tokenizer.resolve()}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    print(f"토크나이저: {args.tokenizer.resolve()}")
    print(f"어휘 크기: {tokenizer.vocab_size():,}")
    print(f"EOS ID: {tokenizer.eos_id()}")

    split_shards = {split: get_shards(args.input_root / split) for split in ("train", "val")}
    for split, shards in split_shards.items():
        stems = [p.name.removesuffix(".gz").removesuffix(".jsonl") for p in shards]
        if len(stems) != len(set(stems)):
            raise ValueError(f"출력 이름이 중복되는 입력 샤드가 있습니다: {split}")
    settings = {
        "tokenizer": file_identity(args.tokenizer),
        "inputs": {split: [file_identity(p) for p in shards] for split, shards in split_shards.items()},
        "output_root": str(args.output_root.resolve()),
        "max_documents": args.max_documents,
    }
    checkpoint = load_json(args.checkpoint) if args.checkpoint.exists() and not args.overwrite else None
    if checkpoint is not None:
        if checkpoint.get("version") != 2 or checkpoint.get("settings") != settings:
            raise RuntimeError("입력/토크나이저/설정이 다른 체크포인트입니다. 별도 출력 경로나 --overwrite가 필요합니다.")
    else:
        if not args.overwrite and any(args.output_root.glob("*/*.bin")):
            raise FileExistsError("체크포인트 없는 기존 토큰 결과를 덮어쓰지 않습니다.")
        checkpoint = {"version": 2, "settings": settings, "splits": {}}
    # 다른 실행의 샤드가 Dataset의 *.bin 검색에 섞이지 않도록 먼저 거부한다.
    for split, shards in split_shards.items():
        expected = {p.name.removesuffix(".gz").removesuffix(".jsonl") for p in shards}
        for path in (args.output_root / split).glob("*"):
            if path.suffix in (".bin", ".idx") and path.stem not in expected:
                raise RuntimeError(f"현재 입력에 없는 이전 산출물이 있습니다. 별도 출력 경로를 사용하세요: {path}")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["completed"] = False
    save_json(args.checkpoint, checkpoint)
    metadata_path = args.output_root / "metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()
    all_stats = {}

    for split in ("train", "val"):
        input_dir = args.input_root / split
        output_dir = args.output_root / split
        shards = split_shards[split]
        completed = checkpoint["splits"].setdefault(split, {})
        print(f"\n[{split}] {len(shards)}개 샤드 처리 시작")

        for shard in shards:
            key = str(shard.resolve())
            existing = completed.get(key)
            stem = shard.name.removesuffix(".gz").removesuffix(".jsonl")
            final_bin = output_dir / f"{stem}.bin"
            final_idx = output_dir / f"{stem}.idx"
            if (existing and final_bin.exists() and final_idx.exists() and not args.overwrite
                    and file_identity(final_bin) == existing.get("bin_identity")
                    and file_identity(final_idx) == existing.get("idx_identity")):
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
            stats["bin_identity"] = file_identity(final_bin)
            stats["idx_identity"] = file_identity(final_idx)
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
        "byte_order": "little",
        "settings": settings,
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
