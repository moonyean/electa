"""data/processed/pretrain을 문서 단위로 train/val(99:1)로 나눈다.
id의 해시값으로 결정론적으로 배정 (재현 가능, 순서에 안 영향받음)."""

import gzip
import hashlib
import json
from pathlib import Path

SRC = Path("data/processed/pretrain")
TRAIN = Path("data/processed/pretrain_split/train")
VAL = Path("data/processed/pretrain_split/val")
VAL_RATIO = 0.01

TRAIN.mkdir(parents=True, exist_ok=True)
VAL.mkdir(parents=True, exist_ok=True)


def is_val(doc_id: str) -> bool:
    h = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16)
    return (h % 10000) < int(VAL_RATIO * 10000)


train_count = 0
val_count = 0

for src_file in sorted(SRC.glob("*.jsonl.gz")):
    train_out = gzip.open(TRAIN / src_file.name, "wt", encoding="utf-8")
    val_out = gzip.open(VAL / src_file.name, "wt", encoding="utf-8")
    try:
        with gzip.open(src_file, "rt", encoding="utf-8") as fh:
            for line in fh:
                doc = json.loads(line)
                if is_val(doc["id"]):
                    val_out.write(line)
                    val_count += 1
                else:
                    train_out.write(line)
                    train_count += 1
    finally:
        train_out.close()
        val_out.close()
    print(f"processed {src_file.name}", flush=True)

print(f"train: {train_count} docs")
print(f"val: {val_count} docs")
print(f"val ratio: {val_count / (train_count + val_count):.4%}")
print("DONE")
