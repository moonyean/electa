"""마지막 2개 샤드(001_00000, 001_00001)를 11분할(22파일)해서 재처리하고,
이미 끝난 8개 + 저장해둔 000_00003 샤드까지 전부 합쳐서 exact/minhash dedup을 진행하는
재개용 스크립트. 조각을 잘게 쪼개서 중간에 죽어도 잃는 양을 줄이고,
동시 실행(workers)은 메모리 안전선으로 따로 제한한다."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")


def main() -> None:
    from data.fineWeb2_preprocess import exact_deduplicate, minhash_deduplicate, preprocess

    ROOT = Path(".")
    WORK_DIR = ROOT / "data/interim/pretrain"
    SPLIT_INPUT = WORK_DIR / "split_remaining"
    SAVED_8 = WORK_DIR / "_saved_8files"
    SAVED_SHARD = WORK_DIR / "_saved_000_00003"
    PREPROCESSED = WORK_DIR / "preprocessed"
    EXACT = WORK_DIR / "exact_deduplicated"
    OUTPUT = ROOT / "data/processed/pretrain"
    PREPROCESS_TASKS = 22
    PREPROCESS_WORKERS = 12
    DEDUP_TASKS = 12

    print("[1/4] 마지막 2개 샤드(22조각) 전처리 (FTFY + 반복탐지 포함)")
    preprocess(SPLIT_INPUT, PREPROCESSED, PREPROCESS_TASKS, PREPROCESS_WORKERS)
    shutil.rmtree(SPLIT_INPUT, ignore_errors=True)
    shutil.rmtree(WORK_DIR / "remaining_raw", ignore_errors=True)

    print("[2/4] 이미 완료된 결과물(8개 + 000_00003 샤드) 합치기")
    for saved_dir in (SAVED_8, SAVED_SHARD):
        if saved_dir.exists():
            for saved_file in sorted(saved_dir.glob("*.jsonl.gz")):
                dest = PREPROCESSED / saved_file.name
                shutil.move(str(saved_file), str(dest))
                print("  merged", saved_file.name)
            saved_dir.rmdir()

    print("[3/4] Exact dedup")
    exact_deduplicate(PREPROCESSED, EXACT, WORK_DIR, DEDUP_TASKS)
    shutil.rmtree(PREPROCESSED, ignore_errors=True)
    shutil.rmtree(WORK_DIR / "exact_signatures", ignore_errors=True)
    shutil.rmtree(WORK_DIR / "exact_duplicates", ignore_errors=True)

    print("[4/4] MinHash fuzzy dedup")
    minhash_deduplicate(EXACT, OUTPUT, WORK_DIR, DEDUP_TASKS)
    shutil.rmtree(WORK_DIR / "minhash_signatures", ignore_errors=True)
    shutil.rmtree(WORK_DIR / "minhash_buckets", ignore_errors=True)
    shutil.rmtree(WORK_DIR / "minhash_remove_ids", ignore_errors=True)
    shutil.rmtree(EXACT, ignore_errors=True)

    print("DONE")


if __name__ == "__main__":
    main()
