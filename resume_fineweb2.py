"""마지막 2개 샤드(001_00000, 001_00001)를 11분할(22파일)해서 재처리하고,
이미 끝난 8개 + 저장해둔 000_00003 샤드까지 전부 합쳐서 exact/minhash dedup을 진행하는
재개용 스크립트. 조각을 잘게 쪼개서 중간에 죽어도 잃는 양을 줄이고,
동시 실행(workers)은 메모리 안전선으로 따로 제한한다."""

import os
import subprocess
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> None:
    from data.fineWeb2_preprocess import (exact_deduplicate, minhash_deduplicate, preprocess,
                                         stage, retire_stage, snapshot, read_state, save_state)

    ROOT = Path(__file__).resolve().parent
    WORK_DIR = ROOT / "data/interim/pretrain"
    SPLIT_INPUT = WORK_DIR / "split_remaining"
    SAVED_8 = WORK_DIR / "_saved_8files"
    SAVED_SHARD = WORK_DIR / "_saved_000_00003"
    PREPROCESSED = WORK_DIR / "preprocessed"
    MERGED = WORK_DIR / "merged_preprocessed"
    STATE = WORK_DIR / "pipeline_state"
    EXACT = WORK_DIR / "exact_deduplicated"
    OUTPUT = ROOT / "data/processed/pretrain"
    PREPROCESS_TASKS = 22
    PREPROCESS_WORKERS = 12
    DEDUP_TASKS = 12

    print("[1/4] 마지막 2개 샤드(22조각) 전처리 (FTFY + 반복탐지 포함)")
    preprocess(SPLIT_INPUT, PREPROCESSED, PREPROCESS_TASKS, PREPROCESS_WORKERS)
    print("[2/4] 이미 완료된 결과물 합치기 (원본 보존)")
    saved_contract = {str(p): snapshot(p) for p in (SAVED_8, SAVED_SHARD)}
    saved_manifest = STATE / "saved_inputs.json"
    if saved_manifest.exists() and read_state(saved_manifest) != saved_contract:
        raise RuntimeError("저장된 샤드 목록이 바뀌었습니다. 새 작업 폴더를 사용하세요.")
    save_state(saved_manifest, saved_contract)
    with stage("merge_saved", PREPROCESSED, MERGED, STATE, 1) as logs:
        if logs is not None:
            MERGED.mkdir(parents=True, exist_ok=True)
            for group, source in enumerate((PREPROCESSED, SAVED_8, SAVED_SHARD)):
                for path in sorted(source.glob("*.jsonl.gz")):
                    dest = MERGED / f"{group}_{path.name}"
                    partial = dest.with_suffix(dest.suffix + ".part")
                    if partial.exists():
                        partial.unlink()
                    try:
                        os.link(path, partial)
                    except OSError:
                        shutil.copy2(path, partial)
                    partial.replace(dest)

    print("[3/4] Exact dedup")
    exact_deduplicate(MERGED, EXACT, WORK_DIR, DEDUP_TASKS, PREPROCESS_WORKERS)
    retire_stage(STATE, "preprocess", PREPROCESSED)
    retire_stage(STATE, "merge_saved", MERGED)
    for name in ("exact_signatures", "exact_duplicates"):
        shutil.rmtree(WORK_DIR / name, ignore_errors=True)

    print("[4/4] MinHash fuzzy dedup")
    minhash_deduplicate(EXACT, OUTPUT, WORK_DIR, DEDUP_TASKS, PREPROCESS_WORKERS)
    retire_stage(STATE, "exact", EXACT)
    for name in ("minhash_signatures", "minhash_buckets", "minhash_remove_ids"):
        shutil.rmtree(WORK_DIR / name, ignore_errors=True)

    print("DONE")


if __name__ == "__main__":
    if os.name == "nt" and os.environ.get("PYTHONUTF8") != "1":
        os.environ["PYTHONUTF8"] = "1"
        sys.exit(subprocess.call([sys.executable] + sys.argv, env=os.environ))
    main()
