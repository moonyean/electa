"""exact dedup까지는 끝난 상태에서 MinHash dedup만 이어서 진행한다
(디스크 부족으로 minhash 단계에서 죽었던 것 재개용)."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")


def main() -> None:
    from data.fineWeb2_preprocess import minhash_deduplicate

    ROOT = Path(".")
    WORK_DIR = ROOT / "data/interim/pretrain_merge"
    EXACT = WORK_DIR / "exact_deduplicated"
    NEW_OUTPUT = ROOT / "data/processed/pretrain_v2"
    OLD_OUTPUT = ROOT / "data/processed/pretrain"
    TASKS = 12

    if not any(EXACT.glob("*.jsonl.gz")):
        raise RuntimeError(f"exact dedup 결과가 없습니다: {EXACT.resolve()}")

    print("[1/1] MinHash dedup (샤드 간 유사 중복 제거)", flush=True)
    minhash_deduplicate(EXACT, NEW_OUTPUT, WORK_DIR, TASKS)
    if not any(NEW_OUTPUT.glob("*.jsonl.gz")):
        raise RuntimeError(f"minhash dedup 결과가 비어 있습니다: {NEW_OUTPUT.resolve()}")
    shutil.rmtree(WORK_DIR / "minhash_signatures", ignore_errors=True)
    shutil.rmtree(WORK_DIR / "minhash_buckets", ignore_errors=True)
    shutil.rmtree(WORK_DIR / "minhash_remove_ids", ignore_errors=True)
    shutil.rmtree(EXACT, ignore_errors=True)

    shutil.rmtree(OLD_OUTPUT, ignore_errors=True)
    NEW_OUTPUT.rename(OLD_OUTPUT)

    print("DONE", flush=True)


if __name__ == "__main__":
    if os.name == "nt" and os.environ.get("PYTHONUTF8") != "1":
        os.environ["PYTHONUTF8"] = "1"
        sys.exit(subprocess.call([sys.executable] + sys.argv, env=os.environ))
    main()
