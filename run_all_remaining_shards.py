"""남은 4개 샤드(000_00003, 000_00004, 001_00000, 001_00001)를 순서대로
11조각 분할 -> 전처리+dedup -> 검증 -> 다음 샤드로 자동 진행한다.
중간에 죽으면(권한 에러, 인터럽트 등) 자동으로 최대 3번까지 재시도한다."""

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(".")
RAW_DIR = ROOT / "data/raw/pretrain/HuggingFaceFW/fineweb-2/data/kor_Hang/train"
N_PARTS = 11
MAX_RETRIES = 3
TIMEOUT_SEC = 5 * 3600  # 샤드 하나에 5시간 이상 걸리면 뭔가 잘못된 것으로 보고 재시도

SHARDS = ["000_00003", "000_00004", "001_00000", "001_00001"]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def split_shard(shard_name: str, input_dir: Path) -> None:
    import pyarrow.parquet as pq
    import pyarrow as pa

    src = RAW_DIR / f"{shard_name}.parquet"
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    writers = [pq.ParquetWriter(input_dir / f"{shard_name}_p{i:02d}.parquet", schema) for i in range(N_PARTS)]
    batch_i = 0
    for batch in pf.iter_batches(batch_size=2000):
        writers[batch_i % N_PARTS].write_table(pa.Table.from_batches([batch]))
        batch_i += 1
    for w in writers:
        w.close()


def run_shard(shard_name: str) -> bool:
    base = ROOT / f"data/interim/_shard_{shard_name}"
    input_dir = base / "input"
    work_dir = base / "work"
    output_dir = base / "output"
    log_file = base / "run.log"

    for attempt in range(1, MAX_RETRIES + 1):
        log(f"{shard_name}: 시도 {attempt}/{MAX_RETRIES} 시작")
        shutil.rmtree(base, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        split_shard(shard_name, input_dir)

        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.run(
                [
                    sys.executable,
                    "src/data/fineWeb2_preprocess.py",
                    "--input", str(input_dir),
                    "--work-dir", str(work_dir),
                    "--output", str(output_dir),
                    "--tasks", str(N_PARTS),
                    "--workers", str(N_PARTS),
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=None,
            )

        output_files = list(output_dir.glob("*.jsonl.gz"))
        if proc.returncode == 0 and output_files:
            log(f"{shard_name}: 성공 ({len(output_files)}개 출력 파일)")
            shutil.rmtree(input_dir, ignore_errors=True)
            return True

        log(f"{shard_name}: 실패 (exit={proc.returncode}, 출력파일={len(output_files)}개) - 재시도")

    log(f"{shard_name}: {MAX_RETRIES}번 다 실패. 중단.")
    return False


def main() -> None:
    for shard_name in SHARDS:
        ok = run_shard(shard_name)
        if not ok:
            log("ALL_STOPPED_ON_FAILURE")
            return
    log("ALL_SHARDS_DONE")


if __name__ == "__main__":
    main()
