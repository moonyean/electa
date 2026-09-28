"""남은 4개 샤드(000_00003, 000_00004, 001_00000, 001_00001)를 순서대로
11조각 분할 -> 전처리+dedup -> 검증 -> 다음 샤드로 자동 진행한다.
중간에 죽으면(권한 에러, 인터럽트 등) 자동으로 최대 3번까지 재시도한다."""

from contextlib import ExitStack
import subprocess
import sys
import time
from pathlib import Path
from src.utils.artifacts import file_identity, read_state, save_state

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data/raw/pretrain/HuggingFaceFW/fineweb-2/data/kor_Hang/train"
N_PARTS = 11
MAX_RETRIES = 3

SHARDS = ["000_00003", "000_00004", "001_00000", "001_00001"]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def split_shard(shard_name: str, input_dir: Path) -> None:
    import pyarrow.parquet as pq
    import pyarrow as pa

    src = RAW_DIR / f"{shard_name}.parquet"
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = input_dir.parent / "split.json"
    contract = {"source": file_identity(src), "parts": N_PARTS}
    state = read_state(manifest) if manifest.exists() else None
    if state is None and any(input_dir.glob("*.parquet")):
        raise FileExistsError("분할 기록 없는 기존 입력을 덮어쓰지 않습니다. 새 작업 폴더가 필요합니다.")
    if state and state["contract"] != contract:
        raise RuntimeError("원본 또는 분할 수가 변경되었습니다. 새 작업 폴더가 필요합니다.")
    if state and state.get("completed"):
        for entry in state["outputs"]:
            if file_identity(Path(entry["path"])) != entry:
                raise RuntimeError("완료된 분할 파일이 변경되었습니다.")
        return
    save_state(manifest, {"contract": contract, "completed": False})
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    outputs = [input_dir / f"{shard_name}_p{i:02d}.parquet" for i in range(N_PARTS)]
    with ExitStack() as stack:
        writers = [stack.enter_context(pq.ParquetWriter(str(p) + ".part", schema)) for p in outputs]
        for index, batch in enumerate(pf.iter_batches(batch_size=2000)):
            writers[index % N_PARTS].write_table(pa.Table.from_batches([batch]))
    for path in outputs:
        Path(str(path) + ".part").replace(path)
    save_state(manifest, {"contract": contract, "completed": True,
                          "outputs": [file_identity(p) for p in outputs]})


def run_shard(shard_name: str) -> bool:
    base = ROOT / f"data/interim/_shard_{shard_name}"
    input_dir = base / "input"
    work_dir = base / "work"
    output_dir = base / "output"
    log_file = base / "run.log"

    for attempt in range(1, MAX_RETRIES + 1):
        log(f"{shard_name}: 시도 {attempt}/{MAX_RETRIES} 시작")
        input_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            split_shard(shard_name, input_dir)
        except Exception as error:
            log(f"{shard_name}: 분할 실패, 완료 결과 보존: {error}")
            continue

        with open(log_file, "a", encoding="utf-8") as f:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-X", "utf8",
                    str(ROOT / "src/data/fineWeb2_preprocess.py"),
                    "--input", str(input_dir),
                    "--work-dir", str(work_dir),
                    "--output", str(output_dir),
                    "--tasks", str(N_PARTS),
                    "--workers", str(N_PARTS),
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )

        output_files = list(output_dir.glob("*.jsonl.gz"))
        receipt = work_dir / "pipeline_state/minhash/stage.json"
        if proc.returncode == 0 and output_files and receipt.exists() and read_state(receipt).get("completed"):
            log(f"{shard_name}: 성공 ({len(output_files)}개 출력 파일)")
            return True

        log(f"{shard_name}: 실패 (exit={proc.returncode}, 출력파일={len(output_files)}개) - 재시도")

    log(f"{shard_name}: {MAX_RETRIES}번 다 실패. 중단.")
    return False


def main() -> None:
    for shard_name in SHARDS:
        ok = run_shard(shard_name)
        if not ok:
            log("ALL_STOPPED_ON_FAILURE")
            raise SystemExit(1)
    log("ALL_SHARDS_DONE")


if __name__ == "__main__":
    main()
