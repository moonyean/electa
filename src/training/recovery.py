"""원자적 체크포인트, 소비 위치 기반 셔플, 프로세스 중복 실행 방지."""

from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import time
import uuid

import numpy as np
import torch

from src.utils.artifacts import save_state


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


class RunLock(AbstractContextManager):
    """OS가 프로세스 종료 시 해제하는 잠금. 남은 lock 파일은 정상이다."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            self.file = None
            raise RuntimeError(f"이미 학습 또는 전송 작업이 실행 중입니다: {self.path}") from exc
        return self

    def __exit__(self, *args):
        if self.file:
            if os.name == "nt":
                import msvcrt
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()


class CursorBatchSampler:
    """Prefetch와 무관하게, 체크포인트에 저장한 소비한 sequence 수로 재개한다.

    epoch별 전체 permutation을 재현하고 cursor 앞의 데이터는 로드하지 않는다.
    마지막 batch와 epoch 경계에서도 샘플을 버리지 않는다.
    """
    def __init__(self, size, batch_size, seed, cursor, count):
        if size <= 0 or batch_size <= 0 or cursor < 0 or count < 0:
            raise ValueError("invalid sampler arguments")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.cursor, self.count = cursor, count

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        cursor, remaining = self.cursor, self.count
        cached_epoch, order = None, None
        while remaining:
            batch = []
            for _ in range(min(self.batch_size, remaining)):
                epoch, offset = divmod(cursor, self.size)
                if epoch != cached_epoch:
                    generator = torch.Generator().manual_seed(self.seed + epoch)
                    order = torch.randperm(self.size, generator=generator)
                    cached_epoch = epoch
                batch.append(int(order[offset]))
                cursor += 1
            remaining -= len(batch)
            yield batch


def capture_rng(device):
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": [name, keys.tolist(), pos, has_gauss, cached],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


class Checkpoints:
    def __init__(self, folder, contract, keep=3):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.keep = keep
        if keep < 2:
            raise ValueError("복구를 위해 keep_checkpoints >= 2가 필요합니다")
        manifest = self.folder / "run.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != contract:
                raise RuntimeError("데이터/모델/학습 설정이 변경되었습니다. 기존 run에는 재개할 수 없습니다. 새 output_dir을 사용하세요.")
        else:
            if any(self.folder.glob("checkpoint-*")):
                raise RuntimeError("run.json 없는 체크포인트 폴더입니다")
            save_state(manifest, contract)
        self.last_step = None

    def committed(self):
        return sorted(p for p in self.folder.glob("checkpoint-*") if p.is_dir())

    def load_latest(self):
        candidates = self.committed()
        for path in reversed(candidates):
            try:
                meta = json.loads((path / "complete.json").read_text(encoding="utf-8"))
                if sha256(path / "state.pt") != meta["sha256"]:
                    raise ValueError("SHA-256 불일치")
                state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
                if not isinstance(state, dict) or not {"contract", "model", "optimizer", "progress", "rng"} <= state.keys():
                    raise ValueError("필수 학습 상태 누락")
                if state["contract"] != self.contract:
                    raise ValueError("체크포인트 계약 불일치")
                if state["progress"]["step"] != meta["step"]:
                    raise ValueError("step 불일치")
            except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
                print(f"[복구] 손상된 체크포인트 건너뜀: {path.name}: {exc}", flush=True)
                path.rename(path.with_name(".corrupt-" + path.name + "-" + uuid.uuid4().hex[:8]))
                continue
            self.last_step = state["progress"]["step"]
            print(f"[재개] {path.name}", flush=True)
            return state
        if candidates or any(self.folder.glob(".corrupt-*")):
            raise RuntimeError("정상 체크포인트가 없습니다. 처음부터 덮어쓰지 않고 중단합니다.")
        return None

    def save(self, model, optimizer, progress, device):
        if self.last_step == progress["step"]:
            return 0.0
        started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        staging = self.folder / (".saving-" + uuid.uuid4().hex)
        staging.mkdir()
        # 실패한 staging은 보존한다. 완료 표시 없는 디렉터리는 재개 후보가 아니다.
        payload = {"contract": self.contract, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "progress": dict(progress),
                   "rng": capture_rng(device)}
        def tensor_bytes(value):
            if isinstance(value, torch.Tensor):
                return value.numel()*value.element_size()
            if isinstance(value, dict):
                return sum(tensor_bytes(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return sum(tensor_bytes(v) for v in value)
            return 0
        required = int(tensor_bytes(payload)*1.05) + 64*1024*1024
        if shutil.disk_usage(self.folder).free < required:
            raise OSError(f"새 체크포인트용 여유 공간 부족: 최소 {required/1e9:.2f} GB 필요")
        with (staging / "state.pt").open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        save_state(staging / "complete.json", {
            "step": progress["step"], "tokens_seen": progress["tokens_seen"],
            "sha256": sha256(staging / "state.pt")})
        destination = self.folder / f"checkpoint-{progress['step']:012d}-{uuid.uuid4().hex[:8]}"
        staging.rename(destination)
        if os.name != "nt":
            fd = os.open(self.folder, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        self.last_step = progress["step"]
        # 새 정상본이 디스크에 확정된 뒤에만 가장 오래된 전체 상태를 제거한다.
        for old in self.committed()[:-self.keep]:
            if old.resolve().parent != self.folder.resolve():
                raise RuntimeError("체크포인트 정리 경로가 run 밖입니다")
            shutil.rmtree(old)
        elapsed = time.monotonic() - started
        print(f"[저장] {destination.name}, {elapsed:.1f}초", flush=True)
        return elapsed
