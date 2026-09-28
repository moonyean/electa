"""대용량 작업의 입력 식별과 원자적 상태 저장."""

import hashlib
import json
import os
from pathlib import Path


def file_identity(path: Path) -> dict:
    path = path.resolve()
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"읽는 동안 입력이 변경되었습니다: {path}")
    return {"path": str(path), "size": after.st_size, "sha256": digest}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_state(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
