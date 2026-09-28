"""단일 GPU 0.8B pretraining. 시작 명령은 scripts/server.py train."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Subset

from src.model.causal_lm import CausalLM, ModelConfig
from src.training.pretrain_dataset import PackedPretrainDataset
from src.training.progress import TrainingProgress
from tqdm import tqdm
from src.training.recovery import (Checkpoints, CursorBatchSampler, RunLock,
                                   restore_rng, sha256)
from src.utils.artifacts import save_state

ROOT = Path(__file__).resolve().parents[2]
# 이 정확한 이전 버전에서 변경된 것은 진행률/로그뿐이다. 이전 run의 계약은 보존하고
# 실행 소스 해시는 started 로그에 별도로 남긴다. 그 외 임의 코드 변경은 계속 거부한다.
PROGRESS_COMPATIBLE_TRAIN_SHA256 = "9f4059e1b5a5377d9e4f014d99d314998218a3f181c57588c48b085067882283"


def reporting_compatible_contract(contract, output):
    manifest = output / "run.json"
    if manifest.exists():
        original = json.loads(manifest.read_text(encoding="utf-8"))
        candidate = {**contract, "code": {**contract["code"],
                    "src/training/train.py": PROGRESS_COMPATIBLE_TRAIN_SHA256}}
        if original == candidate:
            print("[재개] 진행률 표시 추가 전 체크포인트와 학습 계약이 호환됩니다", flush=True)
            return original
    return contract


def verify_data(data_root, tokenizer_path, model_config):
    """각 실행에서 전체 bin의 SHA-256/range를 확인한다. 경로는 상대 경로로 식별한다."""
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if (tokenizer.vocab_size(), tokenizer.unk_id(), tokenizer.bos_id(),
        tokenizer.eos_id(), tokenizer.pad_id()) != (model_config.vocab_size, 0, 1, 2, 3):
        raise ValueError("토크나이저 vocab 또는 특수 ID가 학습 계약과 다릅니다")
    meta = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    if meta["dtype"] != "uint16" or meta["vocab_size"] != tokenizer.vocab_size() or meta["eos_id"] != 2:
        raise ValueError("토큰 메타데이터가 모델과 다릅니다")
    fingerprints = {"tokenizer_sha256": sha256(tokenizer_path), "bins": {}}
    for split in ("train", "val"):
        files = sorted((data_root / split).glob("*.bin"))
        expected = {k for k in meta["shards"] if k.startswith(split + "/")}
        if not files or {split + "/" + p.stem for p in files} != expected:
            raise ValueError(f"{split}: 메타데이터와 bin 목록이 다릅니다")
        for path in files:
            key = split + "/" + path.stem
            before = path.stat()
            if before.st_size != 2*meta["shards"][key]["tokens"]:
                raise ValueError(f"토큰 파일 크기 불일치: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(8*1024*1024), b""):
                    values = np.frombuffer(block, dtype="<u2")
                    if len(values) and int(values.max()) >= tokenizer.vocab_size():
                        raise ValueError(f"vocab 범위 밖 토큰: {path}")
                    digest.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"검사 중 데이터가 변경되었습니다: {path}")
            fingerprints["bins"][key] = {"bytes": after.st_size, "sha256": digest.hexdigest()}
            print(f"[데이터 확인] {key}: {after.st_size // 2:,} tokens", flush=True)
    return fingerprints


def learning_rate(tokens_seen, target_tokens, cfg):
    warmup = max(1, int(target_tokens*cfg["warmup_ratio"]))
    # scheduler 객체 대신 이 함수+불변 설정+tokens_seen 전체를 저장한다.
    if tokens_seen < warmup:
        return cfg["learning_rate"] * min(1.0, (tokens_seen + 1)/warmup)
    fraction = min(1.0, (tokens_seen-warmup)/max(1, target_tokens-warmup))
    multiplier = cfg["min_lr_ratio"] + (1-cfg["min_lr_ratio"])*0.5*(1+math.cos(math.pi*fraction))
    return cfg["learning_rate"]*multiplier


def optimizer_for(model, cfg, device):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], betas=tuple(cfg["betas"]), fused=device.type == "cuda")


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, should_stop):
    model.eval()
    losses, tokens = 0.0, 0
    try:
        for batch in loader:
            if should_stop():
                return None
            x, y = (batch[k].to(device, non_blocking=True) for k in ("input_ids", "labels"))
            with autocast(device):
                loss = model(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("validation loss가 유한하지 않습니다")
            losses += loss.item()*y.numel()
            tokens += y.numel()
        return losses/tokens
    finally:
        model.train()


def train(config, *, device_name="cuda", max_steps=None, deadline=None, root=ROOT):
    started = time.monotonic()
    cfg, mc = config["training"], ModelConfig(**config["model"])
    root = Path(root)
    data_root, tokenizer_path = root / config["data_root"], root / config["tokenizer"]
    output = root / config["output_dir"]
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 CUDA GPU와 호환되는 NVIDIA 드라이버가 필요합니다")
        prop = torch.cuda.get_device_properties(device)
        print(f"[GPU] {prop.name}, {prop.total_memory/2**30:.1f} GiB", flush=True)
        print("[Attention] " + ("Flash SDPA" if torch.backends.cuda.is_flash_attention_available()
                               else "cuDNN SDPA") if mc.flash_attention else "[Attention] math SDPA", flush=True)
        if mc.parameter_count() > 100_000_000 and prop.total_memory < 40*2**30:
            raise RuntimeError("본학습은 48GB급 서버를 대상으로 합니다. 작은 GPU에서는 --smoke를 사용하세요")
        torch.backends.cuda.matmul.allow_tf32 = True
    elif mc.parameter_count() > 100_000_000:
        raise RuntimeError("본학습 CPU 실행은 지원하지 않습니다. --smoke --device cpu를 사용하세요")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg["seed"])
    seq, micro = mc.sequence_length, cfg["micro_batch_size"]
    effective = cfg["effective_batch_tokens"]
    if micro <= 0 or effective <= 0 or effective % (micro*seq):
        raise ValueError("effective_batch_tokens는 micro_batch_size * sequence_length의 양의 배수여야 합니다")
    if not 0 < cfg["warmup_ratio"] < 1 or not 0 < cfg["min_lr_ratio"] <= 1:
        raise ValueError("warmup_ratio/min_lr_ratio 설정이 잘못되었습니다")
    if cfg["learning_rate"] <= 0 or cfg["max_grad_norm"] <= 0:
        raise ValueError("learning_rate와 max_grad_norm은 양수여야 합니다")
    for key in ("save_every_seconds", "eval_every_tokens", "eval_tokens", "log_every_steps"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    if cfg["session_hours"] <= 0 or not 0 <= cfg["stop_margin_seconds"] < cfg["session_hours"]*3600:
        raise ValueError("session_hours/stop_margin_seconds 설정이 잘못되었습니다")
    deadline = deadline or time.time() + cfg["session_hours"]*3600
    stop_file = root / ".server-stop"
    stopping = {"requested": False}
    old_handlers = {}
    def request_stop(signum, frame):
        stopping["requested"] = True
    def should_stop():
        return (stopping["requested"] or stop_file.exists()
                or time.time() >= deadline-cfg["stop_margin_seconds"])
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, request_stop)
    try:
        fingerprints = verify_data(data_root, tokenizer_path, mc)
        dataset = PackedPretrainDataset(data_root / "train", seq)
        validation = PackedPretrainDataset(data_root / "val", seq)
        requested = cfg["target_tokens"]
        target = (len(dataset)*seq if requested is None else int(requested)//seq*seq)
        if target <= 0:
            raise ValueError("target_tokens는 최소 sequence_length 이상이어야 합니다")
        # save/eval/session 주기는 재개 시 변경 가능. 학습 경로를 바꾸는 설정은 고정.
        keys = ("seed", "micro_batch_size", "effective_batch_tokens", "learning_rate",
                "min_lr_ratio", "warmup_ratio", "weight_decay", "betas", "max_grad_norm")
        contract = {"schema": 1, "model": asdict(mc), "training": {k: cfg[k] for k in keys},
                    "target_tokens": target, "data": fingerprints,
                    "precision": "bf16-mixed" if device.type == "cuda" else "fp32",
                    "torch": torch.__version__.split("+")[0], "numpy": np.__version__,
                    "scheduler": "token-cosine-v1", "sampler": "epoch-randperm-v1",
                    "code": {name: sha256(ROOT/name) for name in (
                        "src/model/causal_lm.py", "src/training/train.py",
                        "src/training/recovery.py", "src/training/pretrain_dataset.py")}}
        actual_train_sha256 = contract["code"]["src/training/train.py"]
        contract = reporting_compatible_contract(contract, output)
        checkpoints = Checkpoints(output, contract, cfg["keep_checkpoints"])
        restored = checkpoints.load_latest()
        model = CausalLM(mc).to(device)
        optimizer = optimizer_for(model, cfg, device)
        progress = {"step": 0, "tokens_seen": 0, "sequences_seen": 0,
                    "next_eval_tokens": cfg["eval_every_tokens"], "best_val_loss": None}
        if restored:
            model.load_state_dict(restored["model"], strict=True)
            optimizer.load_state_dict(restored["optimizer"])
            progress = restored["progress"]
            if progress["tokens_seen"] != progress["sequences_seen"]*seq:
                raise RuntimeError("체크포인트의 데이터 cursor와 tokens_seen이 다릅니다")
            restore_rng(restored["rng"], device)
            del restored
        print(f"[모델] {sum(p.numel() for p in model.parameters()):,} parameters; "
              f"목표 {target:,} tokens; 누적 {progress['tokens_seen']:,}; "
              f"micro={micro}; accumulation={effective//(micro*seq)}", flush=True)
        remaining = (target-progress["tokens_seen"])//seq
        if remaining <= 0:
            print("[완료] 목표 토큰에 이미 도달했습니다", flush=True)
            return progress
        # 전용 generator로 DataLoader iterator 생성이 모델 RNG를 소모하지 않게 한다.
        sampler = CursorBatchSampler(len(dataset), micro, cfg["seed"],
                                      progress["sequences_seen"], remaining)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=cfg["num_workers"],
                            pin_memory=device.type == "cuda", persistent_workers=cfg["num_workers"] > 0,
                            generator=torch.Generator().manual_seed(cfg["seed"]+100))
        val_count = min(len(validation), max(1, cfg["eval_tokens"]//seq))
        val_indices = np.linspace(0, len(validation)-1, val_count, dtype=np.int64).tolist()
        val_loader = DataLoader(Subset(validation, val_indices), batch_size=micro, num_workers=0,
                                generator=torch.Generator().manual_seed(cfg["seed"]+101))
        iterator = iter(loader)
        output.mkdir(parents=True, exist_ok=True)
        log = (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
        # 데이터 해시 검사·모델 생성·첫 checkpoint 시간은 처리량 ETA에서 제외한다.
        # 첫 optimizer step부터 실제 학습 속도를 측정해야 초기 ETA가 과대해지지 않는다.
        display = TrainingProgress(target, progress["tokens_seen"], time.monotonic(), deadline,
                                   cfg["stop_margin_seconds"])
        def record(event, **values):
            row = {"event": event, "time": time.time(), **progress,
                   **display.snapshot(progress["tokens_seen"]), **values}
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            save_state(output / "status.json", row)
        max_save_seconds = checkpoints.save(model, optimizer, progress, device)
        last_save = time.monotonic()
        steps_done = 0
        model.train()
        try:
            record("started", pid=os.getpid(), actual_train_sha256=actual_train_sha256)
            while progress["tokens_seen"] < target:
                if should_stop() or (max_steps is not None and steps_done >= max_steps):
                    break
                # 저장 시간과 step 여유를 확보하되, 데이터/모델 초기화 시간도 세션에 포함.
                if time.time() + 2*max_save_seconds + 60 >= deadline:
                    break
                update_tokens = min(effective, target-progress["tokens_seen"])
                rate = learning_rate(progress["tokens_seen"] + update_tokens//2, target, cfg)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                consumed, step_loss = 0, 0.0
                before_step = time.monotonic()
                while consumed < update_tokens:
                    batch = next(iterator)
                    x, y = (batch[k].to(device, non_blocking=True) for k in ("input_ids", "labels"))
                    with autocast(device):
                        loss = model(x, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("train loss가 유한하지 않습니다. 마지막 정상 저장본을 보존합니다")
                    fraction = y.numel()/update_tokens
                    (loss*fraction).backward()
                    consumed += y.numel()
                    step_loss += loss.detach().item()*fraction
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"],
                                                          error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_seconds = time.monotonic()-before_step
                progress["step"] += 1
                progress["tokens_seen"] += consumed
                progress["sequences_seen"] += consumed//seq
                steps_done += 1
                stats = display.update(progress["tokens_seen"], step_loss, rate, consumed/step_seconds)
                # status는 매 optimizer step 갱신하고, 장기 metrics 로그는 지정 간격으로 남긴다.
                save_state(output / "status.json", {"event": "train", "time": time.time(),
                           **progress, **stats, "loss": step_loss, "lr": rate,
                           "tokens_per_second": consumed/step_seconds})
                if steps_done == 1 or progress["step"] % cfg["log_every_steps"] == 0:
                    values = {"loss": step_loss, "lr": rate, "grad_norm": float(grad_norm),
                              "tokens_per_second": consumed/step_seconds, "step_seconds": step_seconds}
                    if device.type == "cuda":
                        values["peak_vram_gib"] = torch.cuda.max_memory_allocated(device)/2**30
                    record("train", **values)
                    eta = "측정 중" if stats["eta_hours"] is None else f"{stats['eta_hours']:.1f}h"
                    tqdm.write(f"{stats['progress_percent']:.2f}% step={progress['step']} "
                               f"tokens={progress['tokens_seen']:,}/{target:,} "
                               f"loss={step_loss:.4f} lr={rate:.3g} "
                               f"tok/s={consumed/step_seconds:,.0f} ETA={eta}")
                if time.monotonic()-last_save >= cfg["save_every_seconds"]:
                    duration = checkpoints.save(model, optimizer, progress, device)
                    max_save_seconds = max(max_save_seconds, duration)
                    last_save = time.monotonic()
                    record("checkpoint", save_seconds=duration)
                if progress["tokens_seen"] >= progress["next_eval_tokens"] and not should_stop():
                    value = evaluate(model, val_loader, device, should_stop)
                    if value is not None:
                        best = progress["best_val_loss"]
                        progress["best_val_loss"] = value if best is None else min(best, value)
                        progress["next_eval_tokens"] = (progress["tokens_seen"]//cfg["eval_every_tokens"]+1)*cfg["eval_every_tokens"]
                        record("validation", val_loss=value, val_tokens=val_count*seq)
                        print(f"[검증] loss={value:.4f}", flush=True)
            duration = checkpoints.save(model, optimizer, progress, device)
            record("completed" if progress["tokens_seen"] >= target else "paused",
                   save_seconds=duration, session_seconds=time.monotonic()-started)
            print("[종료] 같은 명령으로 최신 정상 체크포인트에서 이어갈 수 있습니다", flush=True)
            return progress
        finally:
            display.close()
            log.close()
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain_800m.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--ready-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if args.smoke:
        config["model"].update(hidden_size=64, num_layers=2, intermediate_size=128,
                               num_attention_heads=4, num_key_value_heads=2,
                               sequence_length=128, loss_chunk_tokens=128)
        config["training"].update(micro_batch_size=2, effective_batch_tokens=512,
                                  target_tokens=4096, eval_tokens=512, eval_every_tokens=2048,
                                  num_workers=0, log_every_steps=1)
        config["output_dir"] = "checkpoints/smoke_800m_pipeline"
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.micro_batch is not None:
        config["training"]["micro_batch_size"] = args.micro_batch
    if args.workers is not None:
        config["training"]["num_workers"] = args.workers
    if args.hours is not None:
        config["training"]["session_hours"] = args.hours
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    with RunLock(ROOT / ".training.lock"):
        (ROOT / ".server-stop").unlink(missing_ok=True)
        if args.ready_file:
            save_state(Path(args.ready_file), {"pid": os.getpid()})
        train(config, device_name=args.device, max_steps=args.max_steps, deadline=args.deadline)


if __name__ == "__main__":
    main()
