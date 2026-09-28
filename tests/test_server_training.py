"""실제 0.8B를 학습하지 않고 모델 수식·중단/재개·저장 실패를 검증한다."""

import contextlib
import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid
import types

import numpy as np
import sentencepiece as spm
import torch
from torch.nn import functional as F

from src.model.causal_lm import CausalLM, ModelConfig
from src.training.recovery import Checkpoints, CursorBatchSampler, RunLock
from src.training.train import (train, learning_rate, reporting_compatible_contract,
                                PROGRESS_COMPATIBLE_TRAIN_SHA256)
from src.training.progress import progress_stats, TrainingProgress
from scripts import server

ROOT = Path(__file__).resolve().parents[1]


class ServerTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.root = ROOT / ("server_test_" + uuid.uuid4().hex)
        self.root.mkdir()
        self.capture = contextlib.redirect_stdout(io.StringIO())
        self.capture.__enter__()

    def tearDown(self):
        self.capture.__exit__(None, None, None)
        target = self.root.resolve()
        assert target.parent == ROOT and target.name.startswith("server_test_")
        shutil.rmtree(target)

    def config(self):
        config = json.loads((ROOT / "configs/pretrain_800m.json").read_text(encoding="utf-8"))
        folder = self.root / "tokenizer"
        folder.mkdir()
        stream = io.BytesIO()
        spm.SentencePieceTrainer.train(sentence_iterator=iter([
            "한국어 모델 학습 문장입니다", "사람과 대화하는 작은 언어 모델입니다"]*5),
            model_writer=stream, vocab_size=320, hard_vocab_limit=False, byte_fallback=True,
            unk_id=0, bos_id=1, eos_id=2, pad_id=3, num_threads=1, minloglevel=2)
        (folder / "model.model").write_bytes(stream.getvalue())
        tokenizer = spm.SentencePieceProcessor(model_proto=stream.getvalue())
        config["model"].update(vocab_size=tokenizer.vocab_size(), hidden_size=32,
                               num_layers=2, intermediate_size=64, num_attention_heads=4,
                               num_key_value_heads=2, sequence_length=16, loss_chunk_tokens=11,
                               flash_attention=False)
        config["training"].update(micro_batch_size=2, effective_batch_tokens=128,
                                  target_tokens=512, eval_every_tokens=256, eval_tokens=64,
                                  num_workers=0, log_every_steps=1)
        config.update(data_root="tokens", tokenizer="tokenizer/model.model", output_dir="run")
        metadata = {"dtype": "uint16", "vocab_size": tokenizer.vocab_size(), "eos_id": 2, "shards": {}}
        rng = np.random.default_rng(123)
        for split in ("train", "val"):
            data = self.root / "tokens" / split
            data.mkdir(parents=True)
            for i in range(2):
                values = rng.integers(4, tokenizer.vocab_size(), size=257, dtype="<u2")
                values.tofile(data / f"{i:05d}.bin")
                metadata["shards"][f"{split}/{i:05d}"] = {"tokens": len(values)}
        (self.root / "tokens/metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return config

    @staticmethod
    def latest(folder):
        path = sorted(folder.glob("checkpoint-*/state.pt"))[-1]
        return torch.load(path, map_location="cpu", weights_only=True)

    def test_800m_parameter_count_without_allocation(self):
        config = ModelConfig()
        self.assertEqual(config.parameter_count(), 792423936)
        with torch.device("meta"):
            model = CausalLM(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 792423936)

    def test_eta_uses_only_tokens_processed_since_resume(self):
        stats = progress_stats(10000, 6000, 5000, 400, 100)
        self.assertEqual(stats["progress_percent"], 60)
        self.assertEqual(stats["effective_tokens_per_second"], 2.5)
        self.assertEqual(stats["eta_seconds"], 1600)
        self.assertFalse(stats["eta_provisional"])
        self.assertIsNone(progress_stats(10000, 5000, 5000, 10, 100)["eta_seconds"])
        self.assertEqual(progress_stats(10000, 10000, 5000, 400, -1)["eta_seconds"], 0)

    def test_tty_progress_resumes_at_saved_position(self):
        import time
        stream = io.StringIO()
        stream.isatty = lambda: True
        display = TrainingProgress(10000, 5000, time.monotonic()-400,
                                   time.time()+1000, 600, stream)
        try:
            display.update(6000, 2.0, 0.0003, 100)
            display.update(7000, 1.9, 0.0003, 100)
            self.assertEqual(display.bar.n, 7000)
        finally:
            display.close()
        self.assertIn("70%", stream.getvalue())
        self.assertIn("ETA=", stream.getvalue())

    def test_reporting_compatibility_preserves_other_contract_checks(self):
        contract = {"code": {"src/training/train.py": "new", "model.py": "same"},
                    "target_tokens": 1000}
        old = copy.deepcopy(contract)
        old["code"]["src/training/train.py"] = PROGRESS_COMPATIBLE_TRAIN_SHA256
        (self.root / "run.json").write_text(json.dumps(old), encoding="utf-8")
        self.assertEqual(reporting_compatible_contract(contract, self.root), old)
        changed = copy.deepcopy(contract)
        changed["target_tokens"] = 2000
        self.assertEqual(reporting_compatible_contract(changed, self.root), changed)
        changed = copy.deepcopy(contract)
        changed["code"]["model.py"] = "changed"
        self.assertEqual(reporting_compatible_contract(changed, self.root), changed)

    def test_causality_and_chunked_loss_gradients(self):
        c = ModelConfig(vocab_size=40, hidden_size=32, num_layers=2, intermediate_size=64,
                        num_attention_heads=4, num_key_value_heads=2, sequence_length=16,
                        loss_chunk_tokens=3, flash_attention=False)
        torch.manual_seed(1)
        model = CausalLM(c)
        reference = copy.deepcopy(model)
        x, labels = torch.randint(0, 40, (2, 8)), torch.randint(0, 40, (2, 8))
        actual = model(x, labels)
        logits = reference(x)
        expected = F.cross_entropy(logits.reshape(-1, 40), labels.reshape(-1))
        torch.testing.assert_close(actual, expected)
        actual.backward(); expected.backward()
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)
        model.eval()
        altered = x.clone(); altered[:, 4:] = (altered[:, 4:]+1) % 40
        torch.testing.assert_close(model(x)[:, :4], model(altered)[:, :4])

    def test_sampler_resume_across_epoch_and_partial_batch(self):
        all_batches = list(CursorBatchSampler(11, 3, 42, 0, 29))
        full = [x for batch in all_batches for x in batch]
        resumed = [x for batch in CursorBatchSampler(11, 4, 42, 9, 20) for x in batch]
        self.assertEqual(full[9:], resumed)
        self.assertEqual(sorted(full[:11]), list(range(11)))
        self.assertEqual(sorted(full[11:22]), list(range(11)))

    def test_interrupted_training_exactly_matches_uninterrupted(self):
        cfg = self.config()
        cfg["output_dir"] = "continuous"
        train(cfg, device_name="cpu", root=self.root)
        continuous = self.latest(self.root / "continuous")
        cfg["output_dir"] = "resumed"
        train(cfg, device_name="cpu", max_steps=2, root=self.root)
        midpoint = self.latest(self.root / "resumed")
        status = json.loads((self.root / "resumed/status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["progress_percent"], 50)
        self.assertEqual(status["session_tokens"], 256)
        self.assertGreater(status["eta_seconds"], 0)
        self.assertEqual(midpoint["progress"]["sequences_seen"], 16)
        train(cfg, device_name="cpu", root=self.root)
        resumed = self.latest(self.root / "resumed")
        status = json.loads((self.root / "resumed/status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["progress_percent"], 100)
        self.assertEqual(status["session_tokens"], 256)
        self.assertEqual(status["eta_seconds"], 0)
        self.assertEqual(continuous["progress"], resumed["progress"])
        for key in continuous["model"]:
            torch.testing.assert_close(continuous["model"][key], resumed["model"][key], rtol=0, atol=0)
        for key, state in continuous["optimizer"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, resumed["optimizer"]["state"][key][name], rtol=0, atol=0)

    def test_prefetch_cursor_does_not_skip_data(self):
        cfg = self.config()
        cfg["output_dir"] = "serial"
        train(cfg, device_name="cpu", root=self.root)
        expected = self.latest(self.root / "serial")
        cfg["training"]["num_workers"] = 2
        cfg["output_dir"] = "workers"
        train(cfg, device_name="cpu", max_steps=1, root=self.root)
        train(cfg, device_name="cpu", root=self.root)
        actual = self.latest(self.root / "workers")
        for key in expected["model"]:
            torch.testing.assert_close(expected["model"][key], actual["model"][key], rtol=0, atol=0)

    def test_checkpoint_retention_corruption_and_failed_save(self):
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters())
        manager = Checkpoints(self.root / "checkpoints", {"test": True}, keep=3)
        for step in range(5):
            manager.save(model, opt, {"step": step, "tokens_seen": step*10}, torch.device("cpu"))
        self.assertEqual(len(manager.committed()), 3)
        latest = manager.committed()[-1]
        (latest / "state.pt").write_bytes(b"broken")
        restored = manager.load_latest()
        self.assertEqual(restored["progress"]["step"], 3)
        with patch("src.training.recovery.torch.save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                manager.save(model, opt, {"step": 6, "tokens_seen": 60}, torch.device("cpu"))
        self.assertEqual(manager.load_latest()["progress"]["step"], 3)

    def test_all_corrupt_does_not_silently_restart(self):
        model = torch.nn.Linear(2, 2)
        manager = Checkpoints(self.root / "run", {}, keep=3)
        manager.save(model, torch.optim.AdamW(model.parameters()),
                     {"step": 1, "tokens_seen": 4}, torch.device("cpu"))
        (manager.committed()[0] / "state.pt").write_bytes(b"broken")
        with self.assertRaises(RuntimeError):
            manager.load_latest()
        with self.assertRaises(RuntimeError):
            Checkpoints(self.root / "run", {}, keep=3).load_latest()

    def test_data_or_training_contract_change_rejected(self):
        cfg = self.config()
        train(cfg, device_name="cpu", max_steps=1, root=self.root)
        changed = copy.deepcopy(cfg)
        changed["training"]["learning_rate"] *= 2
        with self.assertRaisesRegex(RuntimeError, "설정"):
            train(changed, device_name="cpu", root=self.root)
        p = self.root / "tokens/train/00000.bin"
        a = np.fromfile(p, dtype="<u2"); a[5] = 4; a.tofile(p)
        with self.assertRaisesRegex(RuntimeError, "설정"):
            train(cfg, device_name="cpu", root=self.root)

    def test_partial_final_update_and_finished_run(self):
        cfg = self.config()
        cfg["training"]["target_tokens"] = 304
        result = train(cfg, device_name="cpu", root=self.root)
        self.assertEqual(result["tokens_seen"], 304)
        self.assertEqual(result["step"], 3)
        self.assertEqual(train(cfg, device_name="cpu", root=self.root), result)

    def test_deadline_saves_without_training(self):
        cfg = self.config()
        import time
        result = train(cfg, device_name="cpu", root=self.root, deadline=time.time()+1)
        self.assertEqual(result["step"], 0)
        self.assertTrue(list((self.root / "run").glob("checkpoint-*/state.pt")))

    def test_locks_prevent_second_writer(self):
        with RunLock(self.root / "lock"):
            with self.assertRaises(RuntimeError):
                with server.lock(self.root / "lock"):
                    pass
        with server.lock(self.root / "lock"):
            pass

    def test_transfer_staging_verified_before_install(self):
        # 원격 Linux Python 코드의 파일 검증/교체 로직을 작은 로컬 파일로 실행한다.
        # fcntl의 실제 잠금은 test_locks_prevent_second_writer에서 별도로 확인한다.
        remote = self.root / "remote"
        remote.mkdir()
        (remote / "a.txt").write_text("old", encoding="utf-8")
        local = self.root / "new.txt"
        local.write_text("new", encoding="utf-8")
        rows = [{"path": "a.txt", "size": 3, "sha256": server.digest(local)},
                {"path": "sub/b.txt", "size": 3, "sha256": server.digest(local)}]
        request = {"root": str(remote), "staging": ".upload-test", "mode": "prepare", "files": rows}
        fake = types.SimpleNamespace(flock=lambda *a: None, LOCK_EX=1, LOCK_NB=2)
        def execute(req):
            output = io.StringIO()
            with patch.dict("sys.modules", {"fcntl": fake}), patch("sys.stdin", io.StringIO(json.dumps(req))), contextlib.redirect_stdout(output):
                namespace = {}
                try:
                    exec(server.REMOTE_TRANSFER, namespace)
                finally:
                    if "lock" in namespace:
                        namespace["lock"].close()
            return output.getvalue()
        self.assertEqual(json.loads(execute(request)), rows)
        staging = remote / ".upload-test"
        (staging / "a.txt").write_text("new", encoding="utf-8")
        (staging / "sub/b.txt").write_text("bad", encoding="utf-8")
        request["mode"] = "commit"
        with self.assertRaises(ValueError):
            execute(request)
        self.assertEqual((remote / "a.txt").read_text(), "old")
        (staging / "sub/b.txt").write_text("new", encoding="utf-8")
        execute(request)
        self.assertEqual((remote / "a.txt").read_text(), "new")
        request["mode"] = "prepare"
        self.assertEqual(json.loads(execute(request)), [])
        request["files"] = [{"path": "../escape", "size": 3, "sha256": "x"}]
        with self.assertRaises(ValueError):
            execute(request)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_bf16_fused_attention_backward_and_resume(self):
        cfg = self.config()
        cfg["model"]["flash_attention"] = True
        cfg["model"]["hidden_size"] = 64
        train(cfg, device_name="cuda", max_steps=1, root=self.root)
        state = train(cfg, device_name="cuda", max_steps=1, root=self.root)
        self.assertEqual(state["step"], 2)
        self.assertEqual(state["tokens_seen"], 256)


if __name__ == "__main__":
    unittest.main()
