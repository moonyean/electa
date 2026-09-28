"""작은 로컬 입력만 사용하는 재실행/손상/샘플링 회귀 테스트."""

import contextlib
import gzip
import io
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sentencepiece as spm

from src.tokenizer import tokenize_and_binarize as binary
from src.tokenizer import unigram
from src.training.pretrain_dataset import create_pretrain_dataloader
from src.utils.artifacts import read_state
from src.data import fineWeb2_preprocess as pipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter


WORKSPACE = Path(__file__).resolve().parents[1]


class FailRankOnce(PipelineStep):
    def __init__(self, marker):
        super().__init__()
        self.marker = marker

    def run(self, data, rank=0, world_size=1):
        if rank == 1 and self.marker.exists():
            raise RuntimeError("injected worker failure")
        yield from data


class ReviewFixes(unittest.TestCase):
    def setUp(self):
        # Python 3.14 mkdtemp(mode=0700)의 Windows ACL 문제를 피한다.
        self.root = WORKSPACE / ("review_test_" + uuid.uuid4().hex)
        self.root.mkdir()
        self.capture = contextlib.redirect_stdout(io.StringIO())
        self.capture.__enter__()

    def tearDown(self):
        self.capture.__exit__(None, None, None)
        target = self.root.resolve()
        assert target.parent == WORKSPACE and target.name.startswith("review_test_")
        shutil.rmtree(target)

    def records(self, folder, texts, name="sample.jsonl"):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "wt", encoding="utf-8") as stream:
            for i, text in enumerate(texts):
                stream.write(json.dumps({"id": str(i), "text": text}, ensure_ascii=False) + "\n")
        return path

    def tiny_model(self, name="model"):
        buffer = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(["안녕하세요 한국어 학습 문장입니다", "document number one two three"]),
            model_writer=buffer, vocab_size=320, byte_fallback=True,
            hard_vocab_limit=False, num_threads=1, minloglevel=2,
        )
        path = self.root / (name + ".model")
        path.write_bytes(buffer.getvalue())
        return path

    def tokenize(self, extra=()):
        argv = ["tokenize", "--input-root", str(self.root / "input"),
                "--output-root", str(self.root / "output"),
                "--checkpoint", str(self.root / "checkpoint.json"),
                "--tokenizer", str(self.root / "model.model"), *extra]
        with patch.object(sys, "argv", argv):
            binary.main()

    def token_inputs(self):
        self.tiny_model()
        for split in ("train", "val"):
            self.records(self.root / "input" / split, ["document one", "document two", "document three"])

    def test_test_limit_cannot_be_reused_as_full_run(self):
        self.token_inputs()
        self.tokenize(["--max-documents", "1"])
        before = (self.root / "output/train/sample.bin").read_bytes()
        with self.assertRaises(RuntimeError):
            self.tokenize()
        self.assertEqual(before, (self.root / "output/train/sample.bin").read_bytes())
        self.tokenize(["--overwrite"])
        metadata = read_state(self.root / "output/metadata.json")
        self.assertEqual(metadata["shards"]["train/sample"]["documents"], 3)

    def test_changed_input_is_rejected(self):
        self.token_inputs()
        self.tokenize()
        path = self.root / "input/train/sample.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("one", "six"), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.tokenize()

    def test_changed_tokenizer_is_rejected(self):
        self.token_inputs()
        self.tokenize()
        from sentencepiece import sentencepiece_model_pb2
        path = self.root / "model.model"
        model = sentencepiece_model_pb2.ModelProto()
        model.ParseFromString(path.read_bytes())
        model.pieces[-1].score -= 1
        path.write_bytes(model.SerializeToString())
        with self.assertRaises(RuntimeError):
            self.tokenize()

    def test_corrupt_output_rebuilt_and_clean_output_reused(self):
        self.token_inputs()
        self.tokenize()
        path = self.root / "output/train/sample.bin"
        expected = path.read_bytes()
        timestamp = path.stat().st_mtime_ns
        self.tokenize()
        self.assertEqual(timestamp, path.stat().st_mtime_ns)
        path.write_bytes(b"\0" * len(expected))
        self.tokenize()
        self.assertEqual(expected, path.read_bytes())

    def test_every_shard_sampled_and_long_text_preserved(self):
        long = "한국어 문장 테스트 " * 1000
        for index in range(3):
            self.records(self.root / "input", [long + str(index)], f"{index}.jsonl")
        corpus = self.root / "corpus.txt"
        state = self.root / "corpus.json"
        docs, chars = unigram.build_corpus(self.root / "input", corpus, state, 30, 300000, 42)
        self.assertEqual(docs, 3)
        self.assertEqual(chars, 3 * (len(long) + 1))
        self.assertTrue(all(len(line) <= unigram.MAX_SENTENCE_BYTES for line in corpus.read_bytes().splitlines()))
        self.assertEqual(len(read_state(state)["shards"]), 3)
        self.assertEqual(unigram.build_corpus(self.root / "input", corpus, state, 30, 300000, 42), (docs, chars))
        with self.assertRaises(RuntimeError):
            unigram.build_corpus(self.root / "input", corpus, state, 31, 300000, 42)

    def test_corpus_resume_removes_partial_shard(self):
        for index in range(2):
            self.records(self.root / "input", [f"document {index}"] * 3, f"{index}.jsonl")
        corpus, state = self.root / "corpus.txt", self.root / "corpus.json"
        original = unigram.sample_shard
        calls = 0

        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected sampling interruption")
            return original(*args)

        with patch.object(unigram, "sample_shard", fail_second), self.assertRaises(RuntimeError):
            unigram.build_corpus(self.root / "input", corpus, state, 6, 1000, 42)
        with corpus.open("a", encoding="utf-8") as stream:
            stream.write("uncommitted garbage")
        docs, _ = unigram.build_corpus(self.root / "input", corpus, state, 6, 1000, 42)
        self.assertEqual(docs, 6)
        self.assertNotIn("garbage", corpus.read_text(encoding="utf-8"))

    def test_sampling_not_just_first_documents(self):
        source = self.records(self.root, [f"row {i}" for i in range(100)])
        sampled, count = unigram.sample_shard(source, 10, 1000, 42)
        self.assertEqual(count, 100)
        self.assertEqual(len(sampled), 10)
        self.assertTrue(any(int(text.split()[1]) > 50 for text in sampled))
        self.assertEqual(sampled, unigram.sample_shard(source, 10, 1000, 42)[0])

    def test_training_rejects_legacy_long_lines(self):
        corpus = self.root / "long.txt"
        corpus.write_text("가" * 5000, encoding="utf-8")
        with self.assertRaises(ValueError):
            unigram.train_tokenizer(corpus, self.root / "model", 320, 1.0, 1)

    def test_seed_and_validation_tail(self):
        folder = self.root / "tokens"
        folder.mkdir()
        np.arange(101, dtype="<u2").tofile(folder / "sample.bin")
        orders = []
        for _ in range(2):
            _, loader = create_pretrain_dataloader(folder, 4, 2, num_workers=0, seed=42)
            orders.append([b["input_ids"][:, 0].tolist() for b in loader])
        self.assertEqual(orders[0], orders[1])
        _, loader = create_pretrain_dataloader(folder, 64, 2, num_workers=0, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 64))
        self.assertEqual(batch["labels"][0, 0].item(), 1)

    def test_strict_reader_and_validation(self):
        self.records(self.root / "input", ["한국어 원문입니다"], "sample.jsonl.gz")
        reader = pipeline.jsonl_reader(self.root / "input")
        self.assertEqual(next(reader.read_file("sample.jsonl.gz")).text, "한국어 원문입니다")
        with gzip.open(self.root / "input/bad.jsonl.gz", "wb") as stream:
            stream.write(b"\xff")
        with self.assertRaises(UnicodeDecodeError):
            list(reader.read_file("bad.jsonl.gz"))
        with self.assertRaises(UnicodeDecodeError):
            pipeline.validate_jsonl(self.root / "input")

    def test_worker_limit_and_log_path(self):
        with patch.object(pipeline, "LocalPipelineExecutor") as executor:
            pipeline.run_pipeline([], 22, 4, self.root / "logs")
            self.assertEqual(executor.call_args.kwargs["workers"], 4)
            self.assertEqual(executor.call_args.kwargs["logging_dir"], str(self.root / "logs"))
            pipeline.run_pipeline([], 1, 4, self.root / "logs")
            self.assertEqual(executor.call_args.kwargs["workers"], 1)

    def test_exact_and_minhash_integration(self):
        source, exact, output = self.root / "input", self.root / "exact", self.root / "output"
        first = "서울의 도서관에서는 시민을 위한 새로운 문화 행사가 열리고 있습니다. 다양한 책을 함께 읽습니다."
        second = "제주도 바다의 생태계를 연구하는 과학자들이 해양 생물의 변화를 관찰하고 결과를 발표했습니다."
        self.records(source, [first, second, first], "sample.jsonl.gz")
        pipeline.exact_deduplicate(source, exact, self.root / "work", 1, 1)
        self.assertEqual(pipeline.validate_jsonl(exact), 2)
        pipeline.minhash_deduplicate(exact, output, self.root / "work", 1, 1)
        self.assertEqual(pipeline.validate_jsonl(output), 2)
        pipeline.retire_stage(self.root / "work/pipeline_state", "exact", exact)
        pipeline.exact_deduplicate(source, exact, self.root / "work", 1, 1)
        pipeline.minhash_deduplicate(exact, output, self.root / "work", 1, 1)
        self.assertFalse(exact.exists())

    def test_small_tokenizer_training_with_chunked_korean(self):
        corpus = self.root / "corpus.txt"
        text = "한국어로 작성한 긴 문서입니다. 다양한 문장을 함께 학습합니다. " * 300
        corpus.write_text("\n".join(unigram.sentence_chunks(text)) + "\n", encoding="utf-8")
        unigram.train_tokenizer(corpus, self.root / "trained", 320, 1.0, 1)
        tokenizer = spm.SentencePieceProcessor(model_file=str(self.root / "trained.model"))
        self.assertEqual(tokenizer.eos_id(), 2)
        self.assertGreater(len(tokenizer.encode("한국어 문장")), 0)

    def test_all_dedup_phases_receive_worker_limit(self):
        @contextlib.contextmanager
        def fake_stage(*args):
            yield self.root / "logs"

        with patch.object(pipeline, "stage", fake_stage), patch.object(pipeline, "run_pipeline") as run:
            pipeline.exact_deduplicate(self.root / "input", self.root / "exact", self.root, 22, 4)
            pipeline.minhash_deduplicate(self.root / "exact", self.root / "output", self.root, 22, 4)
            self.assertEqual(run.call_count, 7)
            self.assertTrue(all(call.args[2] == 4 for call in run.call_args_list))
            self.assertEqual(len({str(call.args[3]) for call in run.call_args_list}), 5)

    def test_stage_rejects_old_artifacts_and_changed_tasks(self):
        source, output, work = self.root / "input", self.root / "output", self.root / "work"
        self.records(source, ["유효한 한국어 문서입니다"], "sample.jsonl.gz")
        stale = work / "exact_signatures"
        stale.mkdir(parents=True)
        (stale / "unknown.bin").write_bytes(b"old")
        with self.assertRaises(FileExistsError):
            with pipeline.stage("exact", source, output, work / "pipeline_state", 2):
                self.fail("must not enter stage")
        with pipeline.stage("test", source, output, work / "pipeline_state", 2):
            self.records(output, ["완료된 한국어 문서입니다"], "sample.jsonl.gz")
        with self.assertRaises(RuntimeError):
            with pipeline.stage("test", source, output, work / "pipeline_state", 3):
                self.fail("must not reuse different rank assignments")

    def test_empty_stage_does_not_allow_cleanup(self):
        source, output, state = self.root / "input", self.root / "output", self.root / "state"
        path = self.records(source, ["보존할 문서입니다"], "sample.jsonl.gz")
        with self.assertRaises(RuntimeError):
            with pipeline.stage("test", source, output, state, 1):
                output.mkdir()
        with self.assertRaises(RuntimeError):
            pipeline.retire_stage(state, "test", output)
        self.assertTrue(path.exists())

    def test_cleanup_interruption_can_resume(self):
        source, output, state = self.root / "input", self.root / "output", self.root / "state"
        self.records(source, ["보존할 입력 문서입니다"], "sample.jsonl.gz")
        with pipeline.stage("test", source, output, state, 1):
            self.records(output, ["완료된 문서입니다"], "sample.jsonl.gz")
        with patch.object(pipeline.shutil, "rmtree", side_effect=OSError("interrupted cleanup")):
            with self.assertRaises(OSError):
                pipeline.retire_stage(state, "test", output)
        self.assertFalse(output.exists())
        with pipeline.stage("test", source, output, state, 1) as logs:
            self.assertIsNone(logs)
        pipeline.retire_stage(state, "test", output)
        self.assertFalse(Path(read_state(state / "test/stage.json")["retired_path"]).exists())

    def test_parquet_split_reuses_completed_files(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import run_all_remaining_shards as runner

        raw = self.root / "raw"
        raw.mkdir()
        pq.write_table(pa.table({"text": [f"row {i}" for i in range(4001)]}), raw / "sample.parquet")
        with patch.object(runner, "RAW_DIR", raw), patch.object(runner, "N_PARTS", 2):
            folder = self.root / "split/input"
            runner.split_shard("sample", folder)
            before = {p.name: p.stat().st_mtime_ns for p in folder.glob("*.parquet")}
            runner.split_shard("sample", folder)
            self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in folder.glob("*.parquet")})
            self.assertEqual(sum(pq.ParquetFile(p).metadata.num_rows for p in folder.glob("*.parquet")), 4001)

    def test_failed_stage_retains_completed_rank_and_resume(self):
        source, output, state = self.root / "input", self.root / "output", self.root / "state"
        for index in range(2):
            self.records(source, [f"한국어 문서 {index}"], f"{index}.jsonl.gz")
        marker = self.root / "fail"
        marker.touch()

        def run():
            with pipeline.stage("test", source, output, state, 2) as logs:
                if logs is not None:
                    pipeline.run_pipeline([pipeline.jsonl_reader(source), FailRankOnce(marker),
                                           JsonlWriter(output_folder=str(output))], 2, 1, logs / "filter")

        with self.assertRaises(RuntimeError):
            run()
        finished = output / "00000.jsonl.gz"
        timestamp = finished.stat().st_mtime_ns
        marker.unlink()
        run()
        self.assertEqual(timestamp, finished.stat().st_mtime_ns)
        self.assertEqual(pipeline.validate_jsonl(output), 2)
        run()
        pipeline.retire_stage(state, "test", output)
        self.assertFalse(output.exists())
        run()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
