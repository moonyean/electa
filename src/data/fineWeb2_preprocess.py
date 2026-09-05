"""FineWeb2 한국어 데이터 다운로드, 전처리, 중복 제거 파이프라인."""

from __future__ import annotations

import argparse
import builtins
import importlib.metadata  # noqa: F401  Python 3.14 + datatrove 호환용 (importlib.metadata 지연 바인딩 버그 회피)
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

if os.name == "nt":
    # Windows 콘솔 기본 코드페이지(cp949)로는 datatrove 로그의 이모지를 출력할 수
    # 없어 UnicodeEncodeError가 난다. 처리 자체는 영향 없지만 로그가 깨지므로 교정한다.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

from datasets import load_dataset
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup import (
    ExactDedupConfig,
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
    MinhashDedupSignature,
)
from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.pipeline.filters import GopherQualityFilter, GopherRepetitionFilter
from datatrove.pipeline.formatters import FTFYFormatter, PIIFormatter, SymbolLinesFormatter
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import KiwiTokenizer, load_word_tokenizer

# 한국어 형태소 중 실제 문장에서 압도적으로 자주 나오는 조사/서술어 어간.
# GopherQualityFilter의 min_stop_words 판정용 (영어 stop word 목록의 한국어 대응).
KOREAN_STOP_WORDS = ["은", "는", "이", "가", "을", "를", "의", "에", "하", "있", "되"]


def _prepare_korean_tokenizer() -> None:
    """Kiwi 기반 한국어 단어 분리기를 사용 가능하게 준비한다.

    datatrove가 내부적으로 여는 tokenizer_assignment.csv는 UTF-8인데, Windows
    기본 코드페이지(cp949)로 열려 디코딩 에러가 난다. 또한 datatrove가 기본으로
    쓰는 Kiwi model_type="sbg"는 kiwipiepy 0.22+에서 기본 배포 모델에서 빠졌다.
    두 문제를 한 번만 우회해서 lru_cache에 정상 결과를 채워둔다.
    """
    KiwiTokenizer.__init__.__defaults__ = ("cong",)

    original_open = builtins.open

    def _utf8_open(file, mode="r", *args, **kwargs):
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return original_open(file, mode, *args, **kwargs)

    builtins.open = _utf8_open
    try:
        load_word_tokenizer(Languages.korean)
    finally:
        builtins.open = original_open


_prepare_korean_tokenizer()


class KoreanRatioFilter(PipelineStep):
    """fasttext 없이 한글 문자 비율을 기준으로 문서를 필터링한다."""

    type = "FILTER"
    name = "KoreanRatioFilter"

    def __init__(self, language_threshold: float = 0.65) -> None:
        super().__init__()
        if not 0.0 <= language_threshold <= 1.0:
            raise ValueError("language_threshold must be between 0 and 1")
        self.language_threshold = language_threshold

    def run(self, data, rank: int = 0, world_size: int = 1):
        for document in data:
            letters = [character for character in document.text if character.isalpha()]
            korean_letters = sum(
                "가" <= character <= "힣" for character in letters
            )
            ratio = korean_letters / len(letters) if letters else 0.0
            if ratio >= self.language_threshold:
                yield document


def has_input_files(input_dir: Path) -> bool:
    """입력 폴더와 하위 폴더에 처리 가능한 데이터 파일이 있는지 확인한다."""
    return input_dir.is_dir() and (
        any(input_dir.rglob("*.parquet")) or any(input_dir.rglob("*.jsonl"))
    )


def download_sample(output_file: Path, size: int) -> None:
    """FineWeb2 한국어(kor_Hang)에서 지정한 개수만큼 샘플을 내려받는다."""
    if size <= 0:
        raise ValueError("sample size must be positive")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2", name="kor_Hang", streaming=True, split="train"
    )
    with output_file.open("w", encoding="utf-8") as file:
        for sample in dataset.take(size):
            file.write(
                json.dumps(
                    {"id": sample.get("id"), "text": sample["text"]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def jsonl_reader(folder: Path) -> JsonlReader:
    """폴더 아래의 JSONL 파일을 읽는 Datatrove 리더를 만든다."""
    return JsonlReader(data_folder=str(folder), glob_pattern="*.jsonl.gz")


def auto_reader(folder: Path):
    """폴더 안 파일 형식(Parquet/JSONL)에 맞는 Datatrove 리더를 고른다."""
    if any(folder.rglob("*.parquet")):
        return ParquetReader(data_folder=str(folder), glob_pattern="*.parquet")
    return jsonl_reader(folder)


def run_pipeline(pipeline: Sequence[object], tasks: int, workers: int = -1) -> None:
    """Datatrove 파이프라인을 운영체제에 맞는 방식으로 실행한다.

    tasks(작업 조각 수)와 workers(동시 실행 수)를 분리해서, 작업을 잘게
    쪼개 조각 하나가 끝나는 데 걸리는 시간을 줄이면서도(중간에 죽어도 잃는
    양이 적어짐) 동시 실행 개수는 메모리 안전선으로 제한할 수 있다.
    """
    if tasks <= 0:
        raise ValueError("tasks must be positive")
    # Windows에서는 forkserver를 지원하지 않으므로 spawn을 사용한다.
    start_method = "spawn" if os.name == "nt" else "forkserver"
    LocalPipelineExecutor(
        pipeline=list(pipeline),
        tasks=tasks,
        workers=workers,
        start_method=start_method,
    ).run()


def preprocess(input_dir: Path, output_dir: Path, tasks: int, workers: int = -1) -> None:
    """기호 라인, 개인정보, 비한국어 문서, 저품질 문서를 차례로 제거한다."""
    steps = [
        auto_reader(input_dir),
        FTFYFormatter(),  # 인코딩이 깨진 문자(모지바케)를 복원한다.
        SymbolLinesFormatter(),  # 의미 없는 기호 라인을 제거한다.
        PIIFormatter(),  # 이메일, IP 등 개인정보를 제거한다.
        KoreanRatioFilter(language_threshold=0.65),  # 한글 비율이 낮은 문서를 제거한다.
        GopherQualityFilter(
            language=Languages.korean,
            # Kiwi 형태소 분석 기준 한국어 평균 단어(형태소) 길이는 1~2자로,
            # 영어 기본값(3~10자)을 그대로 쓰면 정상 문서까지 걸러진다.
            min_avg_word_length=1,
            max_avg_word_length=10,
            stop_words=KOREAN_STOP_WORDS,
        ),  # 반복 문장 등 저품질 문서를 제거한다.
        GopherRepetitionFilter(language=Languages.korean),  # 동일 문단/줄/n-gram 반복 문서를 제거한다.
        JsonlWriter(output_folder=str(output_dir)),
    ]
    run_pipeline(steps, tasks, workers)


def exact_deduplicate(input_dir: Path, output_dir: Path, work_dir: Path, tasks: int) -> None:
    """텍스트 전체가 완전히 같은 문서를 제거한다."""
    def text_content(document) -> bytes:
        # HashConfig 기본값(xxhash)은 입력 타입 상관없이 bytes만 받는다.
        return document.text.encode("utf-8")

    config = ExactDedupConfig(
        content_getter=text_content,
        hash_config=HashConfig(precision=64),
    )
    signatures = work_dir / "exact_signatures"
    duplicates = work_dir / "exact_duplicates"
    run_pipeline(
        [jsonl_reader(input_dir), ExactDedupSignature(str(signatures), config)],
        tasks,
    )
    run_pipeline(
        [ExactFindDedups(str(signatures), str(duplicates), config)],
        tasks,
    )
    run_pipeline(
        [
            jsonl_reader(input_dir),
            ExactDedupFilter(data_folder=str(duplicates), config=config),
            JsonlWriter(output_folder=str(output_dir)),
        ],
        tasks,
    )


def minhash_deduplicate(input_dir: Path, output_dir: Path, work_dir: Path, tasks: int) -> None:
    """MinHash와 LSH로 유사 문서를 찾아 중복 문서를 제거한다."""
    config = MinhashConfig(
        # datatrove 0.10.0의 MinHash 시그니처 계산은 xxhash를 쓰면 문자열을
        # bytes로 인코딩하지 않고 그대로 넘겨서 TypeError가 난다. sha1은
        # 문자열 입력을 정상적으로 처리하므로 이를 우회한다.
        hash_config=HashConfig(precision=64, hash_fc="sha1"),
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
    )
    signatures = work_dir / "minhash_signatures"
    buckets = work_dir / "minhash_buckets"
    remove_ids = work_dir / "minhash_remove_ids"
    run_pipeline(
        [
            jsonl_reader(input_dir),
            MinhashDedupSignature(str(signatures), config, language=Languages.korean),
        ],
        tasks,
    )
    run_pipeline(
        [MinhashDedupBuckets(str(signatures), str(buckets), config=config)],
        config.num_buckets,
    )
    run_pipeline(
        [MinhashDedupCluster(str(buckets), str(remove_ids), config)],
        1,
    )
    run_pipeline(
        [
            jsonl_reader(input_dir),
            MinhashDedupFilter(input_folder=str(remove_ids)),
            JsonlWriter(output_folder=str(output_dir)),
        ],
        tasks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/pretrain/HuggingFaceFW/fineweb-2/data/kor_Hang/train"),
        help="원본 데이터 폴더 (Parquet 또는 JSONL)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/pretrain"), help="최종 출력 폴더"
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("data/interim/pretrain"), help="중간 결과 폴더"
    )
    parser.add_argument("--tasks", type=int, default=4, help="작업을 나눌 조각 수")
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="동시에 실행할 워커 수 (기본 -1은 tasks와 동일). tasks보다 작게 주면 "
        "조각은 잘게 쪼개면서 메모리 사용량은 제한할 수 있다",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="다운로드할 샘플 문서 수")
    parser.add_argument("--skip-fuzzy-dedup", action="store_true", help="유사 중복 제거 생략")
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="단계별 중간 산출물(전처리 결과, exact dedup 결과, minhash 시그니처)을 지우지 않고 보존",
    )
    args = parser.parse_args()

    if args.sample_size is not None:
        download_sample(args.input / "fineweb2_kor_sample.jsonl", args.sample_size)

    if not has_input_files(args.input):
        print(f"입력 데이터 파일을 찾을 수 없습니다: {args.input.resolve()}")
        print("Parquet/JSONL 파일을 --input 폴더에 넣거나, 테스트용 샘플을 받으려면 --sample-size 100을 사용하세요.")
        return

    # 각 단계 산출물은 다음 단계가 원본만큼(때로는 그 이상) 디스크를 다시 잡아먹는다.
    # 안 지우고 쌓아두면 디스크가 꽉 차서 파이프라인이 죽는다(실제로 겪은 문제) —
    # 다음 단계가 성공적으로 끝나면 이전 단계 산출물은 바로 지운다.
    def cleanup(path: Path) -> None:
        if not args.keep_intermediate:
            shutil.rmtree(path, ignore_errors=True)

    preprocessed = args.work_dir / "preprocessed"
    exact = args.work_dir / "exact_deduplicated"
    preprocess(args.input, preprocessed, args.tasks, args.workers)
    exact_deduplicate(preprocessed, exact, args.work_dir, args.tasks)
    if not any(exact.glob("*.jsonl.gz")):
        raise RuntimeError(
            f"exact dedup 결과가 비어 있습니다: {exact.resolve()} — preprocessed 폴더를 "
            "보존하고 중단합니다. 원인을 확인하세요 (예: 인코딩 설정 누락)."
        )
    cleanup(preprocessed)
    cleanup(args.work_dir / "exact_signatures")
    cleanup(args.work_dir / "exact_duplicates")
    if args.skip_fuzzy_dedup:
        run_pipeline(
            [jsonl_reader(exact), JsonlWriter(output_folder=str(args.output))],
            args.tasks,
        )
    else:
        minhash_deduplicate(exact, args.output, args.work_dir, args.tasks)
        cleanup(args.work_dir / "minhash_signatures")
        cleanup(args.work_dir / "minhash_buckets")
        cleanup(args.work_dir / "minhash_remove_ids")
    cleanup(exact)


def _ensure_utf8_process() -> None:
    """프로세스 기본 인코딩을 UTF-8로 강제한다.

    Windows에서는 Python 기본 인코딩이 시스템 코드페이지(cp949)라, datatrove가
    내부적으로 여는 gzip/jsonl 파일이 인코딩 없이 열리면 전부 "손상된 파일"로
    오인돼 통째로 스킵된다. PYTHONUTF8을 프로세스 시작 시점에 반영하려면
    재실행이 필요하다 (실행 중에 os.environ만 바꿔서는 이미 뜬 인터프리터의
    기본 인코딩이 바뀌지 않는다). multiprocessing이 만드는 워커 프로세스는
    이 환경변수를 그대로 물려받으므로 한 번만 해주면 된다.

    Windows에는 진짜 execve가 없어서 os.execv는 새 프로세스를 spawn하고 현재
    프로세스는 바로 종료하는 방식으로 흉내만 낸다. 그러면 이 스크립트를 실행한
    셸이 "언제 끝났는지"를 더 이상 정확히 알 수 없게 되어(원래 프로세스는 이미
    끝났지만 실제 작업은 spawn된 새 프로세스에서 계속 진행) exit code나 완료
    시점이 뒤섞인다. subprocess.call로 자식 프로세스가 끝날 때까지 직접 기다린
    뒤 같은 exit code로 종료하면 셸 입장에서는 평범한 동기 실행과 동일하다.
    """
    if os.name == "nt" and os.environ.get("PYTHONUTF8") != "1":
        os.environ["PYTHONUTF8"] = "1"
        exit_code = subprocess.call([sys.executable] + sys.argv, env=os.environ)
        sys.exit(exit_code)


if __name__ == "__main__":
    _ensure_utf8_process()
    main()
