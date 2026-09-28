"""uint16 토큰 샤드를 Causal LM 학습 배치로 만드는 Dataset.

`tokenize_and_binarize.py`가 생성한 다음 구조를 입력으로 사용한다.

    data/processed/tokenized/
      train/*.bin
      val/*.bin

각 `.bin` 파일은 문서 사이에 EOS가 포함된 연속 토큰 스트림이다. 이 Dataset은
파일을 메모리에 전부 올리지 않고 numpy.memmap으로 읽으며, `sequence_length+1`
토큰을 가져와 다음과 같이 한 칸 이동한 입력과 정답을 만든다.

    input_ids = tokens[:-1]
    labels    = tokens[1:]

샤드 경계를 넘어 시퀀스를 만들지 않기 때문에 서로 다른 파일의 끝과 시작이
인위적으로 이어지지 않는다. 각 샤드 끝에서 남는 토큰은 최대 sequence_length개다.
"""

from __future__ import annotations

import argparse
import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class PackedPretrainDataset(Dataset):
    """여러 uint16 샤드를 고정 길이 Causal LM 시퀀스로 제공한다."""

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 2048,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        if sequence_length <= 0:
            raise ValueError("sequence_length는 양수여야 합니다.")

        self.shards = sorted(self.data_dir.glob("*.bin"))
        if not self.shards:
            raise FileNotFoundError(f".bin 샤드가 없습니다: {self.data_dir.resolve()}")

        self.tokens_per_shard: list[int] = []
        self.sequences_per_shard: list[int] = []
        self.cumulative_sequences: list[int] = []
        cumulative = 0

        for shard in self.shards:
            file_size = shard.stat().st_size
            if file_size % np.dtype("<u2").itemsize != 0:
                raise ValueError(f"uint16 크기와 맞지 않는 파일입니다: {shard}")
            token_count = file_size // np.dtype("<u2").itemsize
            # 정답을 한 칸 이동해야 하므로 시퀀스마다 seq_len+1 토큰이 필요하다.
            sequence_count = max(0, (token_count - 1) // sequence_length)
            self.tokens_per_shard.append(token_count)
            self.sequences_per_shard.append(sequence_count)
            cumulative += sequence_count
            self.cumulative_sequences.append(cumulative)

        if cumulative == 0:
            raise ValueError(
                f"sequence_length={sequence_length}로 만들 수 있는 시퀀스가 없습니다."
            )

        # DataLoader 워커마다 필요할 때 memmap을 열도록 비워 둔다.
        self._memmaps: dict[int, np.memmap] = {}

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens_per_shard)

    def __len__(self) -> int:
        return self.cumulative_sequences[-1]

    def _get_memmap(self, shard_index: int) -> np.memmap:
        """현재 프로세스에서 사용할 샤드 메모리맵을 지연 생성한다."""
        if shard_index not in self._memmaps:
            self._memmaps[shard_index] = np.memmap(
                self.shards[shard_index],
                mode="r",
                dtype="<u2",
            )
        return self._memmaps[shard_index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_index = bisect.bisect_right(self.cumulative_sequences, index)
        previous_total = self.cumulative_sequences[shard_index - 1] if shard_index else 0
        local_sequence_index = index - previous_total
        start = local_sequence_index * self.sequence_length
        stop = start + self.sequence_length + 1

        # memmap 슬라이스는 읽기 전용일 수 있으므로 복사한 뒤 torch Tensor로 바꾼다.
        tokens = np.asarray(self._get_memmap(shard_index)[start:stop], dtype=np.int64).copy()
        if len(tokens) != self.sequence_length + 1:
            raise RuntimeError(
                f"잘못된 시퀀스 길이: {self.shards[shard_index]} "
                f"offset={start}, length={len(tokens)}"
            )

        token_tensor = torch.from_numpy(tokens)
        return {
            "input_ids": token_tensor[:-1],
            "labels": token_tensor[1:],
        }

    def __getstate__(self):
        """Windows spawn 워커에 memmap 객체가 직접 전달되지 않게 한다."""
        state = self.__dict__.copy()
        state["_memmaps"] = {}
        return state


def create_pretrain_dataloader(
    data_dir: str | Path,
    sequence_length: int,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False,
    seed: int = 42,
    drop_last: bool = False,
) -> tuple[PackedPretrainDataset, DataLoader]:
    """단일 GPU/DDP용 로더. 검증 누락 방지를 위해 drop_last 기본값은 False다.

    학습에서 고정 배치가 필요하면 drop_last=True를 명시한다. 같은 seed는
    새 로더의 순서를 재현한다. 학습 중간 재개에는 별도의 sampler/RNG 상태가 필요하다.
    """
    if batch_size <= 0:
        raise ValueError("batch_size는 양수여야 합니다.")
    if num_workers < 0:
        raise ValueError("num_workers는 0 이상이어야 합니다.")

    dataset = PackedPretrainDataset(data_dir, sequence_length)
    sampler = None
    if distributed:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("distributed=True 전에 torch.distributed를 초기화해야 합니다.")
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
        drop_last=drop_last,
    )
    return dataset, loader


def main() -> None:
    """학습 전에 한 배치를 읽어 데이터 형태를 확인한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/tokenized"),
        help="train/val 폴더가 있는 토큰 데이터 루트",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    dataset, loader = create_pretrain_dataloader(
        data_dir=args.data_root / args.split,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )
    batch = next(iter(loader))
    print(f"분할: {args.split}")
    print(f"샤드 수: {len(dataset.shards):,}")
    print(f"전체 토큰 수: {dataset.total_tokens:,}")
    print(f"학습 시퀀스 수: {len(dataset):,}")
    print(f"input_ids 형태: {tuple(batch['input_ids'].shape)}")
    print(f"labels 형태: {tuple(batch['labels'].shape)}")
    print(f"입력 첫 8개: {batch['input_ids'][0, :8].tolist()}")
    print(f"정답 첫 8개: {batch['labels'][0, :8].tolist()}")


if __name__ == "__main__":
    main()
