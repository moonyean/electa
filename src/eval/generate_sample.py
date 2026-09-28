"""체크포인트에서 텍스트를 생성해 pretraining 결과를 눈으로 확인한다.

사용:
    python src/eval/generate_sample.py --checkpoint checkpoints/pretrain_800m \
        --prompt "대한민국의 수도는"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch
import torch.nn.functional as F
import sentencepiece as spm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.causal_lm import CausalLM, ModelConfig

DEFAULT_TOKENIZER = "data/processed/tokenizer/unigram_32k.model"
DEFAULT_MODEL_CONFIG = "configs/pretrain_800m.json"


def load_model(checkpoint_dir: Path, model_config_path: Path, device: torch.device) -> CausalLM:
    cfg = json.loads(model_config_path.read_text(encoding="utf-8"))["model"]
    config = ModelConfig(**cfg)
    model = CausalLM(config)

    state_path = checkpoint_dir / "state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(model_state)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(model, sp, prompt, device, max_new_tokens=100, temperature=0.8, top_p=0.9):
    ids = sp.encode(prompt, out_type=int)
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    max_len = model.config.sequence_length

    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for _ in range(max_new_tokens):
        context = ids[:, -max_len:]
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            logits = model(context)[:, -1, :]
        logits = logits.float() / max(temperature, 1e-5)

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cumulative > top_p).float().argmax(dim=-1, keepdim=True)
        mask = torch.arange(sorted_probs.shape[-1], device=device).unsqueeze(0) > cutoff
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_sorted = torch.multinomial(sorted_probs, 1)
        next_token = sorted_idx.gather(-1, next_sorted)

        ids = torch.cat([ids, next_token], dim=1)
        if next_token.item() == sp.eos_id():
            break

    return sp.decode(ids[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pretrain_800m")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[2]

    sp = spm.SentencePieceProcessor(model_file=str(root / args.tokenizer))
    model = load_model(root / args.checkpoint, root / args.model_config, device)

    prompts = args.prompt or [
        "대한민국의 수도는",
        "오늘 날씨가 좋아서",
        "인공지능이란",
    ]

    for prompt in prompts:
        text = generate(model, sp, prompt, device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_p=args.top_p)
        print(f"[prompt] {prompt}")
        print(f"[output] {text}")
        print("-" * 40)


if __name__ == "__main__":
    main()
