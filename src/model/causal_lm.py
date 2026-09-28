"""직접 구현한 decoder-only LM. labels는 Dataset에서 이미 한 칸 이동한다."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 1536
    num_layers: int = 30
    intermediate_size: int = 4096
    num_attention_heads: int = 24
    num_key_value_heads: int = 6
    sequence_length: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    activation_checkpointing: bool = True
    loss_chunk_tokens: int = 2048
    flash_attention: bool = True

    def __post_init__(self):
        for name in ("vocab_size", "hidden_size", "num_layers", "intermediate_size",
                     "num_attention_heads", "num_key_value_heads", "sequence_length",
                     "loss_chunk_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide into query heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be a multiple of KV heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("RoPE head dimension must be even")

    def parameter_count(self):
        d = self.hidden_size
        kv = d // self.num_attention_heads * self.num_key_value_heads
        return self.vocab_size*d + self.num_layers*(2*d*d + 2*d*kv
                   + 3*d*self.intermediate_size + 2*d) + d


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(x.dtype)


def rotary(x, cos, sin):
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even*cos - odd*sin, even*sin + odd*cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.kv_heads = c.num_attention_heads, c.num_key_value_heads
        self.head_dim = c.hidden_size // self.heads
        self.kv_width = self.kv_heads * self.head_dim
        self.qkv = nn.Linear(c.hidden_size, c.hidden_size + 2*self.kv_width, bias=False)
        self.output = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.flash = c.flash_attention

    def forward(self, x, cos, sin):
        b, s, d = x.shape
        q, k, v = self.qkv(x).split((d, self.kv_width, self.kv_width), dim=-1)
        q = q.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        # CUDA에서는 느린 math fallback을 조용히 사용하지 않는다.
        backend = SDPBackend.MATH
        if x.is_cuda and self.flash:
            backend = (SDPBackend.FLASH_ATTENTION if torch.backends.cuda.is_flash_attention_available()
                       else SDPBackend.CUDNN_ATTENTION)
        with sdpa_kernel(backend):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                              dropout_p=0.0, enable_gqa=True)
        return self.output(y.transpose(1, 2).contiguous().view(b, s, d))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.attn = Attention(c)
        self.ffn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.gate_up = nn.Linear(c.hidden_size, 2*c.intermediate_size, bias=False)
        self.down = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        gate, up = self.gate_up(self.ffn_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        head_dim = config.hidden_size // config.num_attention_heads
        inv = 1.0 / config.rope_theta ** (torch.arange(0, head_dim, 2).float()/head_dim)
        angles = torch.outer(torch.arange(config.sequence_length).float(), inv)
        self.register_buffer("rope_cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("rope_sin", angles.sin()[None, None], persistent=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.output.weight, std=0.02/math.sqrt(2*config.num_layers))
            nn.init.normal_(block.down.weight, std=0.02/math.sqrt(2*config.num_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def _chunk_loss(self, hidden, targets):
        return F.cross_entropy(F.linear(hidden, self.embedding.weight).float(),
                               targets, reduction="sum")

    def forward(self, input_ids, labels=None):
        s = input_ids.shape[1]
        if s > self.config.sequence_length:
            raise ValueError("input exceeds configured context length")
        x = self.embedding(input_ids)
        cos, sin = self.rope_cos[:, :, :s], self.rope_sin[:, :, :s]
        # RoPE도 BF16 attention 입력 dtype을 유지한다.
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        cos, sin = cos.to(dtype), sin.to(dtype)
        for block in self.blocks:
            if self.training and self.config.activation_checkpointing:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.norm(x)
        if labels is None:
            return F.linear(x, self.embedding.weight)
        if labels.shape != input_ids.shape:
            raise ValueError("labels must already be shifted and match input_ids")
        hidden, targets = x.reshape(-1, x.shape[-1]), labels.reshape(-1)
        loss = x.new_zeros((), dtype=torch.float32)
        for start in range(0, targets.numel(), self.config.loss_chunk_tokens):
            end = start + self.config.loss_chunk_tokens
            if self.training and torch.is_grad_enabled():
                # logits 전체를 저장하지 않고 backward에서 chunk별로 재계산한다.
                part = checkpoint(self._chunk_loss, hidden[start:end], targets[start:end],
                                  use_reentrant=False)
            else:
                part = self._chunk_loss(hidden[start:end], targets[start:end])
            loss = loss + part
        return loss / targets.numel()
