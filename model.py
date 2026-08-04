"""Decoder-only 트랜스포머 — Llama 3 / Qwen3 계열 최신 구성.

구성 요소와 채택 이유:
- Pre-RMSNorm: LayerNorm에서 평균 빼기(centering)를 제거한 형태. 성능 동급에 연산 절약.
  pre-norm(블록 입구에서 정규화)이 post-norm보다 깊은 모델에서 안정적.
- RoPE: 위치를 Q·K 벡터의 회전으로 인코딩. 학습 길이 밖 일반화와 상대 위치 표현이
  learned positional embedding보다 우수. 최신 모델 전부 채택.
- GQA: K·V head를 Q head보다 적게 둬서 KV cache와 연산 절감. (n_head=16, n_kv_head=4면
  Q head 4개가 KV head 1개를 공유)
- QK-Norm: attention 직전 Q·K를 head 단위로 RMSNorm. logit 폭주를 막아 학습 안정성 개선
  (Qwen3, Gemma 계열 채택).
- SwiGLU: gate * up 곱 구조의 MLP. 같은 파라미터로 GELU MLP보다 품질 우수.
- bias 없는 Linear, embedding tying(입출력 임베딩 공유 — 소형 모델에서 파라미터 절약).

attention은 교육용 명시적 구현(explicit)과 F.scaled_dot_product_attention(SDPA,
FlashAttention 커널) 두 경로를 두고 단위테스트로 일치를 검증한다.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 65536
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    # SwiGLU hidden dim. 관례상 (8/3)*dim을 64의 배수로 올림
    ffn_dim: int | None = None

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0
        if self.ffn_dim is None:
            self.ffn_dim = ((int(self.dim * 8 / 3) + 63) // 64) * 64


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # float32로 계산해 bf16 정밀도 손실 방지 (표준 관행)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    """RoPE 회전각의 cos/sin 테이블. shape (max_seq_len, dim/2), complex 대신 실수 쌍."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)  # (T, dim/2)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # (T, dim/2, 2)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, hd). 인접 차원 쌍 (x0,x1)을 각도 θ만큼 회전."""
    B, H, T, hd = x.shape
    x = x.float().view(B, H, T, hd // 2, 2)
    cos, sin = rope[:T, :, 0], rope[:T, :, 1]  # (T, hd/2)
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(3)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)

    def forward(self, x, rope, kv_cache=None, explicit: bool = False):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)  # QK-Norm은 RoPE 이전에 적용

        # cache 사용 시 새 토큰의 절대 위치는 pos부터 시작
        pos = 0 if kv_cache is None else kv_cache[0].shape[2]
        q = apply_rope(q, rope[pos:]).to(x.dtype)
        k = apply_rope(k, rope[pos:]).to(x.dtype)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # GQA: KV head를 Q head 수만큼 반복
        rep = self.n_heads // self.n_kv_heads
        k_r = k.repeat_interleave(rep, dim=1)
        v_r = v.repeat_interleave(rep, dim=1)

        if explicit:
            # 교육용 명시적 구현 — SDPA와 일치해야 함 (단위테스트로 검증)
            scores = (q @ k_r.transpose(-2, -1)) / math.sqrt(self.head_dim)
            Tk = k_r.shape[2]
            causal = torch.tril(torch.ones(Tk, Tk, device=x.device, dtype=torch.bool))
            causal = causal[Tk - T : Tk]  # cache 사용 시 새 토큰 행만
            scores = scores.masked_fill(~causal, float("-inf"))
            out = F.softmax(scores.float(), dim=-1).to(q.dtype) @ v_r
        else:
            out = F.scaled_dot_product_attention(q, k_r, v_r, is_causal=(kv_cache is None))
            if kv_cache is not None and T > 1:
                raise NotImplementedError("cache에 다중 토큰 추가는 explicit 경로 사용")

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w_up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w_down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def _chunk_loss(h_chunk, t_chunk, w):
    """한 청크의 logits를 만들어 loss 합을 반환. 체크포인팅으로 감싸 쓰인다."""
    logits = F.linear(h_chunk, w).float()
    return F.cross_entropy(logits, t_chunk, ignore_index=-100, reduction="sum")


def chunked_cross_entropy(h, targets, w, chunk: int):
    """vocab이 클 때 logits 전체를 메모리에 올리지 않는 cross-entropy.

    64K vocab · 32K 토큰이면 fp32 logits만 8.6GB라 그대로는 48GB GPU도 터진다.
    시퀀스를 청크로 쪼개 계산하고, gradient checkpointing으로 backward 때
    청크 하나씩만 다시 만들어 쓴다 — 순간 메모리가 1/N로 줄고 결과는 동일하다.
    """
    h = h.reshape(-1, h.size(-1))
    t = targets.reshape(-1)
    total = h.new_zeros((), dtype=torch.float32)
    n_valid = (t != -100).sum()
    for i in range(0, h.size(0), chunk):
        hc, tc = h[i : i + chunk], t[i : i + chunk]
        if torch.is_grad_enabled() and h.requires_grad:
            total = total + torch.utils.checkpoint.checkpoint(
                _chunk_loss, hc, tc, w, use_reentrant=False
            )
        else:
            total = total + _chunk_loss(hc, tc, w)
    return total / n_valid.clamp(min=1)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, rope, kv_cache=None, explicit=False):
        a, new_cache = self.attn(self.attn_norm(x), rope, kv_cache, explicit)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # embedding tying

        rope = precompute_rope(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope", rope, persistent=False)

        self.apply(self._init_weights)
        # residual 경로에 더해지는 projection은 깊이에 따라 스케일 다운 (GPT-2 레시피)
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("w_down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, explicit=False, loss_chunk=None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x, _ = block(x, self.rope, explicit=explicit)
        x = self.final_norm(x)
        if targets is not None:
            if loss_chunk:
                return None, chunked_cross_entropy(x, targets, self.lm_head.weight, loss_chunk)
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
            return logits, loss
        return self.lm_head(x[:, [-1], :]), None  # 추론 시 마지막 위치만

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.95, eos_id=None):
        """KV cache 기반 생성. idx: (B, T) 프롬프트."""
        self.eval()
        caches = [None] * len(self.blocks)
        x = self.tok_emb(idx)
        # 프롬프트는 explicit 경로로 한 번에 처리하며 cache 구축
        for i, block in enumerate(self.blocks):
            x, caches[i] = block(x, self.rope, kv_cache=(
                torch.empty(idx.shape[0], self.cfg.n_kv_heads, 0,
                            self.cfg.dim // self.cfg.n_heads, device=idx.device, dtype=x.dtype),
            ) * 2, explicit=True)
        x = self.final_norm(x)
        logits = self.lm_head(x[:, -1])

        out = idx
        for _ in range(max_new_tokens):
            next_id = self._sample(logits, temperature, top_p)
            out = torch.cat([out, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
            x = self.tok_emb(next_id)
            for i, block in enumerate(self.blocks):
                x, caches[i] = block(x, self.rope, kv_cache=caches[i], explicit=True)
            x = self.final_norm(x)
            logits = self.lm_head(x[:, -1])
        return out

    @staticmethod
    def _sample(logits, temperature, top_p):
        if temperature <= 0:
            return logits.argmax(-1, keepdim=True)
        probs = F.softmax(logits.float() / temperature, dim=-1)
        # nucleus (top-p) 샘플링
        sorted_p, sorted_idx = probs.sort(descending=True)
        cum = sorted_p.cumsum(-1)
        mask = cum - sorted_p > top_p
        sorted_p[mask] = 0.0
        sorted_p /= sorted_p.sum(-1, keepdim=True)
        pick = torch.multinomial(sorted_p, 1)
        return sorted_idx.gather(-1, pick)

    def num_params(self, non_embedding=True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()  # tying이라 lm_head와 동일 텐서
        return n

    def param_groups(self, weight_decay: float):
        """2D 이상(행렬)에만 weight decay, 1D(norm 가중치)는 제외 — 표준 관행."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
