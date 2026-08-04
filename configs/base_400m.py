"""본 학습 모델 (~400M) — RTX 6000 Ada 48GB, 12–20B 토큰, 1~2주."""

from model import ModelConfig

model = ModelConfig(
    vocab_size=65536,
    dim=1024,
    n_layers=24,
    n_heads=16,
    n_kv_heads=4,
    max_seq_len=2048,
)

train = dict(
    batch_size=16,
    grad_accum=16,           # 유효 배치 = 16*16*2048 ≈ 0.52M tokens
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=2000,
    max_steps=24000,         # ~12.6B tokens (WSD로 연장 가능)
    weight_decay=0.1,
    grad_clip=1.0,
    eval_every=500,
    save_every=1000,
)
