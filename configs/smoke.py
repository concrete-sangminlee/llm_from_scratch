"""파이프라인 스모크 테스트용 (커밋되지만 실험용) — 작은 vocab 토크나이저와 함께 사용."""

from model import ModelConfig

model = ModelConfig(
    vocab_size=2048,
    dim=256,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=256,
)

train = dict(
    batch_size=8,
    grad_accum=2,
    max_lr=1e-3,
    min_lr=1e-4,
    warmup_steps=10,
    max_steps=60,
    weight_decay=0.1,
    grad_clip=1.0,
    eval_every=25,
    save_every=50,
)
