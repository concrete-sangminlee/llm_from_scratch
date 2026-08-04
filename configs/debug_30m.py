"""디버그용 초소형 모델 — Mac/CPU에서 분 단위 실험."""

from model import ModelConfig

model = ModelConfig(
    vocab_size=65536,
    dim=384,
    n_layers=8,
    n_heads=8,
    n_kv_heads=2,
    max_seq_len=512,
)

train = dict(
    batch_size=8,           # 마이크로 배치 (시퀀스 수)
    grad_accum=4,
    max_lr=1e-3,
    min_lr=1e-4,
    warmup_steps=100,
    max_steps=2000,
    weight_decay=0.1,
    grad_clip=1.0,
    eval_every=200,
    save_every=500,
)
