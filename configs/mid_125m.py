"""중형 검증 모델 — RTX 6000 Ada에서 반나절, 레시피 전체 검증용."""

from model import ModelConfig

model = ModelConfig(
    vocab_size=65536,
    dim=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=4,
    max_seq_len=1024,
)

train = dict(
    batch_size=32,
    grad_accum=8,            # 유효 배치 = 32*8*1024 ≈ 0.26M tokens
    max_lr=6e-4,
    min_lr=6e-5,
    warmup_steps=1000,
    max_steps=10000,         # ~2.6B tokens
    weight_decay=0.1,
    grad_clip=1.0,
    loss_chunk=8192,       # 청크 CE — 64K vocab logits 메모리 절감
    eval_every=250,
    save_every=1000,
)
