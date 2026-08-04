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
    # 벤치마크(RTX 6000 Ada + torch.compile) 결과 마이크로배치 4가 최적:
    # 34.7K tok/s / VRAM 14GB (배치 8·12는 오히려 느리고 메모리만 2~3배)
    batch_size=4,
    grad_accum=64,           # 유효 배치 = 4*64*2048 ≈ 0.52M tokens
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=2000,
    max_steps=24000,         # ~12.6B tokens (WSD로 연장 가능)
    weight_decay=0.1,
    grad_clip=1.0,
    loss_chunk=8192,       # 청크 CE — 64K vocab logits 메모리 절감
    eval_every=500,
    save_every=1000,
)
