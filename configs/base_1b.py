"""본 학습 모델 (~1.2B) — RTX 6000 Ada 48GB 단일 GPU, 25B 토큰, 약 1개월.

파라미터 내역:
  임베딩      65,536 × 2,048              = 134M  (입출력 공유)
  층당        attn 10.5M + FFN 33.8M      = 44.3M
  24층                                     = 1,063M
  합계                                     ≈ 1.2B

토큰/파라미터 = 25B / 1.2B ≈ 21 — Chinchilla 적정선.
"""

from model import ModelConfig

model = ModelConfig(
    vocab_size=65536,
    dim=2048,
    n_layers=24,
    n_heads=16,        # head_dim = 128
    n_kv_heads=4,      # GQA 4:1 — KV 캐시 1/4
    max_seq_len=2048,
)

train = dict(
    # 마이크로 배치는 벤치마크로 확정 (bench.py --config base_1b)
    batch_size=4,
    grad_accum=128,          # 유효 배치 = 4*128*2048 ≈ 1.05M tokens
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=2000,
    max_steps=24000,         # ≈ 25.2B tokens
    weight_decay=0.1,
    grad_clip=1.0,
    loss_chunk=8192,
    eval_every=500,
    save_every=500,          # 한 달짜리 학습이라 자주 저장
    milestone_every=5000,    # 별도 보관용 스냅샷 (덮어쓰지 않음)
)
