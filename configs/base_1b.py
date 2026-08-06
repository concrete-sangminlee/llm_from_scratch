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
    # 벤치마크 실측: 마이크로배치 2 → 6.8K tok/s / VRAM 26.1GB.
    # 배치 4는 같은 GPU의 vLLM 서버(14GB)와 겹쳐 OOM.
    batch_size=2,
    grad_accum=256,          # 유효 배치 = 2*256*2048 ≈ 1.05M tokens
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=2000,
    max_steps=24000,         # ≈ 25.2B tokens (step당 약 154초, 총 43일)
    weight_decay=0.1,
    grad_clip=1.0,
    loss_chunk=8192,
    # step 하나가 2.6분이라 500 step은 21시간. 크래시 시 손실이 너무 커서 100으로.
    eval_every=100,          # 약 4.3시간마다
    save_every=100,          # 약 4.3시간마다 (체크포인트 14GB)
    milestone_every=4000,    # 약 7일마다 스냅샷 보관
)
