"""SFT 파이프라인 스모크 테스트용 — configs/smoke.py 모델과 짝."""

from configs.smoke import model

train = dict(
    batch_size=4,
    grad_accum=2,
    max_lr=1e-4,
    min_lr=1e-5,
    warmup_steps=5,
    epochs=1,
    weight_decay=0.0,
    grad_clip=1.0,
    eval_every=100,
    save_every=100,
    max_len=256,
)
