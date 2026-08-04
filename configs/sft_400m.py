"""base_400m 체크포인트에서 시작하는 SFT 설정."""

from configs.base_400m import model  # 아키텍처는 동일

train = dict(
    batch_size=8,
    grad_accum=8,
    max_lr=1e-5,           # 사전학습보다 훨씬 낮게 — 기존 지식 보존
    min_lr=1e-6,
    warmup_steps=50,
    epochs=2,
    weight_decay=0.0,      # SFT에서는 보통 끔
    grad_clip=1.0,
    loss_chunk=8192,       # 청크 CE — 64K vocab logits 메모리 절감
    eval_every=200,
    save_every=500,
    max_len=1024,          # instruction 데이터는 대부분 짧다
)
