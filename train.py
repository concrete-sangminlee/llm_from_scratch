"""사전학습 루프.

레시피:
- AdamW (β=(0.9, 0.95), wd=0.1 — 행렬에만), grad clip 1.0
- WSD 스케줄 (Warmup → Stable → Decay): cosine과 달리 stable 구간이 길어
  max_steps를 나중에 늘려도 이어서 학습 가능. decay는 마지막 20%에서 선형.
- bf16 autocast (+ CUDA면 TF32, torch.compile)
- gradient accumulation으로 유효 배치 확보
- 체크포인트에 model/optimizer/dataloader step까지 저장 → 완전 재개
- 주기적 val loss + 생성 샘플 출력, metrics.csv 로깅 (wandb는 --wandb로 선택)

사용 예:
  python train.py --config debug_30m
  python train.py --config base_400m --resume checkpoints/base_400m/latest.pt
"""

import argparse
import csv
import importlib
import os
import time

import torch

from data.dataloader import ShardedDataLoader
from model import GPT
from tokenizer.bpe import ByteBPETokenizer


def lr_at(step, cfg):
    """WSD: warmup → stable(max_lr) → 마지막 20% 선형 decay(min_lr까지)."""
    if step < cfg["warmup_steps"]:
        return cfg["max_lr"] * (step + 1) / cfg["warmup_steps"]
    decay_start = int(cfg["max_steps"] * 0.8)
    if step < decay_start:
        return cfg["max_lr"]
    frac = (step - decay_start) / max(1, cfg["max_steps"] - decay_start)
    return cfg["max_lr"] + frac * (cfg["min_lr"] - cfg["max_lr"])


@torch.no_grad()
def estimate_val_loss(model, loader, iters, device_type, loss_chunk=None):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = loader.next_batch()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y, loss_chunk=loss_chunk)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/ 모듈명 (예: debug_30m)")
    ap.add_argument("--shard-dir", default="data/shards")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--compile", action="store_true", default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    conf = importlib.import_module(f"configs.{args.config}")
    mcfg, tcfg = conf.model, conf.train

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu" if device == "cpu" else "mps"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32

    ckpt_dir = os.path.join("checkpoints", args.config)
    os.makedirs(ckpt_dir, exist_ok=True)

    tok = ByteBPETokenizer.load(args.tokenizer)
    train_loader = ShardedDataLoader(args.shard_dir, "train", tcfg["batch_size"], mcfg.max_seq_len,
                                     device=device)
    val_loader = ShardedDataLoader(args.shard_dir, "val", tcfg["batch_size"], mcfg.max_seq_len,
                                   seed=123, device=device)

    model = GPT(mcfg).to(device)
    print(f"모델: {model.num_params()/1e6:.1f}M 파라미터 (임베딩 제외) | device: {device}")
    opt = torch.optim.AdamW(model.param_groups(tcfg["weight_decay"]),
                            lr=tcfg["max_lr"], betas=(0.9, 0.95), fused=(device == "cuda"))

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        train_loader.load_state_dict(ckpt["train_loader"])
        start_step = ckpt["step"] + 1
        print(f"재개: step {start_step}부터")

    use_compile = args.compile if args.compile is not None else (device == "cuda")
    if use_compile:
        model = torch.compile(model)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="llm_from_scratch", name=args.config, config=tcfg)

    metrics_path = os.path.join(ckpt_dir, "metrics.csv")
    metrics_f = open(metrics_path, "a", newline="")
    metrics = csv.writer(metrics_f)
    if start_step == 0:
        metrics.writerow(["step", "lr", "train_loss", "val_loss", "tokens_per_sec"])

    tokens_per_step = tcfg["batch_size"] * tcfg["grad_accum"] * mcfg.max_seq_len
    model.train()
    t0 = time.time()
    for step in range(start_step, tcfg["max_steps"]):
        lr = lr_at(step, tcfg)
        for g in opt.param_groups:
            g["lr"] = lr

        loss_acc = 0.0
        for micro in range(tcfg["grad_accum"]):
            x, y = train_loader.next_batch()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y, loss_chunk=tcfg.get("loss_chunk"))
            (loss / tcfg["grad_accum"]).backward()
            loss_acc += loss.item() / tcfg["grad_accum"]
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_step * (10 if step > start_step else 1) / dt
            print(f"step {step:6d} | loss {loss_acc:.4f} | lr {lr:.2e} | {tps/1e3:.0f}K tok/s")

        val_loss = ""
        if step > 0 and step % tcfg["eval_every"] == 0:
            val_loss = estimate_val_loss(model, val_loader, 20, device_type, tcfg.get("loss_chunk"))
            print(f"step {step:6d} | val loss {val_loss:.4f}")
            # 생성 샘플로 정성 확인
            raw = getattr(model, "_orig_mod", model)
            prompt = torch.tensor([tok.encode("대한민국의 수도는")], device=device)
            sample = raw.generate(prompt, 48, temperature=0.8,
                                  eos_id=tok.special_tokens["<|eos|>"])
            print("샘플:", tok.decode(sample[0].tolist()))
            model.train()

        metrics.writerow([step, f"{lr:.3e}", f"{loss_acc:.4f}", val_loss, ""])
        if run:
            log = {"loss": loss_acc, "lr": lr}
            if val_loss != "":
                log["val_loss"] = val_loss
            run.log(log, step=step)

        if step > 0 and step % tcfg["save_every"] == 0 or step == tcfg["max_steps"] - 1:
            raw = getattr(model, "_orig_mod", model)
            torch.save({
                "model": raw.state_dict(),
                "optimizer": opt.state_dict(),
                "train_loader": train_loader.state_dict(),
                "step": step,
                "config": args.config,
            }, os.path.join(ckpt_dir, "latest.pt"))
            metrics_f.flush()
            print(f"체크포인트 저장: {ckpt_dir}/latest.pt")

    metrics_f.close()
    print("학습 완료")


if __name__ == "__main__":
    main()
