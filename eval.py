"""평가: held-out perplexity + KOBEST few-shot.

perplexity: val shard에서 exp(평균 next-token loss). 낮을수록 좋다.

KOBEST(BoolQ·COPA·HellaSwag): 베이스 모델은 지시를 못 따르므로 log-likelihood
스코어링을 쓴다 — 문제를 텍스트로 만들고 각 선택지를 이어붙인 뒤, 선택지 부분의
토큰 로그확률 합이 가장 높은 것을 모델의 답으로 간주 (GPT-3 방식).

사용 예:
  python eval.py --ckpt checkpoints/base_400m/latest.pt --ppl
  python eval.py --ckpt checkpoints/base_400m/latest.pt --kobest copa --shots 5
"""

import argparse

import torch
import torch.nn.functional as F

from data.dataloader import ShardedDataLoader
from sample import load_model


@torch.no_grad()
def perplexity(model, loader, iters, device_type):
    losses = []
    for _ in range(iters):
        x, y = loader.next_batch()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    mean = sum(losses) / len(losses)
    return torch.exp(torch.tensor(mean)).item(), mean


@torch.no_grad()
def choice_logprob(model, tok, context: str, choice: str, device) -> float:
    """context 뒤에 choice가 이어질 로그확률 합 (choice 토큰 구간만)."""
    ctx_ids = tok.encode(context)
    full_ids = tok.encode(context + choice)
    x = torch.tensor([full_ids[:-1]], device=device)
    y = full_ids[1:]
    logits, _ = model(x, targets=torch.tensor([y], device=device))
    logp = F.log_softmax(logits[0].float(), dim=-1)
    start = len(ctx_ids) - 1  # y에서 choice 첫 토큰의 위치
    return sum(logp[t, y[t]].item() for t in range(start, len(y)))


def format_copa(row):
    conn = "왜냐하면" if row["question"] == "원인" else "그래서"
    ctx = f"{row['premise']} {conn} "
    return ctx, [row["alternative_1"], row["alternative_2"]], row["label"]


def format_boolq(row):
    ctx = f"{row['paragraph']}\n질문: {row['question']}\n답: "
    return ctx, ["아니오", "예"], row["label"]


def format_hellaswag(row):
    return row["context"] + " ", [row[f"ending_{i}"] for i in range(1, 5)], row["label"]


FORMATTERS = {"copa": format_copa, "boolq": format_boolq, "hellaswag": format_hellaswag}


def eval_kobest(model, tok, task, shots, device, limit=None):
    from datasets import load_dataset

    ds = load_dataset("skt/kobest_v1", task)
    train, test = ds["train"], ds["test"]
    fmt = FORMATTERS[task]

    # few-shot 예시는 train 앞쪽에서 고정으로 뽑는다 (재현성)
    prefix = ""
    for i in range(shots):
        ctx, choices, label = fmt(train[i])
        prefix += ctx + choices[label] + "\n\n"

    correct = total = 0
    rows = test if limit is None else test.select(range(min(limit, len(test))))
    for row in rows:
        ctx, choices, label = fmt(row)
        scores = [choice_logprob(model, tok, prefix + ctx, c, device) for c in choices]
        correct += int(max(range(len(scores)), key=scores.__getitem__) == label)
        total += 1
    return correct / total, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--ppl", action="store_true")
    ap.add_argument("--shard-dir", default="data/shards")
    ap.add_argument("--kobest", choices=list(FORMATTERS), default=None)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu" if device == "cpu" else "mps"
    model, tok = load_model(args.ckpt, args.tokenizer, device)

    if args.ppl:
        loader = ShardedDataLoader(args.shard_dir, "val", 8, model.cfg.max_seq_len,
                                   seed=123, device=device)
        ppl, loss = perplexity(model, loader, 20, device_type)
        print(f"val perplexity: {ppl:.2f} (loss {loss:.4f})")

    if args.kobest:
        acc, n = eval_kobest(model, tok, args.kobest, args.shots, device, args.limit)
        chance = {"copa": 0.5, "boolq": 0.5, "hellaswag": 0.25}[args.kobest]
        print(f"KOBEST {args.kobest} ({args.shots}-shot, n={n}): "
              f"정확도 {acc*100:.1f}% (무작위 기준 {chance*100:.0f}%)")


if __name__ == "__main__":
    main()
