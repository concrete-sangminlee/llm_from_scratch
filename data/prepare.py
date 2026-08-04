"""코퍼스 스트리밍 → 정제 → 토크나이즈 → shard .bin 생성.

다운로드·정제·토크나이즈를 한 번의 스트리밍 패스로 처리한다 (원문 저장 안 함 —
20B 토큰이면 원문만 ~80GB라 디스크 절약). 결과물:

  data/shards/train_000000.bin, train_000001.bin, ...  (uint16 토큰 id)
  data/shards/val_000000.bin
  data/shards/meta.json                                (토크나이저 경로, shard 크기, 통계)

각 문서는 <|eos|>로 끝난다. 문서 순서는 소스가 섞이도록 라운드로빈.

사용 예:
  python -m data.prepare --tokens 100e6   # 검증용 0.1B
  python -m data.prepare --tokens 12e9    # 본 학습용 (서버에서, 수일)
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from datasets import load_dataset

from tokenizer.bpe import ByteBPETokenizer

from .clean import MinHashDeduper, keep_document

# (HF dataset, config, 비율, 한국어 소스 여부)
MIX = [
    ("HuggingFaceFW/fineweb-2", "kor_Hang", 0.75, True),
    ("wikimedia/wikipedia", "20231101.ko", 0.10, True),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.15, False),
]

SHARD_TOKENS = 50_000_000  # shard당 5천만 토큰 = 100MB(uint16)

_tok = None


def _init_worker(tok_path):
    global _tok
    _tok = ByteBPETokenizer.load(tok_path)


def _encode_doc(text):
    return _tok.encode(text) + [_tok.special_tokens["<|eos|>"]]


class ShardWriter:
    def __init__(self, out_dir, split):
        self.out_dir = out_dir
        self.split = split
        self.buf = np.empty(SHARD_TOKENS, dtype=np.uint16)
        self.fill = 0
        self.shard_idx = 0
        self.total = 0

    def add(self, ids):
        arr = np.asarray(ids, dtype=np.uint16)
        self.total += len(arr)
        while len(arr) > 0:
            n = min(len(arr), SHARD_TOKENS - self.fill)
            self.buf[self.fill : self.fill + n] = arr[:n]
            self.fill += n
            arr = arr[n:]
            if self.fill == SHARD_TOKENS:
                self._flush()

    def _flush(self):
        if self.fill == 0:
            return
        path = os.path.join(self.out_dir, f"{self.split}_{self.shard_idx:06d}.bin")
        self.buf[: self.fill].tofile(path)
        print(f"  저장: {path} ({self.fill/1e6:.1f}M tokens)")
        self.shard_idx += 1
        self.fill = 0

    def close(self):
        self._flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=100e6, help="목표 토큰 수")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out-dir", default="data/shards")
    ap.add_argument("--val-ratio", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    target = int(args.tokens)

    streams = []
    for name, config, ratio, korean in MIX:
        ds = load_dataset(name, config, split="train", streaming=True)
        streams.append({"it": iter(ds), "budget": int(target * ratio), "got": 0,
                        "korean": korean, "name": f"{name}/{config}"})

    deduper = MinHashDeduper()
    train_w = ShardWriter(args.out_dir, "train")
    val_w = ShardWriter(args.out_dir, "val")
    n_docs = n_filtered = n_dup = 0

    def doc_stream():
        """소스 라운드로빈 + 필터링된 문서 텍스트를 생산."""
        nonlocal n_docs, n_filtered, n_dup
        active = list(streams)
        while active:
            for s in list(active):
                if s["got"] >= s["budget"]:
                    active.remove(s)
                    print(f"[{s['name']}] 목표 도달 ({s['got']/1e6:.0f}M tokens)")
                    continue
                try:
                    text = next(s["it"])["text"]
                except StopIteration:
                    active.remove(s)
                    continue
                n_docs += 1
                if not keep_document(text, korean_source=s["korean"]):
                    n_filtered += 1
                    continue
                if deduper.is_duplicate(text[:2000]):
                    n_dup += 1
                    continue
                yield s, text

    with Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        # imap의 순서 보존을 이용해 (소스, 결과)를 짝지어 예산을 갱신한다
        pending_sources = []

        def texts():
            for s, text in doc_stream():
                pending_sources.append(s)
                yield text

        for ids in pool.imap(_encode_doc, texts(), chunksize=16):
            s = pending_sources.pop(0)
            s["got"] += len(ids)
            # 문서 단위로 train/val 분리 (같은 문서가 양쪽에 걸치지 않게)
            if val_w.total < target * args.val_ratio:
                val_w.add(ids)
            else:
                train_w.add(ids)
            if (train_w.total + val_w.total) % 10_000_000 < len(ids):
                done = train_w.total + val_w.total
                print(f"진행: {done/1e6:.0f}M / {target/1e6:.0f}M tokens "
                      f"(문서 {n_docs}, 필터 {n_filtered}, 중복 {n_dup})")

    train_w.close()
    val_w.close()
    meta = {
        "tokenizer": args.tokenizer,
        "shard_tokens": SHARD_TOKENS,
        "train_tokens": train_w.total,
        "val_tokens": val_w.total,
        "docs": n_docs,
        "filtered": n_filtered,
        "duplicates": n_dup,
        "mix": [(m[0], m[1], m[2]) for m in MIX],
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n완료: train {train_w.total/1e6:.1f}M / val {val_w.total/1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
