"""Annealing(학습률 decay) 구간용 고품질 데이터 준비.

사전학습 본 구간은 웹 데이터가 주력이지만, 학습률이 0으로 수렴하는 마지막
구간에서 본 데이터는 거의 그대로 모델에 남는다. 그래서 이 구간에는 지식 밀도가
높은 데이터를 넣는다 (Llama 3, MiniCPM, OLMo 2가 쓰는 방식).

혼합 비율은 각 코퍼스를 품질 분류기로 실측한 결과를 근거로 정했다:
    위키백과   중앙값 0.986   교과서 0.995~0.998   나무위키 0.400
    FineWeb-2 ko 0.009        FineWeb-Edu(영어) 0.023

품질 필터는 **한국어 웹과 나무위키에만** 건다. 분류기가 한국어 positive로
학습돼서 영어는 품질과 무관하게 낮은 점수를 받기 때문이다 (영어에 걸면 99.8%가
버려진다). 위키·교과서는 이미 positive 클래스라 거를 이유가 없다.

사용: python -m data.prepare_anneal --tokens 5e9 --out-dir data/shards_anneal
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from tokenizer.bpe import ByteBPETokenizer

from .clean import MinHashDeduper, keep_document
from .prepare import SHARD_TOKENS, ShardWriter
from .quality import QualityClassifier

# repeat: 코퍼스가 예산보다 작을 때 반복 횟수. annealing에서 고품질 데이터를
# 몇 번 반복하는 건 표준 관행이다.
SOURCES = [
    dict(label="위키백과", name="wikimedia/wikipedia", config="20231101.ko",
         ratio=0.30, threshold=None, repeat=3, korean=True),
    dict(label="나무위키", name="heegyu/namuwiki-extracted", config=None,
         ratio=0.20, threshold=0.3, repeat=1, korean=True),
    dict(label="교과서(wikidata)", name="maywell/korean_textbooks", config="ko_wikidata",
         ratio=0.10, threshold=None, repeat=1, korean=True),
    dict(label="교과서(tiny)", name="maywell/korean_textbooks", config="tiny-textbooks",
         ratio=0.10, threshold=None, repeat=1, korean=True),
    dict(label="선별 웹", name="HuggingFaceFW/fineweb-2", config="kor_Hang",
         ratio=0.15, threshold=0.3, repeat=1, korean=True),
    dict(label="영어 Edu", name="HuggingFaceFW/fineweb-edu", config="sample-10BT",
         ratio=0.10, threshold=None, repeat=1, korean=False),
    dict(label="instruction", name="nlpai-lab/kullm-v2", config=None,
         ratio=0.05, threshold=None, repeat=4, korean=True, instruction=True),
]

_tok = None
_qc = None
_deduper = None


def _init_worker(tok_path, qc_path):
    global _tok, _qc, _deduper
    _tok = ByteBPETokenizer.load(tok_path)
    _qc = QualityClassifier.load(qc_path)
    _deduper = MinHashDeduper()


def _process(job):
    """워커: 필터 → 품질 점수 → (필요시) MinHash 서명 → 토크나이즈.

    dedup=False인 소스는 서명을 만들지 않는다. 위키·교과서처럼 이미 정제된
    코퍼스는 내부 중복이 없고, 오히려 **의도적으로 여러 번 반복해서 넣는다**.
    여기에 중복 제거를 걸면 2회차부터 전부 버려져 반복이 무효가 된다
    (2026-08-13에 실제로 발생: 위키 3회 반복이 1회분만 남았다).
    """
    text, threshold, korean, is_instruction, dedup = job
    if is_instruction:
        # SFT와 같은 chat 형식으로 넣는다 — 이후 SFT 효과가 좋아진다는 게 최신 관행
        ids = _tok.encode(text, allowed_special=True) + [_tok.special_tokens["<|eos|>"]]
        return (_deduper.band_keys(text[:2000]) if dedup else []), ids
    if not keep_document(text, korean_source=korean):
        return None, None
    if threshold is not None and _qc.score(text) < threshold:
        return None, None
    keys = _deduper.band_keys(text[:2000]) if dedup else []
    return keys, _tok.encode(text) + [_tok.special_tokens["<|eos|>"]]


def build_stream(src, target_tokens):
    """한 소스에서 (원문, 임계값, 한국어여부, instruction여부)를 repeat만큼 생산."""
    from datasets import load_dataset

    for r in range(src["repeat"]):
        ds = load_dataset(src["name"], src["config"], split="train", streaming=True)
        for row in ds:
            if src.get("instruction"):
                user = row["instruction"]
                if row.get("input"):
                    user += "\n\n" + row["input"]
                text = (f"<|im_start|>user\n{user}<|im_end|>\n"
                        f"<|im_start|>assistant\n{row['output']}<|im_end|>")
            else:
                text = row["text"]
            if text:
                yield (text, src["threshold"], src["korean"],
                       bool(src.get("instruction")), src.get("dedup", True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=5e9)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--quality", default="data/quality_ko.npz")
    ap.add_argument("--out-dir", default="data/shards_anneal")
    ap.add_argument("--val-ratio", type=float, default=0.004)
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    target = int(args.tokens)

    streams = []
    for src in SOURCES:
        budget = int(target * src["ratio"])
        streams.append({**src, "budget": budget, "got": 0,
                        "it": build_stream(src, budget), "seen": 0, "kept": 0})

    deduper = MinHashDeduper()
    train_w = ShardWriter(args.out_dir, "train")
    val_w = ShardWriter(args.out_dir, "val")
    n_docs = n_filtered = n_dup = 0
    t_start = time.time()

    def jobs():
        """소스 라운드로빈. 예산을 채운 소스는 빠진다."""
        nonlocal n_docs
        active = list(streams)
        while active:
            for s in list(active):
                if s["got"] >= s["budget"]:
                    active.remove(s)
                    print(f"[{s['label']}] 목표 도달 {s['got']/1e6:.0f}M tokens "
                          f"(문서 {s['seen']} 중 {s['kept']} 채택)", flush=True)
                    continue
                try:
                    job = next(s["it"])
                except StopIteration:
                    active.remove(s)
                    print(f"[{s['label']}] 소진 {s['got']/1e6:.0f}M tokens "
                          f"(문서 {s['seen']} 중 {s['kept']} 채택)", flush=True)
                    continue
                s["seen"] += 1
                n_docs += 1
                yield s, job

    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.tokenizer, args.quality)) as pool:
        pending = []

        def payloads():
            for s, job in jobs():
                pending.append(s)
                yield job

        for keys, ids in pool.imap(_process, payloads(), chunksize=8):
            s = pending.pop(0)
            if keys is None:
                n_filtered += 1
                continue
            if keys and deduper.check_keys(keys):  # keys가 비면 dedup 대상이 아니다
                n_dup += 1
                continue
            s["got"] += len(ids)
            s["kept"] += 1
            if val_w.total < target * args.val_ratio:
                val_w.add(ids)
            else:
                train_w.add(ids)
            done = train_w.total + val_w.total
            if done % 10_000_000 < len(ids):
                rate = done / (time.time() - t_start)
                print(f"진행: {done/1e6:.0f}M / {target/1e6:.0f}M tokens "
                      f"(문서 {n_docs}, 필터 {n_filtered}, 중복 {n_dup}) "
                      f"| {rate/1e6:.2f}M tok/s | 남은 시간 ~{(target-done)/rate/3600:.1f}시간",
                      flush=True)

    train_w.close()
    val_w.close()
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({
            "purpose": "annealing (LR decay 구간용 고품질 혼합)",
            "tokenizer": args.tokenizer,
            "quality_model": args.quality,
            "shard_tokens": SHARD_TOKENS,
            "train_tokens": train_w.total,
            "val_tokens": val_w.total,
            "docs": n_docs, "filtered": n_filtered, "duplicates": n_dup,
            "sources": [{k: s[k] for k in ("label", "name", "config", "ratio",
                                           "threshold", "repeat")} | {"got": s["got"],
                        "seen": s["seen"], "kept": s["kept"]} for s in streams],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n완료: train {train_w.total/1e6:.1f}M / val {val_w.total/1e6:.1f}M tokens", flush=True)
    os._exit(0)  # pyarrow 스레드풀 종료 데드락 우회 (prepare.py와 동일)


if __name__ == "__main__":
    main()
