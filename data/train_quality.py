"""품질 분류기 학습 + 검증.

positive: 한국어 위키백과, 한국어 교과서형 텍스트
negative: FineWeb-2 한국어 원본 (품질 필터 전)

학습 후 held-out 정확도와 실제 웹 문서 점수 분포를 출력해, 임계값을 정할 근거를 준다.

사용: python -m data.train_quality --n 8000 --out data/quality_ko.npz
"""

import argparse
import itertools

import numpy as np
from datasets import load_dataset

from .quality import QualityClassifier


def take(name, config, field, n, skip=0):
    ds = load_dataset(name, config, split="train", streaming=True)
    out = []
    for r in itertools.islice(ds, skip, skip + n * 3):
        t = r[field]
        if t and len(t) > 300:
            out.append(t)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000, help="클래스당 학습 문서 수")
    ap.add_argument("--out", default="data/quality_ko.npz")
    args = ap.parse_args()
    n = args.n
    holdout = max(200, n // 10)

    print("코퍼스 수집 중...", flush=True)
    wiki = take("wikimedia/wikipedia", "20231101.ko", "text", n // 2 + holdout)
    book = take("maywell/korean_textbooks", "ko_wikidata", "text", n // 2 + holdout)
    web = take("HuggingFaceFW/fineweb-2", "kor_Hang", "text", n + 2 * holdout)
    print(f"위키 {len(wiki)} / 교과서 {len(book)} / 웹 {len(web)}", flush=True)

    pos = wiki[: n // 2] + book[: n // 2]
    neg = web[:n]
    pos_ho = wiki[n // 2 :] + book[n // 2 :]
    neg_ho = web[n:]

    model = QualityClassifier.train(pos, neg, epochs=3)

    # held-out 성능 — 학습에 안 쓴 문서로 진짜 구분력을 잰다
    ps = np.array([model.score(t) for t in pos_ho])
    ns = np.array([model.score(t) for t in neg_ho])
    acc = ((ps > 0.5).sum() + (ns <= 0.5).sum()) / (len(ps) + len(ns))
    print(f"\nheld-out 정확도: {acc*100:.1f}%  (positive {len(ps)} / negative {len(ns)})")
    print(f"  위키·교과서 점수 중앙값: {np.median(ps):.3f}")
    print(f"  웹 점수 중앙값:          {np.median(ns):.3f}")

    print("\n웹 문서 점수 분포 — 임계값별 통과 비율:")
    for th in (0.3, 0.5, 0.7, 0.8, 0.9):
        print(f"  {th:.1f} 이상: {(ns >= th).mean()*100:5.1f}%")

    # 실제로 어떤 글이 걸러지는지 눈으로 확인
    order = np.argsort(ns)
    print("\n[최고점 웹 문서]")
    for i in order[-2:]:
        print(f"  ({ns[i]:.3f}) {neg_ho[i][:110]}".replace("\n", " "))
    print("[최저점 웹 문서]")
    for i in order[:2]:
        print(f"  ({ns[i]:.3f}) {neg_ho[i][:110]}".replace("\n", " "))

    model.save(args.out)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
