"""한국어 중심 코퍼스 샘플로 BPE vocab 학습.

코퍼스 혼합 (스트리밍, 전체 다운로드 불필요):
- FineWeb-2 한국어 (kor_Hang)  ~80% — 본 학습 데이터와 같은 분포
- 한국어 위키백과              ~10% — 품질 앵커
- FineWeb-Edu (영어)           ~10% — 영어·코드 토큰도 어느 정도 확보

사용 예:
  python -m tokenizer.train_tokenizer --sample-mb 20 --vocab-size 4096   # 파이프라인 검증
  python -m tokenizer.train_tokenizer --sample-mb 500 --vocab-size 65536 # 본 학습 (수 시간)

결과: tokenizer/tokenizer.json
"""

import argparse
import time

from datasets import load_dataset

from .bpe import ByteBPETokenizer

MIX = [
    # (HF dataset, config, split, 비율)
    ("HuggingFaceFW/fineweb-2", "kor_Hang", "train", 0.8),
    ("wikimedia/wikipedia", "20231101.ko", "train", 0.1),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0.1),
]


def iter_corpus(sample_mb: float):
    """각 소스에서 비율만큼 스트리밍으로 텍스트를 읽어온다."""
    for name, config, split, ratio in MIX:
        budget = int(sample_mb * 1e6 * ratio)
        got = 0
        print(f"[{name}/{config}] {budget/1e6:.0f}MB 수집 중...")
        ds = load_dataset(name, config, split=split, streaming=True)
        for row in ds:
            text = row["text"]
            yield text
            got += len(text.encode("utf-8"))
            if got >= budget:
                break
        print(f"[{name}/{config}] 완료 ({got/1e6:.1f}MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-mb", type=float, default=500)
    ap.add_argument("--vocab-size", type=int, default=65536)
    ap.add_argument("--out", default="tokenizer/tokenizer.json")
    args = ap.parse_args()

    t0 = time.time()
    tok = ByteBPETokenizer.train(iter_corpus(args.sample_mb), vocab_size=args.vocab_size)
    tok.save(args.out)
    print(f"\n학습 완료: vocab {tok.vocab_size}, {time.time()-t0:.0f}s → {args.out}")

    # 압축률 리포트
    samples = {
        "한국어": "대한민국은 동아시아의 한반도 남부에 있는 나라이다. 수도는 서울특별시이며, 국화는 무궁화이다.",
        "영어": "The Republic of Korea is a country in East Asia, occupying the southern half of the Korean Peninsula.",
    }
    for lang, s in samples.items():
        n = len(tok.encode(s))
        print(f"{lang}: {len(s)}자 → {n}토큰 ({len(s)/n:.2f} chars/token)")


if __name__ == "__main__":
    main()
