"""annealing 후보 코퍼스의 실제 가용 토큰 수를 측정한다.

앞서 parquet 압축 크기로 어림한 값이 크게 빗나갔다(위키백과 750M 추정 → 실제 260M).
그래서 두 정보를 조합해 추정한다:
  - 전체 문서 수: 데이터셋 메타데이터에서 읽는다 (다운로드 없이)
  - 문서당 평균 토큰 + 필터 통과율: 스트리밍 표본을 실제로 토크나이즈해서 구한다

주의: load_dataset(streaming=False)로 문서 수를 얻으면 데이터셋 전체를 내려받는다.
반드시 load_dataset_builder의 메타데이터를 쓸 것.

사용: python -m data.measure_sources
"""

import argparse
import itertools

from datasets import get_dataset_config_names, load_dataset, load_dataset_builder

from tokenizer.bpe import ByteBPETokenizer

from .clean import keep_document
from .quality import QualityClassifier


def num_examples(name, config):
    b = load_dataset_builder(name, config)
    split = b.info.splits.get("train") if b.info.splits else None
    return split.num_examples if split else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--quality", default="data/quality_ko.npz")
    args = ap.parse_args()

    tok = ByteBPETokenizer.load(args.tokenizer)
    qc = QualityClassifier.load(args.quality)

    sources = [
        ("위키백과", "wikimedia/wikipedia", "20231101.ko", None),
        ("나무위키", "heegyu/namuwiki-extracted", None, 0.3),
    ]
    for c in sorted(get_dataset_config_names("maywell/korean_textbooks")):
        sources.append((f"교과서:{c}", "maywell/korean_textbooks", c, None))

    print(f"{'코퍼스':<34}{'문서수':>11}{'통과율':>8}{'토큰/문서':>10}{'가용토큰':>11}")
    print("-" * 74)
    total = 0
    for label, name, config, th in sources:
        try:
            n_docs = num_examples(name, config)
            if not n_docs:
                print(f"{label:<34}{'문서수 미상':>11}")
                continue
            ds = load_dataset(name, config, split="train", streaming=True)
            kept = toks = seen = 0
            for r in itertools.islice(ds, args.sample):
                seen += 1
                t = r.get("text") or ""
                if not keep_document(t, korean_source=True):
                    continue
                if th is not None and qc.score(t) < th:
                    continue
                kept += 1
                toks += len(tok.encode(t))
            rate = kept / seen if seen else 0
            per = toks / kept if kept else 0
            est = n_docs * rate * per
            total += est
            print(f"{label:<34}{n_docs:>11,}{rate*100:7.1f}%{per:10.0f}{est/1e6:10.0f}M")
        except Exception as e:
            print(f"{label:<34}실패: {type(e).__name__} {str(e)[:30]}")

    print("-" * 74)
    print(f"{'합계 (한국어 고품질, 1회분)':<34}{'':>11}{'':>8}{'':>10}{total/1e6:10.0f}M")


if __name__ == "__main__":
    main()
