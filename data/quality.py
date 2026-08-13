"""문서 품질 분류기 — 웹 문서 중 '지식 밀도가 높은' 것을 골라낸다.

접근: GPT-3와 CCNet이 쓴 방식. 좋은 글의 정의를 사람이 규칙으로 적는 대신,
**이미 품질을 아는 코퍼스와 얼마나 닮았는지**로 판정한다.
  positive = 위키백과·교과서 (사실을 설명하는 글)
  negative = 걸러지지 않은 웹 문서 (광고·후기·잡담이 섞임)
두 집합을 구분하도록 분류기를 학습시킨 뒤, 웹 문서를 점수 매겨 상위만 남긴다.

모델은 해시 특징 위의 로지스틱 회귀다. 수억 문서를 CPU로 훑어야 하므로
가볍고 빨라야 하고, 이 정도로 충분하다는 게 대규모 파이프라인들의 경험이다.
"""

import re

import numpy as np

DIM = 1 << 18  # 해시 버킷 수. 어휘를 미리 만들 필요가 없다 (hashing trick)
MAX_WORDS = 1500  # 문서 앞부분만 봐도 품질 판정에는 충분하다

_word = re.compile(r"[가-힣]+|[a-zA-Z]+|[0-9]+|[^\s가-힣a-zA-Z0-9]")


def _hash(s: str) -> int:
    # 결정론적이어야 학습과 추론이 같은 특징을 본다 (파이썬 hash()는 실행마다 바뀜)
    h = 2166136261
    for c in s.encode("utf-8"):
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return h % DIM


def featurize(text: str) -> tuple[np.ndarray, np.ndarray]:
    """문서 → (해시 인덱스, L2 정규화된 값). 단어 unigram + bigram을 쓴다."""
    words = _word.findall(text[: MAX_WORDS * 12])[:MAX_WORDS]
    if not words:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    idx = [_hash(w) for w in words]
    idx += [_hash(a + "\x00" + b) for a, b in zip(words, words[1:])]
    uniq, counts = np.unique(np.array(idx, dtype=np.int64), return_counts=True)
    vals = np.sqrt(counts).astype(np.float32)  # 빈도의 제곱근 — 긴 문서 편중 완화
    vals /= np.linalg.norm(vals)
    return uniq, vals


class QualityClassifier:
    def __init__(self, w=None, b=0.0):
        self.w = np.zeros(DIM, dtype=np.float32) if w is None else w
        self.b = float(b)

    def score(self, text: str) -> float:
        """0~1. 높을수록 위키·교과서에 가깝다."""
        idx, vals = featurize(text)
        if len(idx) == 0:
            return 0.0
        z = float(self.w[idx] @ vals) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def train(cls, pos_texts, neg_texts, epochs=3, lr=0.5, l2=1e-6, seed=0, verbose=True):
        """SGD 로지스틱 회귀. 특징이 희소해서 갱신도 해당 인덱스에만 일어난다."""
        feats, labels = [], []
        for texts, y in ((pos_texts, 1.0), (neg_texts, 0.0)):
            for t in texts:
                idx, vals = featurize(t)
                if len(idx):
                    feats.append((idx, vals))
                    labels.append(y)
        n = len(feats)
        assert n > 0, "학습 데이터가 비어 있다"
        labels = np.array(labels, dtype=np.float32)
        if verbose:
            print(f"학습 표본 {n}개 (positive {int(labels.sum())} / negative {int(n-labels.sum())})")

        model = cls()
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            order = rng.permutation(n)
            loss_sum = 0.0
            for i in order:
                idx, vals = feats[i]
                z = float(model.w[idx] @ vals) + model.b
                p = 1.0 / (1.0 + np.exp(-z))
                g = p - labels[i]
                model.w[idx] -= lr * (g * vals + l2 * model.w[idx])
                model.b -= lr * g
                loss_sum -= np.log(max(p if labels[i] > 0.5 else 1 - p, 1e-9))
            if verbose:
                preds = np.array([model.score_feat(f) for f in feats])
                acc = ((preds > 0.5) == (labels > 0.5)).mean()
                print(f"  epoch {ep+1}: loss {loss_sum/n:.4f}, 학습 정확도 {acc*100:.1f}%")
        return model

    def score_feat(self, feat) -> float:
        idx, vals = feat
        z = float(self.w[idx] @ vals) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path):
        np.savez_compressed(path, w=self.w, b=np.float32(self.b))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(w=d["w"], b=float(d["b"]))
