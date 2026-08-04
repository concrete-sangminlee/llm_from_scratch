"""문서 정제 필터.

FineWeb-2는 이미 강하게 정제된 코퍼스라(언어 필터, dedup, 품질 필터 적용됨)
여기서는 가벼운 안전망만 둔다: 길이, 한글 비율(한국어 소스에만), 반복 비율.
MinHash 근사 중복 제거는 위키 등 소스 간 중복 제거용.
"""

import hashlib
import re

_hangul = re.compile(r"[가-힣]")


def hangul_ratio(text: str) -> float:
    """공백 제외 문자 중 한글 음절 비율."""
    stripped = re.sub(r"\s", "", text)
    if not stripped:
        return 0.0
    return len(_hangul.findall(stripped)) / len(stripped)


def repetition_ratio(text: str, n: int = 20) -> float:
    """n-gram(문자) 중복 비율 — 보일러플레이트·스팸 감지."""
    if len(text) < n * 2:
        return 0.0
    grams = [text[i : i + n] for i in range(0, len(text) - n, n)]
    return 1.0 - len(set(grams)) / len(grams)


def keep_document(text: str, korean_source: bool = True) -> bool:
    if len(text) < 200:  # 너무 짧은 문서는 문맥 학습에 도움 안 됨
        return False
    if korean_source and hangul_ratio(text) < 0.3:
        return False
    if repetition_ratio(text) > 0.5:
        return False
    return True


class MinHashDeduper:
    """MinHash 서명 기반 근사 중복 제거.

    문서를 문자 5-gram 집합으로 보고, num_perm개의 해시 함수 각각에 대한 최소값을
    서명으로 삼는다. 두 문서의 서명이 밴드 단위로 하나라도 일치하면 중복으로 판정
    (LSH banding). 수백만 문서까지 메모리 내 처리 가능.
    """

    _MERSENNE = (1 << 61) - 1

    def __init__(self, num_perm: int = 64, bands: int = 8, ngram: int = 5, seed: int = 42):
        assert num_perm % bands == 0
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.ngram = ngram
        self.seen: list[set[bytes]] = [set() for _ in range(bands)]
        # gram당 해시는 1번만 하고, num_perm개의 순열은 (a*h+b) mod p 로 근사
        import random

        rng = random.Random(seed)
        self.perms = [
            (rng.randrange(1, self._MERSENNE), rng.randrange(self._MERSENNE))
            for _ in range(num_perm)
        ]

    def _signature(self, text: str) -> list[int]:
        grams = {text[i : i + self.ngram] for i in range(len(text) - self.ngram + 1)}
        if not grams:
            return [0] * self.num_perm
        hashes = [
            int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
            for g in grams
        ]
        return [min((a * h + b) % self._MERSENNE for h in hashes) for a, b in self.perms]

    def is_duplicate(self, text: str) -> bool:
        """중복이면 True. 아니면 등록하고 False."""
        sig = self._signature(text)
        keys = []
        dup = False
        for b in range(self.bands):
            band = sig[b * self.rows : (b + 1) * self.rows]
            key = hashlib.blake2b(str(band).encode(), digest_size=8).digest()
            if key in self.seen[b]:
                dup = True
            keys.append(key)
        if not dup:
            for b, key in enumerate(keys):
                self.seen[b].add(key)
        return dup
