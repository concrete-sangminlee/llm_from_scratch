"""정규식 기반 pre-tokenization.

BPE를 문서 전체에 그대로 돌리면 "저는 오늘" 같은 공백을 넘나드는 merge가 생겨
토큰이 문법 단위와 어긋난다. 그래서 GPT-2 이후 모든 최신 토크나이저는 먼저
정규식으로 텍스트를 chunk 단위로 자르고, BPE merge는 chunk 내부에서만 일어나게 한다.

여기서는 GPT-4(cl100k_base)의 분할 패턴을 기반으로 한다:
- 축약형 ('s, 'll 등) 분리 — 영어 혼합 텍스트 대응
- 선행 공백 1개를 단어에 붙임 (" 안녕" 형태) — 공백 정보를 토큰에 보존
- 숫자는 최대 3자리씩 분할 — 긴 수를 자릿수 단위로 일반화
- 한글은 \\p{L}(letter)로 매칭되므로 어절("안녕하세요는")이 한 chunk가 되고,
  조사·어미 분리는 BPE 통계가 chunk 내부에서 학습한다.
"""

import regex

# GPT-4 (cl100k_base) 분할 패턴. possessive quantifier(++, ?+)로 백트래킹을 방지한다.
GPT4_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?+\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)

_compiled = regex.compile(GPT4_SPLIT_PATTERN)


def pretokenize(text: str) -> list[str]:
    """텍스트를 BPE가 적용될 chunk 리스트로 분할한다."""
    return _compiled.findall(text)
