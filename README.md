# llm_from_scratch

한국어 LLM을 완전히 처음부터 만드는 프로젝트. 토크나이저(byte-level BPE)부터 최신 아키텍처
트랜스포머(RoPE, GQA, SwiGLU, RMSNorm, QK-Norm), 사전학습 루프, SFT까지 전부 순수 PyTorch로
직접 구현한다.

## 목표

- **Base 모델**: ~400M 파라미터, 한국어 중심 12–20B 토큰 사전학습 (RTX 6000 Ada 48GB ×1)
- **Chat 모델**: 한국어 instruction 데이터로 SFT
- HuggingFace `transformers` 미사용 — 모델·토크나이저·학습 루프 전부 직접 구현
  (`datasets`는 코퍼스 다운로드 용도로만 사용)

## 구조

```
configs/            # 모델·학습 설정 (debug_30m, mid_125m, base_400m, sft_400m)
tokenizer/          # byte-level BPE 직접 구현 + 한국어 코퍼스로 vocab 학습
data/               # 코퍼스 다운로드·정제·토크나이즈·데이터로더
model.py            # decoder-only 트랜스포머 (RoPE/GQA/SwiGLU/RMSNorm/QK-Norm)
train.py            # 사전학습 루프 (bf16, torch.compile, WSD 스케줄, 체크포인트 재개)
sft.py              # SFT 루프 (chat template, loss masking)
sample.py           # 텍스트 생성 / chat 데모
eval.py             # perplexity + KOBEST 평가
```

## 셋업

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 진행 단계

- [ ] Stage 1: byte-level BPE 토크나이저 (64K vocab, 한국어 학습)
- [ ] Stage 2: 데이터 파이프라인 (FineWeb-2 ko + 위키 + FineWeb-Edu)
- [ ] Stage 3: 모델 구현 + 오버핏 검증
- [ ] Stage 4: 사전학습 (30M → 125M → 400M)
- [ ] Stage 5: 평가 (PPL, KOBEST)
- [ ] Stage 6: SFT
