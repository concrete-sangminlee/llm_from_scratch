"""모델 검증 관문 (Stage 3):
1. 명시적 attention과 SDPA 출력 일치
2. 초기 loss ≈ ln(vocab_size)
3. 단일 배치 오버핏 → loss ~0
4. KV cache 생성 == cache 없는 full forward의 greedy 결과

실행: python test_model.py
"""

import math

import torch

from model import GPT, ModelConfig


def main():
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1000, dim=128, n_layers=4, n_heads=8, n_kv_heads=2, max_seq_len=128)
    model = GPT(cfg)
    print(f"파라미터: {model.num_params()/1e6:.2f}M (임베딩 제외)")

    idx = torch.randint(0, cfg.vocab_size, (2, 64))
    targets = torch.randint(0, cfg.vocab_size, (2, 64))

    # 1) explicit vs SDPA 일치
    model.eval()
    with torch.no_grad():
        logits_sdpa, _ = model(idx, targets)
        logits_expl, _ = model(idx, targets, explicit=True)
    diff = (logits_sdpa - logits_expl).abs().max().item()
    assert diff < 1e-4, f"attention 경로 불일치: {diff}"
    print(f"1) explicit vs SDPA 일치 (max diff {diff:.2e})")

    # 2) 초기 loss ≈ ln(V)
    with torch.no_grad():
        _, loss = model(idx, targets)
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 0.5, f"초기 loss {loss.item():.2f} vs ln(V)={expected:.2f}"
    print(f"2) 초기 loss {loss.item():.3f} ≈ ln({cfg.vocab_size}) = {expected:.3f}")

    # 3) 단일 배치 오버핏
    model.train()
    opt = torch.optim.AdamW(model.param_groups(0.0), lr=3e-3)
    x, y = idx[:, :-1], idx[:, 1:]
    for step in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"오버핏 실패: loss {loss.item():.3f}"
    print(f"3) 단일 배치 오버핏: 300 step 후 loss {loss.item():.4f}")

    # 4) KV cache greedy 생성 == full forward greedy
    model.eval()
    prompt = idx[:, :8]
    gen_cache = model.generate(prompt, max_new_tokens=16, temperature=0.0)
    # cache 없이: 매 step 전체 시퀀스를 다시 forward
    out = prompt.clone()
    with torch.no_grad():
        for _ in range(16):
            logits, _ = model(out)
            out = torch.cat([out, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(gen_cache, out), "KV cache 생성 결과 불일치"
    print("4) KV cache 생성 == full forward greedy")

    print("\n모든 검증 통과")


if __name__ == "__main__":
    main()
