"""shard .bin에서 학습 배치를 뽑는 데이터로더.

설계: 모든 shard를 memmap으로 열어 하나의 거대한 토큰 스트림처럼 취급하고,
매 step 길이 T+1짜리 윈도우를 무작위 위치에서 샘플링한다.

재개(resume)가 공짜인 이유: 무작위성이 (seed, step)만으로 결정되므로
체크포인트에 step 수만 저장하면 중단 지점의 데이터 순서를 정확히 재현한다.
"""

import glob
import os

import numpy as np
import torch


class ShardedDataLoader:
    def __init__(self, shard_dir: str, split: str, batch_size: int, seq_len: int,
                 seed: int = 42, device: str = "cpu"):
        paths = sorted(glob.glob(os.path.join(shard_dir, f"{split}_*.bin")))
        assert paths, f"{shard_dir}에 {split} shard 없음 — data.prepare 먼저 실행"
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.sizes = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.cum = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total = int(self.cum[-1])
        self.B, self.T = batch_size, seq_len
        self.seed = seed
        self.step = 0
        self.device = device

    def state_dict(self):
        return {"step": self.step, "seed": self.seed}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.seed = state["seed"]

    def next_batch(self):
        rng = np.random.default_rng((self.seed, self.step))
        self.step += 1
        starts = rng.integers(0, self.total - self.T - 1, size=self.B)
        xs = np.empty((self.B, self.T + 1), dtype=np.int64)
        for i, start in enumerate(starts):
            xs[i] = self._read(int(start), self.T + 1)
        batch = torch.from_numpy(xs)
        x, y = batch[:, :-1], batch[:, 1:]
        if self.device != "cpu":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        return x, y

    def _read(self, start: int, n: int) -> np.ndarray:
        """전역 오프셋 start부터 n개 토큰 — shard 경계에 걸치면 이어붙인다."""
        out = np.empty(n, dtype=np.int64)
        got = 0
        si = int(np.searchsorted(self.cum, start, side="right") - 1)
        off = start - int(self.cum[si])
        while got < n:
            take = min(n - got, int(self.sizes[si]) - off)
            out[got : got + take] = self.shards[si][off : off + take]
            got += take
            si += 1
            off = 0
        return out
