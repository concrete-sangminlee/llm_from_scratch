#!/bin/bash
# 장기 학습 감시 스크립트.
#
# 43일짜리 학습은 일시적 OOM(같은 GPU의 다른 프로세스), 네트워크 끊김, 드라이버
# 하이컵 등으로 죽을 수 있다. 그때마다 사람이 붙어 있을 수 없으므로, 죽으면
# 마지막 체크포인트에서 자동으로 이어받는다.
#
# 사용: ./run_training.sh <config> <shard-dir>
set -u

CONFIG="${1:-base_1b}"
SHARD_DIR="${2:-data/shards_26b}"
CKPT="checkpoints/$CONFIG/latest.pt"
LOG="train_$CONFIG.log"
MAX_RETRIES=100
RETRY_WAIT=120

cd "$(dirname "$0")"
# 메모리 단편화 완화 — 같은 GPU를 다른 프로세스와 공유할 때 도움이 된다
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

for attempt in $(seq 0 $MAX_RETRIES); do
    if [ -f "$CKPT" ]; then
        echo "=== [$(date '+%F %T')] 재개 (시도 $attempt) ===" >> "$LOG"
        .venv/bin/python train.py --config "$CONFIG" --shard-dir "$SHARD_DIR" --resume "$CKPT" >> "$LOG" 2>&1
    else
        echo "=== [$(date '+%F %T')] 시작 ===" >> "$LOG"
        .venv/bin/python train.py --config "$CONFIG" --shard-dir "$SHARD_DIR" >> "$LOG" 2>&1
    fi
    code=$?

    if [ $code -eq 0 ]; then
        echo "=== [$(date '+%F %T')] 정상 종료 ===" >> "$LOG"
        exit 0
    fi

    echo "=== [$(date '+%F %T')] 비정상 종료(코드 $code), ${RETRY_WAIT}초 후 재시도 ===" >> "$LOG"
    sleep $RETRY_WAIT
done

echo "=== [$(date '+%F %T')] 재시도 한도 초과, 중단 ===" >> "$LOG"
exit 1
