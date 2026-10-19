#!/usr/bin/env bash
# Pairwise stitch, then full and regional HM. One GPU, datasets in series.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/pairwise_all.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== pairwise start $(date '+%F %T') gpu 1 ==="
STITCH=/home/test/anaconda3/envs/UDIS2/bin/python
EVAL=/home/test/anaconda3/envs/dwsplat/bin/python
for DATASET in lyft1920 lyft1224 widedrive3 nuscenes; do
  echo "=== stitch ${DATASET} $(date '+%F %T') ==="
  "$STITCH" -u stitch_drive.py --dataset "$DATASET" --all --metric --methods pairwise --gpu 1 \
    || echo "STITCH FAIL ${DATASET}"
  echo "=== eval ${DATASET} $(date '+%F %T') ==="
  "$EVAL" -u eval_pairwise_dataset.py --dataset "$DATASET" \
    || echo "EVAL FAIL ${DATASET}"
done
echo "=== eval lyft pool $(date '+%F %T') ==="
"$EVAL" -u eval_pairwise_dataset.py --dataset lyft || echo "EVAL FAIL lyft"
echo "=== pairwise done $(date '+%F %T') ==="
