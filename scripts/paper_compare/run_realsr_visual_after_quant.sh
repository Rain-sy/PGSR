#!/usr/bin/env bash
set -euo pipefail

cd /jumbo/yuwingtai/sy/Dual_SR
export HF_HOME="${HF_HOME:-/jumbo/yuwingtai/sy/}"

STEMS="Canon_002,Canon_034,Nikon_009,Nikon_015,Nikon_025,Nikon_032,Nikon_041,Nikon_044,Nikon_046"
WAIT_PIDS="${WAIT_PIDS:-1868197 1911230 3767022}"
LOG_ROOT="experiments/paper_compare/logs"
mkdir -p "$LOG_ROOT"
WATCH_LOG="$LOG_ROOT/realsr_visual_after_quant_watch.log"

log() {
  echo "[$(date)] $*" | tee -a "$WATCH_LOG"
}

wait_for_pid() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    log "waiting for quant pid=$pid"
    sleep 300
  done
}

run_method() {
  local method="$1"
  local gpu="$2"
  local log_file="$LOG_ROOT/${method}_RealSR_visual_gpu${gpu}.log"
  log "START visual $method RealSR gpu=$gpu"
  python scripts/paper_compare/run_baseline.py "$method" RealSR \
    --manifest configs/paper_compare/baselines.yaml \
    --execute \
    --allow-needs-command \
    --gpu "$gpu" \
    --include-stems "$STEMS" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  log "END visual $method RealSR rc=$rc"
  return "$rc"
}

log "RealSR visual watcher started"
for pid in $WAIT_PIDS; do
  wait_for_pid "$pid"
done
log "quant queues finished; starting RealSR visual selected set"

(
  set -e
  run_method ResShift 0
  run_method DiffBIR 0
  run_method SinSR 0
) &
pid_gpu0=$!

(
  set -e
  run_method StableSR 1
  run_method SeeSR_g1p5_s40_wavelet 1
) &
pid_gpu1=$!

(
  set -e
  source /jumbo/yuwingtai/sy/miniconda3/etc/profile.d/conda.sh
  conda activate pisasr
  pip install diffusers==0.29.2 2>&1 | tee "$LOG_ROOT/pisasr_diffusers_029_for_realsr_visual.log"
  conda deactivate
  run_method PASD 7
  source /jumbo/yuwingtai/sy/miniconda3/etc/profile.d/conda.sh
  conda activate pisasr
  pip install diffusers==0.25.0 2>&1 | tee "$LOG_ROOT/pisasr_diffusers_025_for_realsr_visual.log"
  conda deactivate
  run_method OSEDiff 7
  run_method PiSA-SR 7
) &
pid_gpu7=$!

rc=0
for pid in "$pid_gpu0" "$pid_gpu1" "$pid_gpu7"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

if [ "$rc" -ne 0 ]; then
  log "one or more RealSR visual jobs failed; skip grid generation"
  exit "$rc"
fi

python scripts/paper_compare/make_visual_grid.py RealSR \
  --manifest configs/paper_compare/baselines.yaml \
  --crops configs/paper_compare/crops.yaml \
  --include-stems "$STEMS" \
  --methods "LR/Bicubic,ResShift,StableSR,DiffBIR,SeeSR_g1p5_s40_wavelet,PASD,OSEDiff,SinSR,PiSA-SR,HR" \
  --crop-size 256 2>&1 | tee "$LOG_ROOT/realsr_visual_grid.log"

log "RealSR visual selected set complete"
