#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/reviewer_experiments}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-pyg_2}"
RESULTS_ROOT="${RESULTS_ROOT:-results_vast}"
MERGED_DIR="${MERGED_DIR:-results_vast_merged}"
RCLONE_REMOTE="${RCLONE_REMOTE:-}"
RCLONE_DEST_DIR="${RCLONE_DEST_DIR:-TEDS_GNN_reviewer_results}"

MODELS=(${MODELS:-gcn gin gat})
GRAPH_TYPES=(${GRAPH_TYPES:-statistical fully_connected})
FOLDS=(${FOLDS:-1 2 3 4 5})
EXTRA_ARGS=("$@")
TRAIN_ARGS=()
if [[ "${NO_PROGRESS:-1}" != "0" ]]; then
  TRAIN_ARGS+=(--no-progress)
fi

ts() { date '+%Y-%m-%d %H:%M:%S'; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[$(ts)] ===== run_vast_teds_parallel start ====="
if [[ "${SKIP_SETUP:-0}" == "1" ]]; then
  echo "[$(ts)] skipping setup because SKIP_SETUP=1"
else
  if [[ -f "${SCRIPT_DIR}/setup_vast_teds.sh" ]]; then
    bash "${SCRIPT_DIR}/setup_vast_teds.sh"
  else
    bash "${REPO_DIR}/setup_vast_teds.sh"
  fi
fi

source "$CONDA_SH"
conda activate "$ENV_NAME"

if [[ -n "${RCLONE_CONF_B64:-}" ]]; then
  mkdir -p /root/.config/rclone
  printf "%s" "$RCLONE_CONF_B64" | base64 -d > /root/.config/rclone/rclone.conf
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  GPU_COUNT=0
fi
if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "[$(ts)] ERROR: no CUDA GPU detected."
  exit 1
fi
echo "[$(ts)] detected_gpus=$GPU_COUNT"

cd "$REPO_DIR"
mkdir -p "$RESULTS_ROOT"

declare -a JOB_MODELS=()
declare -a JOB_GRAPHS=()
declare -a JOB_FOLDS=()
for model in "${MODELS[@]}"; do
  for graph_type in "${GRAPH_TYPES[@]}"; do
    for fold in "${FOLDS[@]}"; do
      JOB_MODELS+=("$model")
      JOB_GRAPHS+=("$graph_type")
      JOB_FOLDS+=("$fold")
    done
  done
done

TOTAL_JOBS="${#JOB_MODELS[@]}"
echo "[$(ts)] total_jobs=$TOTAL_JOBS"

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LABELS=()
declare -a ACTIVE_GPUS=()

NEXT_JOB=0
FAIL_RC=0
FAIL_LABEL=""

start_job() {
  local job_idx="$1"
  local gpu="$2"
  local model="${JOB_MODELS[$job_idx]}"
  local graph_type="${JOB_GRAPHS[$job_idx]}"
  local fold="${JOB_FOLDS[$job_idx]}"
  local label="${model}_${graph_type}_fold${fold}"
  local out_dir="${RESULTS_ROOT}/${label}"
  mkdir -p "$out_dir"
  echo "[$(ts)] start job=$label gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python run_tensor_kfold.py \
      --backend pyg \
      --mode full \
      --models "$model" \
      --graph-types "$graph_type" \
      --folds-to-run "$fold" \
      --output-dir "$out_dir" \
      --device cuda \
      "${TRAIN_ARGS[@]}" \
      "${EXTRA_ARGS[@]}" \
      > "${out_dir}/train.log" 2>&1 &

  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("$label")
  ACTIVE_GPUS+=("$gpu")
}

INITIAL_WORKERS="$GPU_COUNT"
if [[ "$TOTAL_JOBS" -lt "$INITIAL_WORKERS" ]]; then
  INITIAL_WORKERS="$TOTAL_JOBS"
fi

for ((slot=0; slot<INITIAL_WORKERS; slot++)); do
  start_job "$NEXT_JOB" "$slot"
  NEXT_JOB=$((NEXT_JOB + 1))
done

while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
  UPDATED_PIDS=()
  UPDATED_LABELS=()
  UPDATED_GPUS=()

  for i in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$i]}"
    label="${ACTIVE_LABELS[$i]}"
    gpu="${ACTIVE_GPUS[$i]}"

    if kill -0 "$pid" 2>/dev/null; then
      UPDATED_PIDS+=("$pid")
      UPDATED_LABELS+=("$label")
      UPDATED_GPUS+=("$gpu")
      continue
    fi

    rc=0
    wait "$pid" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
      echo "[$(ts)] job failed label=$label gpu=$gpu rc=$rc"
      FAIL_RC="$rc"
      FAIL_LABEL="$label"
    else
      echo "[$(ts)] job complete label=$label gpu=$gpu"
    fi

    if [[ "$FAIL_RC" -eq 0 && "$NEXT_JOB" -lt "$TOTAL_JOBS" ]]; then
      start_job "$NEXT_JOB" "$gpu"
      NEXT_JOB=$((NEXT_JOB + 1))
    fi
  done

  ACTIVE_PIDS=("${UPDATED_PIDS[@]}")
  ACTIVE_LABELS=("${UPDATED_LABELS[@]}")
  ACTIVE_GPUS=("${UPDATED_GPUS[@]}")
  if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
    sleep 5
  fi
done

if [[ "$FAIL_RC" -ne 0 ]]; then
  echo "[$(ts)] stopping without aggregation due to failed job: $FAIL_LABEL"
  exit "$FAIL_RC"
fi

python aggregate_vast_results.py \
  --results-root "$RESULTS_ROOT" \
  --output-dir "$MERGED_DIR"

if [[ -n "$RCLONE_REMOTE" ]]; then
  echo "[$(ts)] uploading merged results -> ${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/merged"
  rclone copy "$MERGED_DIR" "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/merged" \
    --create-empty-src-dirs \
    --transfers 8 \
    --checkers 16 \
    --retries 3 \
    --low-level-retries 10 \
    --stats 10s
fi

echo "[$(ts)] all jobs complete"
