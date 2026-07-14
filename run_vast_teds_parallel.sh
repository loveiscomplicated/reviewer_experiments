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
DISCORD_BOT_NAME="${DISCORD_BOT_NAME:-TEDS GNN Bot}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-0}"

MODELS=(${MODELS:-gcn gin gat})
GRAPH_TYPES=(${GRAPH_TYPES:-statistical fully_connected})
FOLDS=(${FOLDS:-1 2 3 4 5})
EXTRA_ARGS=("$@")
TRAIN_ARGS=()
if [[ "${NO_PROGRESS:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--no-progress)
fi
PRELOAD_DEVICE="${PRELOAD_DEVICE:-cuda}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTIFY_PY="${NOTIFY_PY:-${SCRIPT_DIR}/discord_notify.py}"

notify() {
  local msg="$1"
  if [[ "${DISCORD_NOTIFY:-1}" == "0" ]]; then
    return 0
  fi
  if [[ ! -f "$NOTIFY_PY" ]]; then
    echo "[$(ts)] Discord notify skipped: helper not found at $NOTIFY_PY"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 "$NOTIFY_PY" "$msg" "$DISCORD_BOT_NAME" || true
  elif command -v python >/dev/null 2>&1; then
    python "$NOTIFY_PY" "$msg" "$DISCORD_BOT_NAME" || true
  else
    echo "[$(ts)] Discord notify skipped: python not found"
  fi
}

echo "[$(ts)] ===== run_vast_teds_parallel start ====="
notify "[START] TEDS reviewer parallel run started. host=$(hostname 2>/dev/null || echo unknown)"
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
  notify "[FAIL] TEDS reviewer parallel run failed: no CUDA GPU detected."
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

MATRIX_TOTAL_JOBS="${#JOB_MODELS[@]}"
RANGE_MODE=0
JOB_INDEX_SPEC="all"
JOB_INDEX_LABEL="all"

declare -a SELECTED_JOB_INDICES=()

append_selected_job_index() {
  local raw_idx="$1"
  local idx=$((10#$raw_idx))
  if (( idx < 0 || idx >= MATRIX_TOTAL_JOBS )); then
    echo "[$(ts)] ERROR: job index out of range: $raw_idx (valid: 0..$((MATRIX_TOTAL_JOBS - 1)))"
    exit 1
  fi
  local existing
  for existing in "${SELECTED_JOB_INDICES[@]}"; do
    if [[ "$existing" == "$idx" ]]; then
      return 0
    fi
  done
  SELECTED_JOB_INDICES+=("$idx")
}

parse_job_index_spec() {
  local spec="$1"
  local token start end idx
  spec="${spec//,/ }"
  for token in $spec; do
    if [[ "$token" =~ ^([0-9]+)(-|\.\.|:)([0-9]+)$ ]]; then
      start=$((10#${BASH_REMATCH[1]}))
      end=$((10#${BASH_REMATCH[3]}))
      if (( start > end )); then
        echo "[$(ts)] ERROR: descending job index range is not supported: $token"
        exit 1
      fi
      for ((idx=start; idx<=end; idx++)); do
        append_selected_job_index "$idx"
      done
    elif [[ "$token" =~ ^[0-9]+$ ]]; then
      append_selected_job_index "$token"
    else
      echo "[$(ts)] ERROR: invalid job index token: $token"
      echo "[$(ts)] Use JOB_INDEX_RANGE=0-9 or JOB_INDEXES='0 1 2 10-14'."
      exit 1
    fi
  done
}

if [[ -n "${JOB_INDEX_RANGE:-}" || -n "${JOB_INDEXES:-}" ]]; then
  RANGE_MODE=1
  JOB_INDEX_SPEC="${JOB_INDEX_RANGE:-${JOB_INDEXES:-}}"
  parse_job_index_spec "$JOB_INDEX_SPEC"
  if [[ "${#SELECTED_JOB_INDICES[@]}" -lt 1 ]]; then
    echo "[$(ts)] ERROR: no jobs selected by JOB_INDEX_RANGE/JOB_INDEXES."
    exit 1
  fi

  declare -a ALL_JOB_MODELS=("${JOB_MODELS[@]}")
  declare -a ALL_JOB_GRAPHS=("${JOB_GRAPHS[@]}")
  declare -a ALL_JOB_FOLDS=("${JOB_FOLDS[@]}")
  JOB_MODELS=()
  JOB_GRAPHS=()
  JOB_FOLDS=()
  for idx in "${SELECTED_JOB_INDICES[@]}"; do
    JOB_MODELS+=("${ALL_JOB_MODELS[$idx]}")
    JOB_GRAPHS+=("${ALL_JOB_GRAPHS[$idx]}")
    JOB_FOLDS+=("${ALL_JOB_FOLDS[$idx]}")
  done
  JOB_INDEX_LABEL="$(printf "%s" "$JOB_INDEX_SPEC" | tr -c 'A-Za-z0-9._-' '_')"
fi

TOTAL_JOBS="${#JOB_MODELS[@]}"
if [[ -z "${UPLOAD_RAW_RESULTS+x}" ]]; then
  if [[ "$RANGE_MODE" -eq 1 ]]; then
    UPLOAD_RAW_RESULTS=1
  else
    UPLOAD_RAW_RESULTS=0
  fi
fi

echo "[$(ts)] matrix_total_jobs=$MATRIX_TOTAL_JOBS selected_jobs=$TOTAL_JOBS job_index_spec=$JOB_INDEX_SPEC"
notify "[START] TEDS reviewer parallel matrix ready. matrix_total_jobs=${MATRIX_TOTAL_JOBS} selected_jobs=${TOTAL_JOBS} job_index_spec=${JOB_INDEX_SPEC} detected_gpus=${GPU_COUNT}"

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
      --preload-device "$PRELOAD_DEVICE" \
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
      notify "[FAIL] TEDS reviewer job failed. label=${label} gpu=${gpu} rc=${rc}"
      FAIL_RC="$rc"
      FAIL_LABEL="$label"
    else
      echo "[$(ts)] job complete label=$label gpu=$gpu"
      notify "[SUCCESS] TEDS reviewer job completed. label=${label} gpu=${gpu}"
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
  notify "[FAIL] TEDS reviewer parallel run stopping without aggregation. failed_job=${FAIL_LABEL} rc=${FAIL_RC}"
  exit "$FAIL_RC"
fi

if [[ "$UPLOAD_RESULTS" == "1" && -n "$RCLONE_REMOTE" && "$UPLOAD_RAW_RESULTS" == "1" ]]; then
  echo "[$(ts)] uploading raw selected results -> ${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/raw"
  if rclone copy "$RESULTS_ROOT" "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/raw" \
    --create-empty-src-dirs \
    --transfers 8 \
    --checkers 16 \
    --retries 3 \
    --low-level-retries 10 \
    --stats 10s
  then
    notify "[UPLOAD_OK] TEDS reviewer raw selected results uploaded. job_index_spec=${JOB_INDEX_SPEC} remote=${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/raw"
  else
    UPLOAD_RC=$?
    notify "[UPLOAD_FAIL] TEDS reviewer raw selected upload failed. job_index_spec=${JOB_INDEX_SPEC} rc=${UPLOAD_RC} results_root=${RESULTS_ROOT}"
    exit "$UPLOAD_RC"
  fi
elif [[ "$UPLOAD_RESULTS" == "1" && -z "$RCLONE_REMOTE" ]]; then
  echo "[$(ts)] raw upload requested but RCLONE_REMOTE is empty; skipping raw upload"
else
  echo "[$(ts)] raw upload disabled (set UPLOAD_RESULTS=1 to enable rclone upload)"
fi

set +e
python aggregate_vast_results.py \
  --results-root "$RESULTS_ROOT" \
  --output-dir "$MERGED_DIR"
AGG_RC=$?
set -e

if [[ "$AGG_RC" -ne 0 ]]; then
  notify "[FAIL] TEDS reviewer aggregation failed. rc=${AGG_RC} results_root=${RESULTS_ROOT}"
  exit "$AGG_RC"
fi

if [[ "$UPLOAD_RESULTS" == "1" && -n "$RCLONE_REMOTE" ]]; then
  MERGED_REMOTE_DIR="${RCLONE_DEST_DIR}/merged"
  if [[ "$RANGE_MODE" -eq 1 ]]; then
    MERGED_REMOTE_DIR="${RCLONE_DEST_DIR}/merged_ranges/${JOB_INDEX_LABEL}"
  fi
  echo "[$(ts)] uploading merged results -> ${RCLONE_REMOTE}:${MERGED_REMOTE_DIR}"
  if rclone copy "$MERGED_DIR" "${RCLONE_REMOTE}:${MERGED_REMOTE_DIR}" \
    --create-empty-src-dirs \
    --transfers 8 \
    --checkers 16 \
    --retries 3 \
    --low-level-retries 10 \
    --stats 10s
  then
    notify "[UPLOAD_OK] TEDS reviewer merged results uploaded. job_index_spec=${JOB_INDEX_SPEC} remote=${RCLONE_REMOTE}:${MERGED_REMOTE_DIR}"
  else
    UPLOAD_RC=$?
    notify "[UPLOAD_FAIL] TEDS reviewer merged upload failed. rc=${UPLOAD_RC} merged_dir=${MERGED_DIR}"
    exit "$UPLOAD_RC"
  fi
elif [[ "$UPLOAD_RESULTS" == "1" ]]; then
  echo "[$(ts)] merged upload requested but RCLONE_REMOTE is empty; skipping merged upload"
else
  echo "[$(ts)] merged upload disabled (set UPLOAD_RESULTS=1 to enable rclone upload)"
fi

echo "[$(ts)] all jobs complete"
notify "[SUCCESS] TEDS reviewer parallel run completed. total_jobs=${TOTAL_JOBS} merged_dir=${MERGED_DIR}"
