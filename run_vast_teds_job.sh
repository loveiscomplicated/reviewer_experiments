#!/usr/bin/env bash
set -euo pipefail

MODEL_ORDER=(${MODEL_ORDER:-gcn gin gat})
GRAPH_ORDER=(${GRAPH_ORDER:-statistical fully_connected})
FOLD_ORDER=(${FOLD_ORDER:-1 2 3 4 5})

usage() {
  echo "Usage:"
  echo "  bash run_vast_teds_job.sh <model> <graph_type> <fold> [extra_args...]"
  echo "  JOB_INDEX=<0..29> bash run_vast_teds_job.sh [extra_args...]"
  echo "Example:"
  echo "  bash run_vast_teds_job.sh gcn statistical 1"
  echo "  JOB_INDEX=0 bash run_vast_teds_job.sh"
}

if [[ $# -ge 3 ]]; then
  MODEL="$1"
  GRAPH_TYPE="$2"
  FOLD="$3"
  shift 3
elif [[ -n "${JOB_INDEX:-}" ]]; then
  if ! [[ "$JOB_INDEX" =~ ^[0-9]+$ ]]; then
    echo "JOB_INDEX must be an integer, got: $JOB_INDEX"
    usage
    exit 1
  fi
  job_index=$((10#$JOB_INDEX))
  jobs_per_model=$((${#GRAPH_ORDER[@]} * ${#FOLD_ORDER[@]}))
  total_jobs=$((${#MODEL_ORDER[@]} * jobs_per_model))
  if (( job_index < 0 || job_index >= total_jobs )); then
    echo "JOB_INDEX out of range: $JOB_INDEX (valid: 0..$((total_jobs - 1)))"
    usage
    exit 1
  fi
  model_idx=$((job_index / jobs_per_model))
  rem=$((job_index % jobs_per_model))
  graph_idx=$((rem / ${#FOLD_ORDER[@]}))
  fold_idx=$((rem % ${#FOLD_ORDER[@]}))
  MODEL="${MODEL_ORDER[$model_idx]}"
  GRAPH_TYPE="${GRAPH_ORDER[$graph_idx]}"
  FOLD="${FOLD_ORDER[$fold_idx]}"
else
  usage
  exit 1
fi
EXTRA_ARGS=("$@")
JOB_LABEL="${MODEL}_${GRAPH_TYPE}_fold${FOLD}"
TRAIN_ARGS=()
if [[ "${NO_PROGRESS:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--no-progress)
fi
PRELOAD_DEVICE="${PRELOAD_DEVICE:-cuda}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/reviewer_experiments}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-pyg_2}"
RESULTS_ROOT="${RESULTS_ROOT:-results_vast}"
RCLONE_REMOTE="${RCLONE_REMOTE:-}"
RCLONE_DEST_DIR="${RCLONE_DEST_DIR:-TEDS_GNN_reviewer_results}"
RCLONE_B64_FILE="${RCLONE_B64_FILE:-/tmp/rclone_conf.b64}"
DISCORD_BOT_NAME="${DISCORD_BOT_NAME:-TEDS GNN Bot}"

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

echo "[$(ts)] ===== run_vast_teds_job start ====="
echo "[$(ts)] model=$MODEL graph_type=$GRAPH_TYPE fold=$FOLD"
notify "[START] TEDS reviewer job started. job=${JOB_LABEL} host=$(hostname 2>/dev/null || echo unknown)"

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
  printf "%s" "$RCLONE_CONF_B64" > "$RCLONE_B64_FILE"
  base64 -d "$RCLONE_B64_FILE" > /root/.config/rclone/rclone.conf
fi

OUT_DIR="${RESULTS_ROOT}/${JOB_LABEL}"
mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

set +e
python run_tensor_kfold.py \
  --backend pyg \
  --mode full \
  --models "$MODEL" \
  --graph-types "$GRAPH_TYPE" \
  --folds-to-run "$FOLD" \
  --output-dir "$OUT_DIR" \
  --device cuda \
  --preload-device "$PRELOAD_DEVICE" \
  "${TRAIN_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${OUT_DIR}/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e

if [[ "$TRAIN_RC" -ne 0 ]]; then
  notify "[FAIL] TEDS reviewer job failed. job=${JOB_LABEL} rc=${TRAIN_RC} out_dir=${OUT_DIR}"
  exit "$TRAIN_RC"
fi

if [[ -n "$RCLONE_REMOTE" ]]; then
  echo "[$(ts)] uploading $OUT_DIR -> ${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/${MODEL}_${GRAPH_TYPE}_fold${FOLD}"
  if rclone copy "$OUT_DIR" "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/${MODEL}_${GRAPH_TYPE}_fold${FOLD}" \
    --create-empty-src-dirs \
    --transfers 8 \
    --checkers 16 \
    --retries 3 \
    --low-level-retries 10 \
    --stats 10s
  then
    notify "[UPLOAD_OK] TEDS reviewer job uploaded. job=${JOB_LABEL} remote=${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/${JOB_LABEL}"
  else
    UPLOAD_RC=$?
    notify "[UPLOAD_FAIL] TEDS reviewer job upload failed. job=${JOB_LABEL} rc=${UPLOAD_RC} out_dir=${OUT_DIR}"
    exit "$UPLOAD_RC"
  fi
fi

echo "[$(ts)] job complete: $OUT_DIR"
notify "[SUCCESS] TEDS reviewer job completed. job=${JOB_LABEL} out_dir=${OUT_DIR}"
