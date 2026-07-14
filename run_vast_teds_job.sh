#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash run_vast_teds_job.sh <model> <graph_type> <fold> [extra_args...]"
  echo "Example: bash run_vast_teds_job.sh gcn statistical 1"
  exit 1
fi

MODEL="$1"
GRAPH_TYPE="$2"
FOLD="$3"
shift 3
EXTRA_ARGS=("$@")

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/reviewer_experiments}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-pyg_2}"
RESULTS_ROOT="${RESULTS_ROOT:-results_vast}"
RCLONE_REMOTE="${RCLONE_REMOTE:-}"
RCLONE_DEST_DIR="${RCLONE_DEST_DIR:-TEDS_GNN_reviewer_results}"
RCLONE_B64_FILE="${RCLONE_B64_FILE:-/tmp/rclone_conf.b64}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[$(ts)] ===== run_vast_teds_job start ====="
echo "[$(ts)] model=$MODEL graph_type=$GRAPH_TYPE fold=$FOLD"

if [[ -f "${SCRIPT_DIR}/setup_vast_teds.sh" ]]; then
  bash "${SCRIPT_DIR}/setup_vast_teds.sh"
else
  bash "${REPO_DIR}/setup_vast_teds.sh"
fi

source "$CONDA_SH"
conda activate "$ENV_NAME"

if [[ -n "${RCLONE_CONF_B64:-}" ]]; then
  mkdir -p /root/.config/rclone
  printf "%s" "$RCLONE_CONF_B64" > "$RCLONE_B64_FILE"
  base64 -d "$RCLONE_B64_FILE" > /root/.config/rclone/rclone.conf
fi

OUT_DIR="${RESULTS_ROOT}/${MODEL}_${GRAPH_TYPE}_fold${FOLD}"
mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

python run_tensor_kfold.py \
  --backend pyg \
  --mode full \
  --models "$MODEL" \
  --graph-types "$GRAPH_TYPE" \
  --folds-to-run "$FOLD" \
  --output-dir "$OUT_DIR" \
  --device cuda \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${OUT_DIR}/train.log"

if [[ -n "$RCLONE_REMOTE" ]]; then
  echo "[$(ts)] uploading $OUT_DIR -> ${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/${MODEL}_${GRAPH_TYPE}_fold${FOLD}"
  rclone copy "$OUT_DIR" "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/${MODEL}_${GRAPH_TYPE}_fold${FOLD}" \
    --create-empty-src-dirs \
    --transfers 8 \
    --checkers 16 \
    --retries 3 \
    --low-level-retries 10 \
    --stats 10s
fi

echo "[$(ts)] job complete: $OUT_DIR"
