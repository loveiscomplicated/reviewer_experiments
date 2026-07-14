#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/reviewer_experiments}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-pyg_2}"
CODE_ARCHIVE_FILE_ID="${CODE_ARCHIVE_FILE_ID:-}"
TEDS_GDRIVE_FILE_ID="${TEDS_GDRIVE_FILE_ID:-}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

download_drive_file() {
  local file_id="$1"
  local output_path="$2"
  if [[ -z "$file_id" ]]; then
    echo "[$(ts)] missing Google Drive file id for $output_path"
    return 1
  fi
  if [[ -f "$output_path" ]]; then
    echo "[$(ts)] already exists: $output_path"
    return 0
  fi
  echo "[$(ts)] downloading Google Drive file -> $output_path"
  gdown "https://drive.google.com/uc?id=${file_id}" -O "$output_path"
}

echo "[$(ts)] ===== setup_vast_teds start ====="
apt update
apt install -y git wget tmux rclone python3-pip

mkdir -p "$WORKSPACE_ROOT"
cd "$WORKSPACE_ROOT"

if [[ ! -d "$REPO_DIR" && -n "$CODE_ARCHIVE_FILE_ID" ]]; then
  mkdir -p "$REPO_DIR"
  ARCHIVE_PATH="${WORKSPACE_ROOT}/reviewer_experiments.tgz"
  EXTRACT_DIR="${WORKSPACE_ROOT}/reviewer_experiments_extract"
  python3 -m pip install -U gdown
  download_drive_file "$CODE_ARCHIVE_FILE_ID" "$ARCHIVE_PATH"
  rm -rf "$EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR"
  tar -xzf "$ARCHIVE_PATH" -C "$EXTRACT_DIR"
  if [[ -d "${EXTRACT_DIR}/reviewer_experiments" ]]; then
    cp -a "${EXTRACT_DIR}/reviewer_experiments/." "$REPO_DIR/"
  else
    cp -a "${EXTRACT_DIR}/." "$REPO_DIR/"
  fi
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "[$(ts)] ERROR: repo dir not found: $REPO_DIR"
  echo "[$(ts)] Provide CODE_ARCHIVE_FILE_ID or pre-place code at REPO_DIR."
  exit 1
fi

if [[ ! -d "$CONDA_DIR" ]]; then
  echo "[$(ts)] installing miniconda"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR"
fi

source "$CONDA_SH"
conda activate base || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[$(ts)] creating conda env $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.12 pip
fi

conda activate "$ENV_NAME"
python -m pip install -U pip

CUDA_RAW=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/' || echo "")
if [[ -z "$CUDA_RAW" ]]; then
  CUDA_RAW=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/' || echo "")
fi

case "$CUDA_RAW" in
  128|129) CUDA_TAG="cu128" ;;
  126|127) CUDA_TAG="cu126" ;;
  121|122|123) CUDA_TAG="cu121" ;;
  *) CUDA_TAG="cu124" ;;
esac

echo "[$(ts)] installing PyTorch for ${CUDA_TAG} (raw CUDA: ${CUDA_RAW:-unknown})"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install torch-geometric
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install torch-scatter torch-sparse torch-cluster \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"

pip install numpy pandas scipy scikit-learn tqdm gdown

cd "$REPO_DIR"
if [[ -n "$TEDS_GDRIVE_FILE_ID" ]]; then
  download_drive_file "$TEDS_GDRIVE_FILE_ID" "${REPO_DIR}/TEDS_Discharge.csv"
elif [[ -f "${REPO_DIR}/TEDS_Discharge.csv" ]]; then
  echo "[$(ts)] TEDS_Discharge.csv already present"
else
  echo "[$(ts)] WARNING: TEDS_GDRIVE_FILE_ID not set and TEDS_Discharge.csv not present."
fi

python - <<'PY'
import importlib.util
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch_geometric", bool(importlib.util.find_spec("torch_geometric")))
PY

echo "[$(ts)] setup complete: $REPO_DIR"
