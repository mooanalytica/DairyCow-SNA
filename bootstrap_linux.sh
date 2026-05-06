#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_DIR="${PROJECT_ROOT}/venv"
TORCH_VARIANT="${TORCH_VARIANT:-auto}"   # auto | cpu | cu118

echo "[info] project root: ${PROJECT_ROOT}"
echo "[info] requested torch variant: ${TORCH_VARIANT}"

choose_torch_variant() {
  local requested="$1"
  if [[ "${requested}" != "auto" ]]; then
    echo "${requested}"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "cu118"
  else
    echo "cpu"
  fi
}

run_privileged() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "[warn] Need elevated privileges to run: $*"
    return 1
  fi
}

SYSTEM_PACKAGES=(ffmpeg libgl1 libglib2.0-0 build-essential python3-dev python3-venv python3-pip git)

if command -v apt-get >/dev/null 2>&1; then
  echo "[info] Using apt-get to install Linux system packages"
  run_privileged apt-get update
  run_privileged apt-get install -y "${SYSTEM_PACKAGES[@]}"
else
  echo "[warn] apt-get not found. Install these packages manually before continuing:"
  printf '  - %s\n' "${SYSTEM_PACKAGES[@]}"
fi

RESOLVED_TORCH_VARIANT="$(choose_torch_variant "${TORCH_VARIANT}")"
echo "[info] resolved torch variant: ${RESOLVED_TORCH_VARIANT}"

case "${RESOLVED_TORCH_VARIANT}" in
  cu118)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
    ;;
  cpu)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    ;;
  *)
    echo "[error] Unsupported TORCH_VARIANT: ${RESOLVED_TORCH_VARIANT}"
    exit 1
    ;;
esac

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL}" torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
python -m pip install cython_bbox

python - <<PY
from pathlib import Path
import sys

project_root = Path(r"${PROJECT_ROOT}")
sys.path.insert(0, str(project_root / "vendor" / "ByteTrack"))
sys.path.insert(0, str(project_root / "vendor" / "ZebraPoseViTPose"))

import cv2
import torch
from yolox.tracker.byte_tracker import BYTETracker
from mmpose.apis import init_pose_model

required = [
    project_root / "run_full_interaction_pipeline.py",
    project_root / "models" / "Object_Detection_Trained_Model.pt",
    project_root / "models" / "Identification_Model_Trained.pt",
    project_root / "models" / "Keypoint_Model_Trained.pth",
    project_root / "models" / "stage1_tcn_best.pt",
    project_root / "models" / "stage2_inception_best.pt",
    project_root / "models" / "svm_model.joblib",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing bundled files: {missing}")

print("[ok] torch =", torch.__version__)
print("[ok] torch_cuda_available =", torch.cuda.is_available())
print("[ok] opencv =", cv2.__version__)
print("[ok] BYTETracker import = ok")
print("[ok] mmpose import = ok")
print("[ok] project files present")
PY

echo "[ok] linux bootstrap complete"
echo "[next] source \"${VENV_DIR}/bin/activate\""
echo "[next] python \"${PROJECT_ROOT}/run_full_interaction_pipeline.py\" --input-dir \"${PROJECT_ROOT}/inputs\""
