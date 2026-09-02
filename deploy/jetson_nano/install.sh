#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This installer must run on a Jetson Nano (aarch64)." >&2
  exit 2
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: ./install.sh /path/to/NVIDIA_PYTORCH_WHEEL.whl" >&2
  exit 2
fi

torch_wheel="$1"
if [[ ! -f "$torch_wheel" ]]; then
  echo "PyTorch wheel not found: $torch_wheel" >&2
  exit 2
fi

sudo apt-get update
sudo apt-get install -y \
  libopenblas-dev \
  libopenmpi-dev \
  python3-opencv \
  python3-pip

python3 -m pip install --user --upgrade "pip==21.3.1"
python3 -m pip install --user "numpy==1.19.4" "$torch_wheel"

python3 - <<'PY'
import cv2
import torch

print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled NVIDIA PyTorch wheel is required")
PY
