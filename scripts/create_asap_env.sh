#!/usr/bin/env bash
# Create the ASAP conda environment used for OpenPCDet-based KITTI experiments.
#
# This script installs Python, PyTorch CUDA wheels, OpenPCDet Python
# dependencies, and spconv. It does not clone OpenPCDet and does not install
# pcdet itself; install pcdet from third_party/OpenPCDet after manually cloning
# OpenPCDet.
#
# Environment variables:
#   ENV_NAME          default: asap
#   PYTHON_VERSION    default: 3.9
#
# Usage:
#   bash scripts/create_asap_env.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-asap}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"

info() {
  echo "[ASAP] $*"
}

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  info "Conda environment already exists: ${ENV_NAME}"
else
  info "Creating conda environment: ${ENV_NAME}"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
fi

info "Installing build helpers"
conda run -n "${ENV_NAME}" pip install --upgrade pip setuptools wheel

info "Installing PyTorch CUDA 12.4 wheels"
conda run -n "${ENV_NAME}" pip install \
  torch==2.5.1+cu124 \
  torchvision==0.20.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

info "Installing OpenPCDet Python dependencies and spconv"
conda run -n "${ENV_NAME}" pip install \
  'numpy<2' \
  llvmlite \
  numba \
  tensorboardX \
  easydict \
  pyyaml \
  scikit-image \
  tqdm \
  SharedArray \
  opencv-python \
  pyquaternion \
  spconv-cu120

info "Verifying environment"
conda run -n "${ENV_NAME}" python scripts/verify_openpcdet_env.py

info "Environment ${ENV_NAME} is ready for manually cloned OpenPCDet."
