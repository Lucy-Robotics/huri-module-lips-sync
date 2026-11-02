#!/usr/bin/env bash
# Installs this module with the dependency variant that matches the machine:
# requirements-nvidia.txt on NVIDIA, requirements-amd.txt on ROCm, else
# requirements-cpu.txt; then its events (events/) and the module itself; then
# the model: LAM's weights (373 MB download, once) and, on a CPU, their int8
# ONNX export (145 MB), in $HURI_LIPS_HOME (default ~/.cache/huri-lips).
#
#   ./install.sh                     # probe the hardware
#   ./install.sh --variant cpu       # or choose: cpu, nvidia or amd
#   ./install.sh --no-models         # packages only (CI, offline machines)
#
# Installs with $PIP if set, else `python -m pip`, else `uv pip` (a virtualenv
# made by uv has no pip).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() { echo "usage: install.sh [--variant cpu|nvidia|amd] [--no-models]" >&2; }

variant=""
models=1
while (($#)); do
  case "$1" in
    --variant)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      variant="$2"
      shift 2
      ;;
    --variant=*)
      variant="${1#--variant=}"
      shift
      ;;
    --no-models)
      models=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done
if [[ -z "$variant" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    variant=nvidia
  elif [[ -e /dev/kfd ]] && command -v rocminfo >/dev/null 2>&1; then
    variant=amd
  else
    variant=cpu
  fi
fi
case "$variant" in
  cpu | nvidia | amd) ;;
  *)
    echo "unknown variant: $variant (cpu|nvidia|amd)" >&2
    exit 2
    ;;
esac
pins="requirements-$variant.txt"
[[ -f "$pins" ]] || { echo "missing $pins" >&2; exit 1; }
echo "installing for: $variant"

if [[ -n "${PIP:-}" ]]; then
  pip="$PIP"
elif python -m pip --version >/dev/null 2>&1; then
  pip="python -m pip"
elif command -v uv >/dev/null 2>&1; then
  pip="uv pip"
else
  echo "no pip: run in a virtualenv with pip, or install uv, or set PIP" >&2
  exit 1
fi
# A pin file with nothing but comments has nothing to install.
if grep -qvE '^\s*(#|$)' "$pins"; then
  $pip install -r "$pins"
fi
# The module and its events in one call: pip finds huri-events-lips only here.
$pip install -e ./events -e ".[dev]"

if ((models)); then
  python="${PYTHON:-python}"
  "$python" -m huri_lips fetch
  if [[ "$variant" == cpu ]]; then
    "$python" -m huri_lips export-onnx --precision int8
  fi
fi
