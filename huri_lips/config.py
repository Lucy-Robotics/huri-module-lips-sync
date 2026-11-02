"""What the lips module's parts share, without importing torch."""

import os
from pathlib import Path

SAMPLE_RATE = 16000  # the model's audio
FPS = 30  # its frames
WINDOW_FRAMES = 64  # frames of one streaming window (2.13 s)
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_FRAMES // FPS  # 34133
IDENTITIES = 12  # speaking styles the model was trained with
SHAPES = 52  # ARKit blend shapes

# LAM Audio2Expression's streaming weights (Apache-2.0).
HF_REPO = "3DAIGC/LAM_audio2exp"
HF_FILE = "LAM_audio2exp_streaming.tar"
# The repository commit (2025-04-17) the module was checked against.
HF_REVISION = os.environ.get(
    "HURI_LIPS_HF_REVISION", "0fe5f4dbb283ec7d9c01688681e6e4b6ac314858"
)
CHECKPOINT = "lam_a2e_streaming.pt"
ONNX_FILE = "lam_a2e_streaming.onnx"  # float32
ONNX_INT8_FILE = "lam_a2e_streaming.int8.onnx"  # int8 weights: 145 MB, faster


def models_dir() -> Path:
    """Where weights and exports are kept: $HURI_LIPS_HOME, else ~/.cache/huri-lips."""
    home = os.environ.get("HURI_LIPS_HOME")
    return Path(home).expanduser() if home else Path.home() / ".cache" / "huri-lips"
