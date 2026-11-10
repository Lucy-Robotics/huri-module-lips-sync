import os
from pathlib import Path

SAMPLE_RATE = 16000
FPS = 30
WINDOW_FRAMES = 64
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_FRAMES // FPS
IDENTITIES = 12
SHAPES = 52

HF_REPO = "3DAIGC/LAM_audio2exp"
HF_FILE = "LAM_audio2exp_streaming.tar"
HF_REVISION = os.environ.get(
    "HURI_LIPS_HF_REVISION", "0fe5f4dbb283ec7d9c01688681e6e4b6ac314858"
)
CHECKPOINT = "lam_a2e_streaming.pt"
ONNX_FILE = "lam_a2e_streaming.onnx"
ONNX_INT8_FILE = "lam_a2e_streaming.int8.onnx"


def models_dir() -> Path:
    home = os.environ.get("HURI_LIPS_HOME")
    return Path(home).expanduser() if home else Path.home() / ".cache" / "huri-lips"
