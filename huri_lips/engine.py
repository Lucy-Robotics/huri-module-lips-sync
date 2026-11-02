import os
from pathlib import Path
from typing import Optional

import numpy as np

from huri_lips.config import (
    IDENTITIES,
    ONNX_FILE,
    ONNX_INT8_FILE,
    WINDOW_FRAMES,
    WINDOW_SAMPLES,
    models_dir,
)


class Engine:
    name = "engine"

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def warmup(self, batch: int = 1) -> None:
        self(np.zeros((batch, WINDOW_SAMPLES), dtype=np.float32))


class TorchEngine(Engine):
    def __init__(
        self,
        device: Optional[str] = None,
        identity: int = 0,
        checkpoint: Optional[os.PathLike] = None,
        half: bool = False,
    ):
        import torch

        from huri_lips.lam import load_model, one_hot_identity

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(checkpoint, self.device)
        self.dtype = torch.float16 if half and self.device != "cpu" else torch.float32
        self.model.to(self.dtype)
        self.identity = one_hot_identity(identity).to(self.device, self.dtype)
        self.name = f"torch:{self.device}"

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
        x = x.to(self.device, self.dtype)
        with torch.inference_mode():
            out = self.model(x, self.identity.expand(x.shape[0], -1), WINDOW_FRAMES)
        return out.float().cpu().numpy()


class OnnxEngine(Engine):
    def __init__(
        self,
        path: os.PathLike,
        identity: int = 0,
        threads: int = 0,
    ):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.identity = np.zeros((1, IDENTITIES), dtype=np.float32)
        self.identity[0, identity] = 1.0
        self.name = "onnx:cpu"

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(windows, dtype=np.float32)
        ident = np.repeat(self.identity, x.shape[0], axis=0)
        (out,) = self.session.run(None, {"audio": x, "identity": ident})
        return out


def onnx_path() -> Path:
    int8 = models_dir() / ONNX_INT8_FILE
    return int8 if int8.exists() else models_dir() / ONNX_FILE


def _gpu() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def load_engine(
    backend: str = "auto",
    device: Optional[str] = None,
    identity: int = 0,
    onnx: Optional[os.PathLike] = None,
    threads: int = 0,
    half: bool = False,
) -> Engine:
    onnx = Path(onnx) if onnx else onnx_path()
    if backend == "auto":
        if device not in (None, "cpu") or (device is None and _gpu()):
            backend = "torch"
        elif onnx.exists():
            backend = "onnx"
        else:
            backend = "torch"
    if backend == "torch":
        return TorchEngine(device, identity, half=half)
    if backend == "onnx":
        if not onnx.exists():
            raise FileNotFoundError(
                f"{onnx}: no ONNX export yet, run `python -m huri_lips export-onnx`"
            )
        return OnnxEngine(onnx, identity, threads)
    raise ValueError(f"unknown backend {backend!r} (auto, torch, onnx)")
