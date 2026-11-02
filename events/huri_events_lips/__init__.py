import struct
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from huri_core.core.events import EventData

# Apple ARKit's 52 blend shapes, in this order on the wire (alphabetical, as
# ARKit lists them and as the LAM model outputs them).
ARKIT_NAMES = (
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
)
ARKIT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ARKIT_NAMES)}
N_SHAPES = len(ARKIT_NAMES)
VISEME_NAMES = (
    "sil",
    "PP",
    "FF",
    "TH",
    "DD",
    "kk",
    "CH",
    "SS",
    "nn",
    "RR",
    "aa",
    "E",
    "I",
    "O",
    "U",
)
VISEME_INDEX: Dict[str, int] = {name: i for i, name in enumerate(VISEME_NAMES)}
N_VISEMES = len(VISEME_NAMES)

_HEADER = struct.Struct(">dII")


@dataclass
class FaceWeights(EventData[bytes]):
    """ARKit blend shape weights (0..1), `fps` frames per second.

    Wire (binary): ``>dII`` header — pts (f64, seconds on the utterance's
    timeline, like Audio.pts), fps (u32), frame count n (u32) — then n x 52
    float32 weights, frame after frame, each frame in ARKIT_NAMES order; then,
    optionally, n x 15 viseme weights in VISEME_NAMES order.
    """

    weights: np.ndarray
    fps: int = 30
    pts: float = 0.0
    visemes: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float32).reshape(-1, N_SHAPES)
        if self.visemes is not None:
            self.visemes = np.asarray(self.visemes, dtype=np.float32).reshape(
                -1, N_VISEMES
            )
            if self.visemes.shape[0] != self.weights.shape[0]:
                raise ValueError("visemes and weights need as many frames")

    @property
    def frames(self) -> int:
        return int(self.weights.shape[0])

    def shape(self, name: str) -> np.ndarray:
        return self.weights[:, ARKIT_INDEX[name]]

    def to_wire(self) -> bytes:
        header = _HEADER.pack(float(self.pts), int(self.fps), self.frames)
        body = self.weights.astype("<f4", copy=False).tobytes()
        if self.visemes is not None:
            body += self.visemes.astype("<f4", copy=False).tobytes()
        return header + body

    @classmethod
    def from_wire(cls, data: bytes) -> "FaceWeights":
        if len(data) < _HEADER.size:
            raise ValueError(f"truncated lips.face header: {len(data)} bytes")
        pts, fps, n = _HEADER.unpack_from(data)
        body = np.frombuffer(data, dtype="<f4", offset=_HEADER.size)
        if body.size not in (n * N_SHAPES, n * (N_SHAPES + N_VISEMES)):
            raise ValueError(
                f"lips.face: {body.size} floats for {n} frames of {N_SHAPES} shapes"
            )
        weights = body[: n * N_SHAPES].reshape(n, N_SHAPES).copy()
        visemes = None
        if body.size > n * N_SHAPES:
            visemes = body[n * N_SHAPES :].reshape(n, N_VISEMES).copy()
        return cls(weights=weights, fps=fps, pts=pts, visemes=visemes)

    def summarize(self) -> str:
        return f"FaceWeights(frames={self.frames}, fps={self.fps}, pts={self.pts:.3f}s)"
