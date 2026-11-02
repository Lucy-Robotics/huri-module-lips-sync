import asyncio
import os
from typing import AsyncGenerator, List, Optional

import numpy as np
from huri_core.core.module import ModuleWithHandle
from huri_events.audio import Audio
from ray import serve

from huri_events_lips import FaceWeights
from huri_lips.config import FPS, SAMPLE_RATE
from huri_lips.phonemes import arkit_from_visemes, viseme_frames
from huri_lips.stream import LipStream, Smoother, to_16k

_BACKEND = os.environ.get("HURI_LIPS_BACKEND", "auto")
_DEVICE = os.environ.get("HURI_LIPS_DEVICE") or None
_HALF = os.environ.get("HURI_LIPS_HALF", "0") == "1"
_THREADS = int(os.environ.get("HURI_LIPS_THREADS", "0"))
_IDENTITY = int(os.environ.get("HURI_LIPS_IDENTITY", "0"))

_DELAY_S = 0.13

_GAP_S = 0.02
_RESTART_S = 0.05


@serve.deployment(name="LipsHandle")
class LipsHandle:
    def __init__(
        self,
        backend: str = _BACKEND,
        device: Optional[str] = _DEVICE,
        identity: int = _IDENTITY,
        threads: int = _THREADS,
        half: bool = _HALF,
    ):
        from huri_lips.engine import load_engine

        self.engine = load_engine(
            backend, device=device, identity=identity, threads=threads, half=half
        )
        for batch in (1, 2, 4):
            self.engine.warmup(batch)
        print(f"[LIPS] ready on {self.engine.name}")

    async def infer(self, windows: np.ndarray) -> np.ndarray:
        return await asyncio.to_thread(self.engine, windows)


class Lips(ModuleWithHandle):
    _handle_cls = LipsHandle
    input_type = "huri.audio.out"
    output_type = "lips.face"
    input_event = Audio
    output_event = FaceWeights

    def __init__(
        self,
        _handle,
        hop_frames: int = 15,
        lookahead_frames: int = 6,
        max_batch: int = 4,
        silence_rms: float = 0.001,
        smooth_frames: int = 0,
        symmetric: bool = True,
        delay_s: float = _DELAY_S,
    ):
        super().__init__(_handle)
        self._stream = LipStream(
            hop=int(hop_frames),
            lookahead=int(lookahead_frames),
            smoother=Smoother(
                silence_rms=float(silence_rms),
                savgol=int(smooth_frames),
                symmetric=bool(symmetric),
            ),
        )
        self._max_batch = max(1, int(max_batch))
        self._delay_s = float(delay_s)
        self._t0: Optional[float] = None
        # Chunks arrive as concurrent process() calls on this instance.
        self._lock = asyncio.Lock()

    async def process(self, audio: Audio) -> AsyncGenerator[FaceWeights, None]:
        phonemes = getattr(audio, "phonemes", None)
        if phonemes is not None:
            if audio.data.size:
                yield self._from_phonemes(audio, phonemes)
            return
        async with self._lock:
            if audio.data.size:
                if self._t0 is not None:
                    at = audio.pts - (self._t0 + self._stream.seconds)
                    if at < -_RESTART_S:
                        async for face in self._run(final=True):
                            yield face
                    elif at > _GAP_S:
                        self._stream.add(np.zeros(int(at * SAMPLE_RATE), np.float32))
                if self._t0 is None:
                    self._t0 = float(audio.pts)
                self._stream.add(to_16k(audio.data, audio.sample_rate))
            if self._t0 is None:
                return
            async for face in self._run(final=audio.end):
                yield face

    def _from_phonemes(self, audio: Audio, phonemes) -> FaceWeights:
        visemes = viseme_frames(phonemes, audio.data.size / audio.sample_rate, FPS)
        weights = arkit_from_visemes(visemes)
        return FaceWeights(weights=weights, fps=FPS, pts=audio.pts, visemes=visemes)

    async def _run(self, final: bool) -> AsyncGenerator[FaceWeights, None]:
        starts: List[int] = self._stream.plan(final)
        for i in range(0, len(starts), self._max_batch):
            batch = starts[i : i + self._max_batch]
            windows = self._stream.windows(batch)
            weights = await self._handle.infer.remote(windows)
            emitted = self._stream.emit(batch, weights)
            first = emitted[0][0]
            frames = np.concatenate([w for _, w in emitted])
            pts = self._t0 + first / FPS + self._delay_s
            yield FaceWeights(weights=frames, fps=FPS, pts=pts)
        if final:
            self._stream.reset()
            self._t0 = None
