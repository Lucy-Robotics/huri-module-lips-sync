"""Lip sync: the voice's face, as ARKit blend shapes.

    huri.audio.out ──> lips ──> lips.face   (FaceWeights: 52 ARKit weights,
                                             30 fps, on the voice's timeline)

`lips` hears what HuRI says (the TTS, and huri-module-think's thinking
phrases) and computes, with LAM Audio2Expression, the mouth, jaw, cheeks and
brows that go with it. The frames carry the utterance's timeline (`pts`, like
Audio.pts), so a client plays them in step with the voice; they are computed
faster than the voice is heard, a sentence ahead with a TTS that sends whole
sentences (Piper).

The model runs in the `LipsHandle` deployment, on a GPU or a CPU (see
huri_lips.engine); its settings come from the environment, like the other
HuRI deployments:

    HURI_LIPS_BACKEND   auto (default), torch or onnx
    HURI_LIPS_DEVICE    torch device: cuda, cpu (default: cuda if any)
    HURI_LIPS_HALF      1: float16 on a GPU
    HURI_LIPS_THREADS   onnxruntime threads (default: all cores)
    HURI_LIPS_IDENTITY  speaking style, 0..11 (default 0)
    HURI_LIPS_HOME      weights and exports (default ~/.cache/huri-lips)
"""

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

# LAM's mouth leads the voice: measured against a Piper voice's phoneme
# timings (huri-module-lips-sync's eval), its frames match the sounds best
# 0.13 s later (correlation of the lip gap with the sounds: -0.01 at 0, 0.51
# at 0.13 s; "p", "b", "m" closed 29% -> 80%). Offline, whole sentences in one
# pass, it leads as much: it is the model's, not the streaming's.
_DELAY_S = 0.13

# A chunk this much later than the end of the audio so far leaves a gap,
# filled with silence; earlier, it starts a new utterance.
_GAP_S = 0.02
_RESTART_S = 0.05


@serve.deployment(name="LipsHandle")
class LipsHandle:
    """LAM Audio2Expression on batches of 2.13 s windows (see huri_lips.engine)."""

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
        # First calls pay one-time costs (kernels, allocations) per batch size.
        for batch in (1, 2, 4):
            self.engine.warmup(batch)
        print(f"[LIPS] ready on {self.engine.name}")

    async def infer(self, windows: np.ndarray) -> np.ndarray:
        # The model is compute-bound: keep the replica's event loop free.
        return await asyncio.to_thread(self.engine, windows)


class Lips(ModuleWithHandle):
    """Lip sync: ARKit blend shapes from the voice.

    input: huri.audio.out (Audio)
    output: lips.face (FaceWeights)

    The voice is cut in windows of 2.13 s (see huri_lips.stream): each
    yields `hop_frames` frames, from audio up to `lookahead_frames` after
    them; an utterance's last frames come when it ends (Audio.end). Frames
    are post-processed as LAM does: the mouth closes in silence, a light
    smoothing, left and right shapes alike.

    :hop_frames: frames per window (15: 0.5 s).
    :lookahead_frames: audio past the frames each window hears (6: 0.2 s).
    :max_batch: most windows per model call.
    :silence_rms: quieter frames are silence (mouth closed after 7 of them).
    :smooth_frames: Savitzky-Golay window, odd (0: none). LAM's 5 frames
        smooth away quick closures ("p", "b", "m": 98% -> 81% closed).
    :symmetric: same weight for left and right shapes.
    :delay_s: how much later than LAM puts them the frames are shown (see
        _DELAY_S).
    """

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
        self._t0: Optional[float] = None  # pts of the utterance's start
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
                        # A new utterance began without the last one's end.
                        async for face in self._run(final=True):
                            yield face
                    elif at > _GAP_S:
                        self._stream.add(np.zeros(int(at * SAMPLE_RATE), np.float32))
                if self._t0 is None:
                    self._t0 = float(audio.pts)
                self._stream.add(to_16k(audio.data, audio.sample_rate))
            if self._t0 is None:
                return  # a lone end marker
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
