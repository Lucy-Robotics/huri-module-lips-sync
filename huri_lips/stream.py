"""One utterance's voice in, ARKit frames out (numpy only).

LAM's streaming model reads windows of WINDOW_FRAMES frames (64 at 30 fps,
2.13 s of 16 kHz audio) and is most accurate in a window's middle: it hears
the past and some of the future. So `LipStream` emits frames `hop` at a time,
each from a window that ends `lookahead` frames after them (silence past the
end of the utterance), the rest of the window being the audio before them:

    window (64 frames) ......................|
    [ context, already emitted ][ hop ][ la ]
                                 ^ emitted

The emitted frames are then post-processed as LAM does for streaming (see
`Smoother`): the mouth closes in silence, frames ease out of the neutral face
and across seams, a light Savitzky-Golay filter, left/right symmetry.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly, savgol_filter

from huri_events_lips import ARKIT_INDEX, ARKIT_NAMES, N_SHAPES
from huri_lips.config import FPS, SAMPLE_RATE, WINDOW_FRAMES, WINDOW_SAMPLES

# window (WINDOW_SAMPLES,) float32 -> (WINDOW_FRAMES, 52) weights; batched:
# (k, WINDOW_SAMPLES) -> (k, WINDOW_FRAMES, 52).
Infer = Callable[[np.ndarray], np.ndarray]

MOUTH = (
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "noseSneerLeft", "noseSneerRight", "cheekPuff",
)  # fmt: skip
_MOUTH_IDX = np.array([ARKIT_INDEX[n] for n in MOUTH])
_PAIRS = np.array(
    [
        (ARKIT_INDEX[n], ARKIT_INDEX[n[: -len("Left")] + "Right"])
        for n in ARKIT_NAMES
        if n.endswith("Left") and n[: -len("Left")] + "Right" in ARKIT_INDEX
    ]
)
# Eye shapes keep their asymmetry (a glance is not symmetric).
_PAIRS = np.array([p for p in _PAIRS if not ARKIT_NAMES[p[0]].startswith("eyeLook")])


def to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono float audio at `sample_rate` -> 16 kHz float32."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate == SAMPLE_RATE or samples.size == 0:
        return samples
    g = np.gcd(int(sample_rate), SAMPLE_RATE)
    return resample_poly(samples, SAMPLE_RATE // g, int(sample_rate) // g).astype(
        np.float32
    )


def frame_sample(frame: int) -> int:
    """The 16 kHz sample at which 30 fps frame `frame` starts."""
    return (frame * SAMPLE_RATE) // FPS


def frame_rms(audio: np.ndarray, first: int, count: int) -> np.ndarray:
    """Loudness of frames first..first+count of `audio` (16 kHz)."""
    out = np.zeros(count, dtype=np.float32)
    for i in range(count):
        a, b = frame_sample(first + i), frame_sample(first + i + 1)
        seg = audio[a:b]
        out[i] = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
    return out


class Smoother:
    """LAM's streaming post-processing, frame batch after frame batch.

    Each batch is processed with up to `history` frames before it (already
    sent, not changed again), as LAM's `apply_expression_postprocessing`.

    :silence_rms: frames quieter than this are silence...
    :silence_frames: ...when at least this many follow each other: the mouth
        shapes then fall to 10%.
    :blend: frames over which a batch eases from the previous frame (from the
        neutral face for the first).
    :savgol: Savitzky-Golay window (odd, frames; 0 or 1: off).
    :symmetric: average left and right shapes (not the eyes' gaze).
    """

    def __init__(
        self,
        silence_rms: float = 0.001,
        silence_frames: int = 7,
        blend: int = 3,
        savgol: int = 5,
        symmetric: bool = True,
        history: int = WINDOW_FRAMES,
    ):
        self.silence_rms = silence_rms
        self.silence_frames = silence_frames
        self.blend = blend
        self.savgol = savgol
        self.symmetric = symmetric
        self.history = history
        self._frames = np.zeros((0, N_SHAPES), dtype=np.float32)
        self._rms = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._frames = self._frames[:0]
        self._rms = self._rms[:0]

    def __call__(self, frames: np.ndarray, rms: np.ndarray) -> np.ndarray:
        done = self._frames.shape[0]
        x = np.concatenate([self._frames, frames.astype(np.float32)])
        loud = np.concatenate([self._rms, rms.astype(np.float32)])
        self._close_in_silence(x, loud, done)
        self._ease_in(x, done)
        if self.savgol > 2 and x.shape[0] >= self.savgol:
            x[done:] = np.clip(
                savgol_filter(x, self.savgol, 2, axis=0, mode="mirror")[done:], 0, 1
            )
        if self.symmetric and len(_PAIRS):
            new = x[done:]
            mean = (new[:, _PAIRS[:, 0]] + new[:, _PAIRS[:, 1]]) / 2
            new[:, _PAIRS[:, 0]] = mean
            new[:, _PAIRS[:, 1]] = mean
        out = x[done:].copy()
        self._frames = x[-self.history :]
        self._rms = loud[-self.history :]
        return out

    def _close_in_silence(self, x: np.ndarray, loud: np.ndarray, done: int) -> None:
        quiet = loud < self.silence_rms
        i = 0
        while i < len(quiet):
            if not quiet[i]:
                i += 1
                continue
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if j - i >= self.silence_frames:
                lo = max(i, done)
                x[lo:j, _MOUTH_IDX] *= 0.1
                # Ease into and out of the closed mouth.
                for k in range(min(self.blend, j - lo)):
                    if lo == i and i > 0 and i + k < len(x):
                        w = (k + 1) / (self.blend + 1)
                        x[i + k] = x[i - 1] * (1 - w) + x[i + k] * w
                for k in range(min(self.blend, len(x) - j)):
                    w = (k + 1) / (self.blend + 1)
                    if j - 1 - k >= done:
                        x[j - 1 - k] = x[j] * (1 - w) + x[j - 1 - k] * w
            i = j

    def _ease_in(self, x: np.ndarray, done: int) -> None:
        ref = x[done - 1].copy() if done else np.zeros(N_SHAPES, np.float32)
        n = min(self.blend if done == 0 else self.blend + 2, x.shape[0] - done)
        for k in range(n):
            w = (k + 1) / (n + 1)
            x[done + k] = ref * (1 - w) + x[done + k] * w


class LipStream:
    """One utterance's 16 kHz voice -> ARKit frames, `hop` frames at a time.

    Frame i is heard at i / FPS seconds into the utterance. Two ways to drive
    it:

    - with an `infer` function: `push()` audio as it comes, `flush()` at the
      end of the utterance; both return [(first_frame, weights (n, 52))] for
      the frames now ready;
    - step by step, when the model is called asynchronously (the module):
      `add()` audio, `plan()` the windows to infer, `windows()` to get them,
      `emit()` their weights; `reset()` after the last one.

    :infer: the model on a batch of windows (see Infer).
    :hop: frames emitted per window (the window's fresh part).
    :lookahead: frames of audio after them the window also hears.
    :max_batch: most windows per call of `infer`.
    """

    def __init__(
        self,
        infer: Optional[Infer] = None,
        hop: int = 15,
        lookahead: int = 6,
        max_batch: int = 8,
        smoother: Optional[Smoother] = None,
    ):
        if hop <= 0 or lookahead < 0 or hop + lookahead > WINDOW_FRAMES:
            raise ValueError("need 0 < hop and hop + lookahead <= 64")
        self.infer = infer
        self.hop = hop
        self.lookahead = lookahead
        self.max_batch = max(1, max_batch)
        self.smoother = smoother if smoother is not None else Smoother()
        self.audio = np.zeros(0, dtype=np.float32)
        self.emitted = 0  # frames sent

    @property
    def seconds(self) -> float:
        return self.audio.size / SAMPLE_RATE

    def total_frames(self) -> int:
        return -(-self.audio.size * FPS // SAMPLE_RATE)  # ceil

    def add(self, samples16k: np.ndarray) -> None:
        self.audio = np.concatenate([self.audio, samples16k.astype(np.float32)])

    def plan(self, final: bool = False) -> List[int]:
        """First frames of the windows now ready (all the rest when final)."""
        total = self.total_frames()
        starts = []
        start = self.emitted
        while start < total and (final or start + self.hop + self.lookahead <= total):
            starts.append(start)
            start += self.hop
        return starts

    def windows(self, starts: List[int]) -> np.ndarray:
        return np.stack([self._window(s) for s in starts])

    def emit(
        self, starts: List[int], weights: np.ndarray
    ) -> List[Tuple[int, np.ndarray]]:
        """The post-processed frames of windows `starts` from their weights."""
        total = self.total_frames()
        out = []
        for s, w in zip(starts, np.asarray(weights, dtype=np.float32)):
            if s != self.emitted:
                raise ValueError(f"window at frame {s}, expected {self.emitted}")
            count = min(self.hop, total - s)
            # The window ends at s + hop + lookahead: its last frame is the
            # frame before that end.
            first = WINDOW_FRAMES - self.hop - self.lookahead
            rms = frame_rms(self.audio, s, count)
            out.append((s, self.smoother(w[first : first + count], rms)))
            self.emitted = s + count
        return out

    def reset(self) -> None:
        self.audio = self.audio[:0]
        self.emitted = 0
        self.smoother.reset()

    def push(self, samples16k: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        self.add(samples16k)
        return self._run(final=False)

    def flush(self) -> List[Tuple[int, np.ndarray]]:
        out = self._run(final=True)
        self.reset()
        return out

    def _run(self, final: bool) -> List[Tuple[int, np.ndarray]]:
        if self.infer is None:
            raise RuntimeError("no infer function: use plan/windows/emit")
        starts = self.plan(final)
        out = []
        for i in range(0, len(starts), self.max_batch):
            batch = starts[i : i + self.max_batch]
            out += self.emit(batch, self.infer(self.windows(batch)))
        return out

    def _window(self, start: int) -> np.ndarray:
        end = frame_sample(start + self.hop + self.lookahead)
        begin = end - WINDOW_SAMPLES
        window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        lo, hi = max(begin, 0), min(end, self.audio.size)
        if hi > lo:
            window[lo - begin : hi - begin] = self.audio[lo:hi]
        return window
