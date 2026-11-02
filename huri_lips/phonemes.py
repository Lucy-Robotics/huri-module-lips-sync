import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from huri_events_lips import ARKIT_INDEX, N_SHAPES, N_VISEMES, VISEME_INDEX

Phoneme = Tuple[str, float, float]

LEAD_S = 0.04
CLOSED = {"PP"}

_VISEME_OF: Dict[str, str] = {}
for _visemes, _phones in (
    ("sil", "^$_,.!?;:…"),
    ("PP", "pbm"),
    ("FF", "fv"),
    ("TH", "θð"),
    ("DD", "td"),
    ("kk", "kgɡŋxç"),
    ("CH", "ʃʒ"),
    ("SS", "sz"),
    ("nn", "nl"),
    ("RR", "ɹrɾɚɝ"),
    ("aa", "aɑæʌɐɒ"),
    ("E", "eɛəɜ"),
    ("I", "iɪj"),
    ("O", "oɔ"),
    ("U", "uʊw"),
):
    for _p in _phones:
        _VISEME_OF[_p] = _visemes

_ARKIT_OF: Dict[str, Dict[str, float]] = {
    "sil": {},
    "PP": {"mouthClose": 0.1, "mouthPressLeft": 0.4, "mouthPressRight": 0.4},
    "FF": {"mouthRollLower": 0.5, "mouthUpperUpLeft": 0.2, "mouthUpperUpRight": 0.2},
    "TH": {"jawOpen": 0.12, "tongueOut": 0.3},
    "DD": {"jawOpen": 0.15, "mouthStretchLeft": 0.1, "mouthStretchRight": 0.1},
    "kk": {"jawOpen": 0.2},
    "CH": {"jawOpen": 0.1, "mouthFunnel": 0.4, "mouthPucker": 0.2},
    "SS": {"jawOpen": 0.05, "mouthStretchLeft": 0.3, "mouthStretchRight": 0.3},
    "nn": {"jawOpen": 0.12},
    "RR": {"jawOpen": 0.1, "mouthFunnel": 0.2, "mouthPucker": 0.3},
    "aa": {"jawOpen": 0.6, "mouthLowerDownLeft": 0.3, "mouthLowerDownRight": 0.3},
    "E": {"jawOpen": 0.35, "mouthStretchLeft": 0.3, "mouthStretchRight": 0.3},
    "I": {
        "jawOpen": 0.2,
        "mouthStretchLeft": 0.4,
        "mouthStretchRight": 0.4,
        "mouthSmileLeft": 0.2,
        "mouthSmileRight": 0.2,
    },
    "O": {"jawOpen": 0.4, "mouthFunnel": 0.5},
    "U": {"jawOpen": 0.15, "mouthFunnel": 0.3, "mouthPucker": 0.7},
}
ARKIT_FROM_VISEMES = np.zeros((N_VISEMES, N_SHAPES), dtype=np.float32)
for _viseme, _shapes in _ARKIT_OF.items():
    for _shape, _w in _shapes.items():
        ARKIT_FROM_VISEMES[VISEME_INDEX[_viseme], ARKIT_INDEX[_shape]] = _w


def viseme_of(phoneme: str) -> Optional[str]:
    return _VISEME_OF.get(phoneme)


def merge_marks(phonemes: Sequence[Phoneme]) -> List[Phoneme]:
    out: List[Phoneme] = []
    pending = 0.0
    for p, t0, t1 in phonemes:
        if p in "ˈˌ":
            pending += t1 - t0
            continue
        if p == "ː" and out:
            q, a, _ = out[-1]
            out[-1] = (q, a, t1)
            continue
        out.append((p, t0 - pending, t1))
        pending = 0.0
    return out


def viseme_frames(
    phonemes: Sequence[Phoneme],
    seconds: float,
    fps: int = 30,
    lead: float = LEAD_S,
    blend: float = 0.6,
) -> np.ndarray:
    n = max(1, math.ceil(seconds * fps - 1e-9))
    out = np.zeros((n, N_VISEMES), dtype=np.float32)
    segments = []
    for p, t0, t1 in merge_marks(phonemes):
        v = viseme_of(p)
        if v is not None and t1 > t0:
            segments.append((VISEME_INDEX[v], t0 - lead, t1 - lead, v in CLOSED))
    half = blend / fps
    for i in range(n):
        t = i / fps
        lo, hi = t - half, t + half
        for v, a, b, closed in segments:
            overlap = min(hi, b) - max(lo, a)
            if overlap > 0:
                out[i, v] += overlap
        total = out[i].sum()
        if total <= 0:
            out[i, VISEME_INDEX["sil"]] = 1.0
        else:
            out[i] /= total
    for v, a, b, closed in segments:
        if not closed:
            continue
        first = max(0, math.ceil(a * fps - 1e-9))
        last = min(n - 1, math.floor((b - 1e-9) * fps))
        frames = (
            range(first, last + 1)
            if first <= last
            else [min(n - 1, max(0, round((a + b) / 2 * fps)))]
        )
        for i in frames:
            out[i] *= 0.15
            out[i, v] += 0.85
    return out


def arkit_from_visemes(visemes: np.ndarray) -> np.ndarray:
    return np.clip(visemes @ ARKIT_FROM_VISEMES, 0.0, 1.0).astype(np.float32)
