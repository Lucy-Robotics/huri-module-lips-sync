import asyncio

import numpy as np
from huri_core.core.compat import check_manifests, check_modules, check_session
from huri_core.core.plugins import find_manifest, load_events, load_modules
from huri_core.testing import LocalHandle, ModuleHarness
from huri_events.audio import Audio

from huri_events_lips import ARKIT_INDEX, N_SHAPES, FaceWeights
from huri_lips.config import WINDOW_FRAMES
from huri_lips.module import Lips

SR = 22050
JAW = ARKIT_INDEX["jawOpen"]


class FakeLam:
    def __init__(self):
        self.batches = []

    async def infer(self, windows):
        self.batches.append(windows.shape[0])
        return np.full((windows.shape[0], WINDOW_FRAMES, N_SHAPES), 0.5, np.float32)


def voice(seconds, pts=0.0, end=False):
    n = int(seconds * SR)
    tone = 0.1 * np.sin(np.arange(n) * 2 * np.pi * 220 / SR)
    return Audio(tone.astype(np.float32), SR, end, pts)


def end_marker(pts):
    return Audio(np.zeros(0, np.float32), SR, True, pts)


def lips(**args):
    lam = FakeLam()
    args.setdefault("delay_s", 0.0)
    return lam, ModuleHarness(Lips, handle=LocalHandle(lam), **args)


def frames_of(faces):
    return sum(f.frames for f in faces)


def test_an_utterance_gets_all_its_frames_on_the_voice_timeline():
    lam, harness = lips()
    faces = harness.run(voice(1.0, 0.0), voice(1.0, 1.0), end_marker(2.0))
    assert all(isinstance(f, FaceWeights) and f.fps == 30 for f in faces)
    assert frames_of(faces) == 60
    t = 0.0
    for face in faces:
        assert abs(face.pts - t) < 1e-9
        t += face.frames / 30
    assert faces[-1].weights[-1, JAW] > 0.3


def test_frames_follow_the_utterance_pts():
    lam, harness = lips()
    faces = harness.run(voice(0.5, 3.0), end_marker(3.5))
    assert faces[0].pts == 3.0
    assert frames_of(faces) == 15


def test_frames_are_shown_later_than_lam_puts_them():
    lam, harness = lips(delay_s=0.13)
    faces = harness.run(voice(0.5, 3.0), end_marker(3.5))
    assert abs(faces[0].pts - 3.13) < 1e-9
    assert frames_of(faces) == 15
    assert abs(Lips.__init__.__defaults__[-1] - 0.13) < 1e-9


def test_the_end_resets_for_the_next_utterance():
    lam, harness = lips()

    async def two():
        first = await harness.feed_all(voice(0.5, 0.0), end_marker(0.5))
        second = await harness.feed_all(voice(0.5, 0.0), end_marker(0.5))
        return first, second

    first, second = asyncio.run(two())
    assert frames_of(first) == frames_of(second) == 15
    assert second[0].pts == 0.0


def test_a_gap_in_the_voice_is_silence():
    lam, harness = lips()
    faces = harness.run(voice(0.5, 0.0), voice(0.5, 1.0), end_marker(1.5))
    assert frames_of(faces) == 45
    weights = np.concatenate([f.weights for f in faces])
    assert weights[21:28, JAW].max() < 0.1
    assert weights[33:, JAW].min() > 0.3


def test_a_new_utterance_without_an_end_flushes_the_last():
    lam, harness = lips()
    faces = harness.run(voice(1.0, 0.0), voice(0.5, 0.0), end_marker(0.5))
    assert frames_of(faces) == 30 + 15
    assert [f.pts for f in faces][-1] < 0.5


def test_a_lone_end_marker_emits_nothing():
    lam, harness = lips()
    assert harness.run(end_marker(0.0)) == []
    assert lam.batches == []


def test_windows_are_batched():
    lam, harness = lips(max_batch=2)
    harness.run(voice(3.0, 0.0), end_marker(3.0))
    assert max(lam.batches) == 2


def test_the_server_accepts_the_module_and_its_manifest():
    modules = load_modules(["lips"])
    events = load_events()
    assert "lips.face" in events
    check_modules(modules, events)
    manifests = {"lips": find_manifest("lips")}
    assert None not in manifests.values()
    check_manifests(modules, manifests, events)


def test_a_session_with_a_tts_is_complete():
    class Tts:
        input_type, output_type = "huri.question", "huri.audio.out"

    check_session(
        [Tts, Lips], sent=["huri.question"], subscribed=["huri.audio.out", "lips.face"]
    )


def test_a_voice_with_phonemes_gets_visemes_not_lam():
    lam, harness = lips()
    audio = voice(0.5, 2.0)
    audio.phonemes = [
        ("^", 0.0, 0.1),
        ("m", 0.1, 0.2),
        ("a", 0.2, 0.4),
        ("$", 0.4, 0.5),
    ]
    faces = harness.run(audio, end_marker(2.5))
    assert len(faces) == 1 and lam.batches == []
    face = faces[0]
    assert face.pts == 2.0 and face.frames == 15
    assert face.visemes is not None and face.visemes.shape == (15, 15)
    assert face.weights[:, JAW].max() > 0.3
