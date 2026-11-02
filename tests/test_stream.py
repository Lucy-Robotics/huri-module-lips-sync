import numpy as np
import pytest

from huri_events_lips import ARKIT_INDEX, N_SHAPES
from huri_lips.config import SAMPLE_RATE, WINDOW_FRAMES, WINDOW_SAMPLES
from huri_lips.stream import LipStream, Smoother, frame_sample, to_16k

JAW = ARKIT_INDEX["jawOpen"]


def raw(**args):
    return Smoother(silence_rms=-1, blend=0, savgol=0, symmetric=False, **args)


class FrameClock:
    def __init__(self):
        self.calls = []

    def __call__(self, windows):
        self.calls.append(windows.shape[0])
        frames = np.arange(WINDOW_FRAMES, dtype=np.float32) / 100
        return np.tile(frames[None, :, None], (windows.shape[0], 1, N_SHAPES))


def speech(seconds, level=0.1):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_frames_come_hop_at_a_time_once_the_lookahead_is_heard():
    model = FrameClock()
    stream = LipStream(model, hop=15, lookahead=6, smoother=raw())
    assert stream.push(speech(0.6)) == []
    (first,) = stream.push(speech(0.1))
    start, frames = first
    assert start == 0 and frames.shape == (15, N_SHAPES)
    np.testing.assert_allclose(frames[:, JAW], np.arange(43, 58) / 100)


def test_flush_emits_the_rest_and_resets():
    stream = LipStream(FrameClock(), hop=15, lookahead=6, smoother=raw())
    out = stream.push(speech(1.0)) + stream.flush()
    starts = [s for s, _ in out]
    assert starts == [0, 15]
    assert sum(f.shape[0] for _, f in out) == 30
    assert stream.audio.size == 0 and stream.emitted == 0


def test_an_utterance_gets_ceil_of_its_frames():
    stream = LipStream(FrameClock(), smoother=raw())
    out = stream.push(speech(0.71)) + stream.flush()
    assert sum(f.shape[0] for _, f in out) == 22


def test_windows_hold_the_audio_they_end_with():
    stream = LipStream(FrameClock(), hop=15, lookahead=6)
    audio = speech(3.0)
    stream.add(audio)
    (window,) = stream.windows([45])
    end = frame_sample(45 + 15 + 6)
    np.testing.assert_array_equal(window, audio[end - WINDOW_SAMPLES : end])
    (first,) = stream.windows([0])
    assert not first[: WINDOW_SAMPLES - frame_sample(21)].any()


def test_long_audio_is_inferred_in_batches():
    model = FrameClock()
    stream = LipStream(model, max_batch=3, smoother=raw())
    stream.push(speech(5.0))
    assert model.calls == [3, 3, 3]


def test_emit_refuses_windows_out_of_order():
    stream = LipStream(smoother=raw())
    stream.add(speech(2.0))
    weights = FrameClock()(np.zeros((1, WINDOW_SAMPLES)))
    with pytest.raises(ValueError):
        stream.emit([15], weights)


def test_the_mouth_closes_in_silence():
    open_mouth = lambda w: np.full((w.shape[0], WINDOW_FRAMES, N_SHAPES), 0.6)  # noqa
    stream = LipStream(open_mouth, smoother=Smoother(savgol=0))
    audio = np.concatenate([speech(1.0), np.zeros(SAMPLE_RATE, np.float32)])
    frames = np.concatenate([f for _, f in stream.push(audio) + stream.flush()])
    assert frames[10:25, JAW].min() > 0.5
    assert frames[40:, JAW].max() < 0.1
    eye = ARKIT_INDEX["eyeBlinkLeft"]
    assert frames[40:, eye].min() > 0.5


def test_left_and_right_are_made_alike_but_not_the_gaze():
    def lopsided(windows):
        out = np.zeros((windows.shape[0], WINDOW_FRAMES, N_SHAPES), np.float32)
        out[..., ARKIT_INDEX["mouthSmileLeft"]] = 0.8
        out[..., ARKIT_INDEX["eyeLookInLeft"]] = 0.8
        return out

    stream = LipStream(lopsided, smoother=Smoother(blend=0, savgol=0))
    frames = np.concatenate([f for _, f in stream.push(speech(1.0)) + stream.flush()])
    np.testing.assert_allclose(frames[:, ARKIT_INDEX["mouthSmileRight"]], 0.4)
    np.testing.assert_allclose(frames[:, ARKIT_INDEX["mouthSmileLeft"]], 0.4)
    np.testing.assert_allclose(frames[:, ARKIT_INDEX["eyeLookInLeft"]], 0.8)


def test_the_first_frames_ease_out_of_the_neutral_face():
    full = lambda w: np.ones((w.shape[0], WINDOW_FRAMES, N_SHAPES))  # noqa
    stream = LipStream(full, smoother=Smoother(silence_rms=-1, savgol=0))
    frames = np.concatenate([f for _, f in stream.push(speech(1.0)) + stream.flush()])
    assert frames[0, JAW] < frames[1, JAW] < frames[2, JAW] < 1.0
    np.testing.assert_allclose(frames[5:, JAW], 1.0)


def test_resampling_to_16k():
    assert to_16k(np.zeros(22050, np.float32), 22050).size == SAMPLE_RATE
    assert to_16k(np.zeros(24000, np.float32), 24000).size == SAMPLE_RATE
    same = speech(0.5)
    assert to_16k(same, SAMPLE_RATE) is not None
    np.testing.assert_array_equal(to_16k(same, SAMPLE_RATE), same)
