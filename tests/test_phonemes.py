import numpy as np

from huri_events_lips import ARKIT_INDEX, N_VISEMES
from huri_events_lips import VISEME_INDEX as V
from huri_lips.phonemes import (
    arkit_from_visemes,
    merge_marks,
    viseme_frames,
    viseme_of,
)

FPS = 30


def test_ipa_to_oculus_visemes():
    assert [viseme_of(p) for p in "pbmfvtdsznlaeiou"] == [
        "PP",
        "PP",
        "PP",
        "FF",
        "FF",
        "DD",
        "DD",
        "SS",
        "SS",
        "nn",
        "nn",
        "aa",
        "E",
        "I",
        "O",
        "U",
    ]
    assert viseme_of(" ") is None and viseme_of("h") is None
    assert viseme_of("^") == "sil" and viseme_of(",") == "sil"


def test_stress_goes_to_the_next_phoneme_and_length_to_the_last():
    merged = merge_marks(
        [("b", 0.0, 0.1), ("ˈ", 0.1, 0.15), ("ɑ", 0.15, 0.3), ("ː", 0.3, 0.4)]
    )
    assert merged == [("b", 0.0, 0.1), ("ɑ", 0.1, 0.4)]


def test_frames_follow_the_phonemes_a_little_ahead():
    phones = [("^", 0.0, 0.1), ("b", 0.1, 0.2), ("ɑ", 0.2, 0.5), ("$", 0.5, 0.6)]
    v = viseme_frames(phones, 0.6, FPS, lead=0.04)
    assert v.shape == (18, N_VISEMES)
    np.testing.assert_allclose(v.sum(axis=1), 1.0, atol=1e-5)
    assert v[0].argmax() == V["sil"]
    assert v[4, V["PP"]] > 0.8
    assert v[10].argmax() == V["aa"]
    assert v[15, V["sil"]] > v[15, V["aa"]]


def test_an_arkit_face_for_rigs_without_visemes():
    v = np.zeros((2, N_VISEMES), np.float32)
    v[0, V["aa"]] = 1.0
    v[1, V["PP"]] = 1.0
    arkit = arkit_from_visemes(v)
    assert arkit[0, ARKIT_INDEX["jawOpen"]] > 0.5
    assert arkit[1, ARKIT_INDEX["jawOpen"]] == 0.0
    assert arkit[1, ARKIT_INDEX["mouthPressLeft"]] > 0.3
