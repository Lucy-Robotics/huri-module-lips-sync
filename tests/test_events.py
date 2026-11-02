import struct

import numpy as np
import pytest

from huri_events_lips import (
    ARKIT_INDEX,
    ARKIT_NAMES,
    N_SHAPES,
    N_VISEMES,
    VISEME_NAMES,
    FaceWeights,
)


def test_the_52_arkit_shapes_in_alphabetical_order():
    assert N_SHAPES == 52
    assert list(ARKIT_NAMES) == sorted(ARKIT_NAMES)
    assert ARKIT_INDEX["jawOpen"] == 24


def test_round_trip_keeps_frames_fps_and_pts():
    weights = np.random.default_rng(0).random((7, N_SHAPES), dtype=np.float32)
    face = FaceWeights(weights=weights, fps=30, pts=1.25)
    back = FaceWeights.from_wire(face.to_wire())
    assert back.frames == 7 and back.fps == 30 and back.pts == 1.25
    np.testing.assert_array_equal(back.weights, weights)
    np.testing.assert_array_equal(back.shape("jawOpen"), weights[:, 24])


def test_wire_layout_is_header_then_little_endian_floats():
    face = FaceWeights(weights=np.full((2, N_SHAPES), 0.5, np.float32), pts=2.0)
    wire = face.to_wire()
    assert struct.unpack(">dII", wire[:16]) == (2.0, 30, 2)
    assert len(wire) == 16 + 2 * N_SHAPES * 4
    assert struct.unpack("<f", wire[16:20]) == (0.5,)


def test_truncated_frames_are_refused():
    wire = FaceWeights(weights=np.zeros((3, N_SHAPES), np.float32)).to_wire()
    with pytest.raises(ValueError):
        FaceWeights.from_wire(wire[:-4])
    with pytest.raises(ValueError):
        FaceWeights.from_wire(wire[:10])


def test_visemes_ride_after_the_arkit_weights():
    rng = np.random.default_rng(1)
    weights = rng.random((4, N_SHAPES), dtype=np.float32)
    visemes = rng.random((4, N_VISEMES), dtype=np.float32)
    wire = FaceWeights(weights=weights, visemes=visemes, pts=0.5).to_wire()
    assert len(wire) == 16 + 4 * (N_SHAPES + N_VISEMES) * 4
    back = FaceWeights.from_wire(wire)
    np.testing.assert_array_equal(back.weights, weights)
    np.testing.assert_array_equal(back.visemes, visemes)
    assert FaceWeights.from_wire(FaceWeights(weights=weights).to_wire()).visemes is None


def test_the_15_oculus_visemes():
    assert N_VISEMES == 15 and VISEME_NAMES[0] == "sil" and VISEME_NAMES[1] == "PP"
    with pytest.raises(ValueError):
        FaceWeights(weights=np.zeros((3, N_SHAPES)), visemes=np.zeros((2, N_VISEMES)))
