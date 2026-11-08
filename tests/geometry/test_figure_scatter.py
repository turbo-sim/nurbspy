"""Measurement overlays survive scene persistence without becoming editable geometry."""

from io import StringIO

import numpy as np
import pytest

import nurbspy as nrb


def make_figure():
    return nrb.Figure(nrb.NurbsCurve([[0, 1], [0, 1]]), resolution=8)


def test_scatter_round_trip_copies_coordinates_and_preserves_reference_opacity():
    points = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    original = points.copy()
    fig = make_figure().update(display={"opacity": 0.25})
    fig.scatter(points, color="#123456", size=4.0, depth_test=True)
    points[:] = 9
    restored = nrb.Figure.from_yaml(StringIO(fig.to_yaml()))
    assert restored.to_state() == fig.to_state()
    assert restored.names == ("geometry",)
    assert restored.settings()["opacity"] == 0.25
    marker = restored.to_state()["markers"][0]
    np.testing.assert_array_equal(marker["points"][:2], original)
    assert marker["points"][2] == [0.0, 0.0, 0.0]
    assert marker["depth_test"] is True
    marker["points"][0][0] = 100
    assert restored.to_state()["markers"][0]["points"][0][0] == 0


def test_older_states_without_opacity_or_annotations_still_load():
    state = make_figure().to_state()
    assert "markers" not in state
    del state["objects"]["geometry"]["display"]["opacity"]
    assert nrb.Figure.from_state(state).settings()["opacity"] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"points": [[0.0, np.nan], [0.0, 1.0]]},
        {"points": [[0.0, 1.0], [0.0, 1.0]], "color": "red"},
        {"points": [[0.0, 1.0], [0.0, 1.0]], "size": 0},
        {"points": [[0.0, 1.0], [0.0, 1.0]], "depth_test": 1},
    ],
)
def test_invalid_marker_data_is_rejected_when_loading_a_scene(kwargs):
    state = make_figure().to_state()
    state["markers"] = [
        {"color": "#123456", "size": 3.0, "depth_test": False, **kwargs}
    ]
    with pytest.raises(ValueError):
        nrb.Figure.from_state(state)


@pytest.mark.parametrize("opacity", [-0.01, 1.01, True, float("nan")])
def test_invalid_opacity_is_rejected(opacity):
    with pytest.raises(ValueError, match="opacity"):
        make_figure().update(display={"opacity": opacity})
