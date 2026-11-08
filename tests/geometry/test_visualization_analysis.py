"""Analytical diagnostics, spatial promotion, and exact export-preview contracts."""

import base64
import json
import struct
import numpy as np
import pytest

from nurbspy import NurbsCurve, NurbsSurface
from nurbspy.visualization.analysis import (
    scalar_field,
    frenet_frames,
    display_options,
    control_point_normal,
)
from nurbspy.visualization.model import _GeometryModel
from nurbspy.visualization.desktop import _DesktopBridge
from tests.geometry.test_web_explorer_transport import app_for


def parabola():
    return NurbsCurve(np.array([[0.0, 0.5, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))


def plane():
    x, y = np.meshgrid([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], indexing="ij")
    return NurbsSurface(np.stack((x, y, np.zeros_like(x))))


def packet(model, **options):
    state = model.initial_state
    return dict(
        revision=7,
        control_points=state["control_points"],
        weights=state["weights"],
        **options
    )


def test_control_point_normals_use_native_greville_locations_and_follow_geometry():
    for index, u in enumerate([0.0, 0.5, 1.0]):
        normal = control_point_normal(parabola(), index)
        np.testing.assert_allclose(normal["parameters"], [u])
        np.testing.assert_allclose(
            normal["direction"], np.array([-2 * u, 1, 0]) / np.sqrt(1 + 4 * u * u)
        )
    surface = plane()
    normal = control_point_normal(surface, 4)
    np.testing.assert_allclose(normal["parameters"], [0.5, 0.5])
    np.testing.assert_allclose(normal["direction"], [0, 0, 1])
    surface.P[2] = 0.5 * surface.P[0]
    np.testing.assert_allclose(
        control_point_normal(surface, 4)["direction"],
        np.array([-0.5, 0, 1]) / np.sqrt(1.25),
    )


@pytest.mark.parametrize("index", [-1, 3, True, 0.5, None])
def test_control_point_normal_validates_the_selected_index(index):
    with pytest.raises(ValueError, match="index"):
        control_point_normal(parabola(), index)


def test_normal_constraint_reports_undefined_and_nonsmooth_geometry():
    line = NurbsCurve(np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    collapsed = NurbsSurface(np.zeros((3, 2, 2)))
    joined = NurbsCurve(
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        degree=2,
        knots=np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]),
    )
    for geometry, index in [(line, 0), (collapsed, 0), (joined, 2)]:
        normal = control_point_normal(geometry, index)
        assert normal["direction"] is None and normal["reason"]


def test_normal_inspection_matches_desktop_and_embedded_transports():
    from nurbspy.visualization import GeometryEditor

    model = GeometryEditor(parabola(), id="normal", resolution=12)
    query = packet(model, normal_index=1)
    expected = model.inspect(query)
    bridge = _DesktopBridge(model, read_only=False)
    assert bridge.inspect(query) == expected
    app = app_for(model)
    response = app.server.test_client().post("/_nurbspy_web/inspect/normal", json=query)
    assert response.status_code == 200
    assert response.get_json() == expected


def test_scalar_values_match_analytical_parabola_and_cumulative_length():
    curve = parabola()
    u = np.linspace(0, 1, 20)
    curvature = scalar_field(curve, u, None, "curvature")
    np.testing.assert_allclose(curvature["values"], 2 / (1 + 4 * u * u) ** 1.5)
    np.testing.assert_allclose(
        scalar_field(curve, u, None, "speed")["values"], np.sqrt(1 + 4 * u * u)
    )
    distance = scalar_field(curve, u, None, "arc_length")["values"]
    assert distance[0] == 0 and np.all(np.diff(distance) > 0)
    assert distance[-1] == pytest.approx(
        0.5 * np.sqrt(5) + 0.25 * np.arcsinh(2), abs=1e-6
    )


def test_torsion_matches_spatial_polynomial_and_straight_curve_is_explicitly_undefined():
    curve = NurbsCurve(
        np.array([[0, 1 / 3, 2 / 3, 1], [0, 0, 1 / 3, 1], [0, 0, 0, 1.0]])
    )
    u = np.linspace(0, 1, 15)
    np.testing.assert_allclose(
        scalar_field(curve, u, None, "torsion")["values"],
        3 / (9 * u**4 + 9 * u * u + 1),
    )
    line = NurbsCurve(np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    field = scalar_field(line, u, None, "torsion")
    assert field["range"] is None and field["undefined"] == len(u)
    assert all(value is None for value in field["values"])
    assert frenet_frames(line)["omitted"] == 8


@pytest.mark.parametrize(
    "name", ["mean_curvature", "gaussian_curvature", "principal_min", "principal_max"]
)
def test_planar_surface_curvatures_are_zero(name):
    u = np.linspace(0, 1, 10)
    field = scalar_field(plane(), u, u, name)
    np.testing.assert_allclose(field["values"], 0)
    assert field["undefined"] == 0


def test_frenet_frames_are_orthonormal_and_use_approximately_equal_arc_spacing():
    frames = frenet_frames(parabola(), count=8, scale=0.1)
    assert len(frames["frames"]) == 8 and frames["omitted"] == 0
    for frame in frames["frames"]:
        matrix = np.stack([frame[key] for key in ("tangent", "normal", "binormal")])
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1)
    origins = np.array([frame["origin"] for frame in frames["frames"]])
    chords = np.linalg.norm(np.diff(origins, axis=0), axis=1)
    assert chords.max() / chords.min() < 1.015


def test_older_planar_inputs_become_editable_spatial_snapshots_without_mutation():
    original = NurbsCurve(parabola().P[:2])
    before = original.P.copy()
    model = _GeometryModel(original, id="spatial")
    assert model.dimension == 3
    state = model.initial_state
    np.testing.assert_allclose(state["control_points"][2], 0)
    state["control_points"][2][1] = 0.7
    assert model.geometry_from_state(state).P[2, 1] == 0.7
    np.testing.assert_array_equal(original.P, before)


def test_export_preview_and_download_contain_identical_triangles():
    model = _GeometryModel(plane(), id="mesh", resolution=12)
    result = model.inspect(packet(model, export_mesh=True, nu=7, nv=9))
    mesh = result["mesh"]
    binary = base64.b64decode(mesh["stl"])
    count = struct.unpack_from("<I", binary, 80)[0]
    assert count == len(mesh["triangles"]) == 2 * 6 * 8
    records = np.frombuffer(
        binary,
        offset=84,
        dtype=np.dtype(
            [("normal", "<f4", (3,)), ("points", "<f4", (3, 3)), ("attribute", "<u2")]
        ),
    )
    expected = np.asarray(mesh["vertices"], dtype="<f4")[mesh["triangles"]]
    np.testing.assert_array_equal(records["points"], expected)


def test_desktop_and_http_inspection_agree_and_allow_view_only_analysis():
    from nurbspy.visualization import GeometryEditor

    editor = GeometryEditor(parabola(), id="analysis", resolution=12)
    query = packet(editor, field="curvature", frames=True)
    bridge = _DesktopBridge(editor, read_only=True)
    expected = bridge.inspect(query)
    assert "error" not in expected
    response = (
        app_for(editor)
        .server.test_client()
        .post("/_nurbspy_web/inspect/analysis", json=query)
    )
    assert response.status_code == 200 and response.get_json() == expected
    assert len(expected["field"]["values"]) == 60
    json.dumps(expected, allow_nan=False)


@pytest.mark.parametrize(
    "options",
    [
        dict(view="bad"),
        dict(quantity="speed"),
        dict(mesh_u=1000),
        dict(geometry_color="red"),
        dict(extra=True),
        dict(orientation_cube="yes"),
        dict(frames=True),
    ],
)
def test_invalid_surface_display_options_are_rejected(options):
    with pytest.raises(ValueError):
        display_options(options, surface=True)


def test_unknown_fields_and_incompatible_diagnostics_are_rejected():
    model = _GeometryModel(plane(), id="analysis")
    for options in (
        dict(field="torsion"),
        dict(frames=True),
        dict(export_mesh=True, nu=10000),
    ):
        with pytest.raises(ValueError):
            model.inspect(packet(model, **options))


def test_selected_point_colour_is_configurable_and_validated():
    settings = display_options(dict(selected_color="#8544cc"), surface=False)
    assert settings["selected_color"] == "#8544cc"
    with pytest.raises(ValueError, match="selected_color"):
        display_options(dict(selected_color="purple"), surface=False)
