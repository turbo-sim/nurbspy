"""Oloid construction correspondence, continuous rulings and seamless timing."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def demo(monkeypatch):
    spec = importlib.util.spec_from_file_location("oloid_animation_demo", ROOT / "demos/oloid/demo_oloid_animation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RESOLUTION", 8)
    return module


def test_ruling_endpoints_are_on_reference_circles_and_have_constant_length(demo):
    for sign in (-1, 1):
        a, b = demo.boundaries(np.linspace(0., 1., 1025), sign)
        np.testing.assert_allclose(np.linalg.norm(a - np.array([[-.5], [0], [0]]), axis=0), 1, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(b - np.array([[.5], [0], [0]]), axis=0), 1, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(b - a, axis=0), np.sqrt(3), atol=1e-14)
        assert np.all(a[2] == 0) and np.all(b[1] == 0)


def test_cubic_boundary_fit_reduces_controls_and_matches_analytical_circles(demo):
    parameter = np.linspace(0., 1., 16381)  # Independent of the fitting/check grids.
    for sign in (-1, 1):
        spline = demo.boundary_spline(sign)
        a, b = demo.boundaries(parameter, sign)
        expected = np.concatenate((a, b), axis=0).T
        errors = np.linalg.norm((spline(parameter) - expected).reshape(-1, 2, 3), axis=2)
        assert errors.max() <= demo.FIT_TOLERANCE
        assert 4 * len(spline.c) < 516 / 2
        # Simple cubic interior knots give C2 continuity throughout each boundary.
        _, multiplicities = np.unique(spline.t[4:-4], return_counts=True)
        assert np.all(multiplicities == 1)
        surface = demo.patch(1., sign)
        assert surface.p == 3 and surface.q == 1
        assert surface.P.shape == (3, len(spline.c), 2)
        u = np.linspace(0., 1., 257)
        v = np.linspace(.13, .87, len(u))
        a, b = demo.boundaries(u, sign)
        assert np.linalg.norm(surface.get_value(u, v) - ((1-v)*a + v*b), axis=0).max() <= demo.FIT_TOLERANCE


@pytest.mark.parametrize("fraction", [0., 1e-7, .125, .3194, 1.])
def test_partial_patches_reveal_the_same_fit_and_meet_the_generator(demo, fraction):
    u = np.linspace(0., 1., 129)
    for sign in (-1, 1):
        surface = demo.patch(fraction, sign)
        expected = demo.boundary_spline(sign)(fraction * u)
        np.testing.assert_allclose(surface.get_value(u, np.zeros_like(u)).T, expected[:, :3], atol=2e-14)
        np.testing.assert_allclose(surface.get_value(u, np.ones_like(u)).T, expected[:, 3:], atol=2e-14)
        endpoints = np.column_stack((surface.get_value(1., 0.)[:, 0], surface.get_value(1., 1.)[:, 0]))
        np.testing.assert_allclose(demo.generating_line(fraction, sign).P, endpoints, atol=2e-14)


def test_generator_is_continuous_across_both_shared_seams(demo):
    np.testing.assert_array_equal(demo.generating_line(1., 1).P, demo.generating_line(0., -1).P)
    np.testing.assert_array_equal(demo.generating_line(0., 1).P, demo.generating_line(1., -1).P)


def test_phase_order_cosine_speed_and_exact_loop_closure(demo):
    end = demo.total_seconds()
    assert demo.phase_at(0)[0] == demo.phase_at(end)[0] == 0.
    np.testing.assert_allclose([demo.phase_at(t)[0] for t in [0., 5.5, 10., 15.5, 20., 25.5, 30., 35.5, end]],
                               [0., .5, 1., .5, 0., .5, 1., .5, 0.], atol=1e-14)
    delta = 1e-4
    assert demo.phase_at(delta)[0] / delta < 1e-4
    assert demo.phase_at(end - delta)[0] / delta < 1e-4
    for extremum in np.linspace(0., end, 5):
        at = demo.phase_at(extremum)[0]
        assert abs(demo.phase_at(extremum + delta)[0] - at) / delta < 1e-4
        assert abs(demo.phase_at(extremum - delta)[0] - at) / delta < 1e-4
    figure = demo.make_figure()
    camera = dict(figure.camera)
    snapshots = [frame.to_state() for frame in demo.animation_frames(figure, camera, 5)]
    assert snapshots[0] == snapshots[-1]
    assert not snapshots[0]["objects"]["upper sweep"]["visible"]
    assert not snapshots[0]["objects"]["lower sweep"]["visible"]


def test_one_second_holds_at_each_extreme_and_continuous_camera_motion(demo):
    assert demo.HOLD_SECONDS == 1.
    cycle_seconds = demo.total_seconds() / demo.DEVELOPMENT_CYCLES
    for cycle in range(demo.DEVELOPMENT_CYCLES):
        for local_start, expected in [(0., 0.), (cycle_seconds / 2, 1.)]:
            start = cycle * cycle_seconds + local_start
            for offset in np.linspace(0., demo.HOLD_SECONDS, 31):
                assert demo.phase_at(start + offset)[0] == expected
            assert demo.phase_at(start + demo.HOLD_SECONDS + .01)[0] != expected
    figure = demo.make_figure()
    camera = dict(figure.camera)
    before = demo.HOLD_SECONDS / 4
    after = 3 * demo.HOLD_SECONDS / 4
    demo.orbit_camera(figure, before, camera)
    first = dict(figure.camera)
    demo.orbit_camera(figure, after, camera)
    assert figure.camera != first
    demo.following_camera(figure, demo.phase_at(before)[0])
    first = dict(figure.camera)
    demo.following_camera(figure, demo.phase_at(after)[0])
    assert figure.camera == first


def test_camera_rotates_at_constant_speed_through_all_development_phases(demo):
    figure = demo.make_figure()
    camera = dict(figure.camera)
    angles = []
    times = np.linspace(0., demo.total_seconds(), 129)
    radius = np.linalg.norm(np.subtract(camera["position"], camera["target"]))
    for seconds in times:
        demo.orbit_camera(figure, seconds, camera)
        relative = np.subtract(figure.camera["position"], camera["target"])
        angles.append(np.arctan2(relative[1], relative[0]))
        assert np.linalg.norm(relative) == pytest.approx(radius)
        assert figure.camera["position"][2] == camera["position"][2]
        assert figure.camera["target"] == camera["target"]
    expected = 2 * np.pi * demo.CAMERA_ORBITS * times / demo.total_seconds()
    np.testing.assert_allclose(np.unwrap(angles) - angles[0], expected, atol=1e-14)
    assert figure.camera == camera


def test_successive_developments_are_seen_from_opposite_sides(demo):
    figure = demo.make_figure()
    camera = dict(figure.camera)
    offsets = []
    for time in (demo.total_seconds() / 4, 3 * demo.total_seconds() / 4):
        assert demo.phase_at(time)[0] == 1.
        demo.orbit_camera(figure, time, camera)
        offsets.append(np.subtract(figure.camera["position"], camera["target"]))
    np.testing.assert_allclose(offsets[0][:2], -offsets[1][:2], atol=1e-14)


def test_following_camera_keeps_the_line_broadside_and_the_whole_oloid_in_frame(demo):
    figure = demo.make_figure()
    angle = np.linspace(0., 2 * np.pi, 513)
    circles = np.concatenate((np.stack((-.5 + np.cos(angle), np.sin(angle), 0*angle)),
                              np.stack((.5 + np.cos(angle), 0*angle, np.sin(angle)))), axis=1)
    for progress in np.linspace(0., 1., 257):
        demo.following_camera(figure, progress)
        camera = figure.camera
        position, target, up = (np.array(camera[key]) for key in ("position", "target", "up"))
        fraction, sign = demo.sweep_position(progress)
        line = demo.generating_line(fraction, sign).P
        np.testing.assert_allclose(target, line.mean(axis=1), atol=1e-14)
        offset = position - target
        assert np.linalg.norm(offset) == pytest.approx(demo.FOLLOW_DISTANCE)
        assert abs(np.dot(offset, line[:, 1]-line[:, 0])) < 1e-12
        forward = -offset / np.linalg.norm(offset)
        right = np.cross(forward, up)
        assert np.linalg.norm(right) > .1  # No look-at singularities/up-vector flips.
        right /= np.linalg.norm(right)
        vertical = np.cross(right, forward)
        points = circles - position[:, None]
        depths = forward @ points
        half_height = depths * np.tan(np.deg2rad(camera["fov"]) / 2)
        assert depths.min() > camera["near"]
        assert np.all(np.abs(vertical @ points) < half_height)
        assert np.all(np.abs(right @ points) < half_height * 4 / 3)


def test_following_camera_stays_outside_the_active_surface_and_closes_the_loop(demo):
    figure = demo.make_figure()
    for progress in np.linspace(0., 1., 65):
        demo.following_camera(figure, progress)
        fraction, sign = demo.sweep_position(progress)
        surface = demo.patch(1., sign)
        du = surface.get_derivative(fraction, .5, 1, 0)[:, 0]
        dv = surface.get_derivative(fraction, .5, 0, 1)[:, 0]
        normal = np.cross(du, dv)
        normal /= np.linalg.norm(normal)
        midpoint = np.array(figure.camera["target"])
        if np.dot(normal, midpoint) < 0:
            normal *= -1
        assert np.dot(normal, np.array(figure.camera["position"])-midpoint) > .9 * demo.FOLLOW_DISTANCE
    snapshots = [frame.to_state() for frame in demo.animation_frames(figure, figure.camera, 9, view="following")]
    assert snapshots[0] == snapshots[-1]


def test_export_switch_keeps_visualizer_and_creates_no_output(demo, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(demo, "EXPORT_ANIMATION", False)
    monkeypatch.setattr(demo, "OUTPUT", tmp_path / "unused")
    monkeypatch.setattr(demo.nrb.Figure, "show", lambda self: calls.append(self))
    assert demo.main() is None
    assert len(calls) == 1
    assert not demo.OUTPUT.exists()


def test_combined_frames_preserve_both_panels_and_reject_mismatched_sequences(demo, tmp_path):
    from PIL import Image
    left, right = [], []
    for index in range(2):
        a, b = tmp_path / f"left_{index}.png", tmp_path / f"right_{index}.png"
        Image.new("RGB", (64, 48), "red").save(a)
        Image.new("RGB", (64, 48), "blue").save(b)
        left.append(a)
        right.append(b)
    paths = demo.combine_frames(left, right, tmp_path / "combined")
    assert len(paths) == 2
    with Image.open(paths[0]) as combined:
        assert combined.size == (128, 48)
        assert combined.getpixel((0, 0)) == (255, 0, 0)
        assert combined.getpixel((127, 47)) == (0, 0, 255)
    with pytest.raises(ValueError, match="same nonzero"):
        demo.combine_frames(left, right[:1], tmp_path / "unused")


def test_gif_timing_and_palette_encoding(demo, tmp_path):
    from PIL import Image
    paths = []
    for index, colour in enumerate(("red", "green", "blue")):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (48, 36), colour).save(path)
        paths.append(path)
    gif = demo.create_gif(paths, tmp_path / "test.gif", fps=30, size=(48, 36))
    with Image.open(gif) as image:
        assert image.n_frames == 3 and image.info["loop"] == 0
        assert image.size == (48, 36)
        durations = []
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(image.info["duration"])
        assert sum(durations) == 100
