"""Independent derivative, constraint and flat-start checks for studies 03-05."""

import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "development/surface_fitting"
    monkeypatch.syspath_prepend(str(folder))
    return importlib.import_module("_experiment"), importlib.import_module("_advanced")


def test_paired_basis_values_and_derivatives_match_native_geometry(modules):
    e, a = modules
    model = e.flat(8)
    rng = np.random.default_rng(21)
    model.P[2] = rng.normal(0, 0.1, model.W.shape)
    model.W[:] = rng.uniform(0.5, 2.0, model.W.shape)
    uv = rng.uniform(0.01, 0.99, (53, 2))
    b, bu, bv = a.paired_basis(model, uv, derivatives=True)
    state = a.terms(b, model.P.reshape(3, -1).T, model.W.ravel(), bu, bv)
    np.testing.assert_allclose(state["s"], model.get_value(*uv.T).T, atol=2e-14)
    np.testing.assert_allclose(
        state["su"], model.get_derivative(*uv.T, 1, 0).T, atol=2e-13
    )
    np.testing.assert_allclose(
        state["sv"], model.get_derivative(*uv.T, 0, 1).T, atol=2e-13
    )


def test_area_orientation_and_data_jacobians_match_central_differences(modules):
    e, a = modules
    initial = e.flat(4)
    observations = e.data(9)
    x, bounds, geometry, values, info = a.constrained_problem(
        initial, observations, quadrature_order=4
    )
    rng = np.random.default_rng(8)
    x += rng.normal(0, 0.03, x.size)
    direction = rng.normal(size=x.size)
    direction /= np.linalg.norm(direction)
    state = values(x)
    expected_residual = state["jac"] @ direction
    expected_constraints = state["gjac"] @ direction
    step = 1e-6
    plus = values(x + step * direction)
    rp = plus["residual"].copy()
    gp = plus["g"].copy()
    minus = values(x - step * direction)
    np.testing.assert_allclose(
        expected_residual, (rp - minus["residual"]) / (2 * step), rtol=2e-5, atol=2e-9
    )
    np.testing.assert_allclose(
        expected_constraints, (gp - minus["g"]) / (2 * step), rtol=2e-5, atol=2e-8
    )


def test_flat_start_is_feasible_and_planar_control_net_stays_fixed(modules):
    e, a = modules
    initial = e.flat(4)
    x, _, geometry, values, info = a.constrained_problem(initial, e.data(9))
    state = values(x)
    assert state["area"] == pytest.approx(1.0, abs=1e-13)
    assert state["g"].max() < 0 and info["area_limit"] < info["reference_area"]
    x[: initial.W.size] = 0.03
    moved = geometry(x)
    np.testing.assert_array_equal(moved.P[:2], initial.P[:2])
    assert np.all(moved.W[[0, -1], :] == 1) and np.all(moved.W[:, [0, -1]] == 1)
    uv = np.stack((np.linspace(0, 1, 53), np.zeros(53)))
    np.testing.assert_allclose(moved.get_value(*uv)[:2], uv, atol=2e-14)


def solver_modules():
    """Load optional exact-Hessian IPM helpers for the nonlinear studies."""
    pytest.importorskip("jax")
    pytest.importorskip("pysolver_view")
    pytest.importorskip("pounce")
    return [
        importlib.import_module(name)
        for name in ("_joint", "_constrained", "_projection", "_alternating")
    ]


def test_joint_sparse_hessian_includes_parameter_couplings(modules):
    import jax

    e, _ = modules
    joint, _, _, _ = solver_modules()
    with jax.enable_x64():
        observations = e.data(9)
        p = joint.JointSurfaceProblem(e.flat(4), observations)
        rng = np.random.default_rng(45)
        x = p.x0.copy()
        x[:48] += rng.normal(0, 0.02, 48)
        x[p.surface_size :] = np.clip(x[p.surface_size :], 0.01, 0.99)
        model = p.surface(x)
        uv = p.parameters(x)
        error = model.get_value(*uv.T).T - observations["points"]
        expected = 500 * np.mean(np.sum(error**2, axis=1))
        expected += 1000 * (
            1e-4 / (4 * p.count) * np.sum((model.P[:2] - p.initial.P[:2]) ** 2)
            + 1e-6 / (2 * (p.count - 1)) * np.sum(np.log(model.W.ravel()[1:]) ** 2)
        )
        np.testing.assert_allclose(p.fitness(x)[0], expected, rtol=2e-13)
        rows, cols = p.hessian_sparsity()
        values = p.hessians(x)[0]
        h = np.zeros((len(x), len(x)))
        h[rows, cols] = values
        h[cols, rows] = values
        v = rng.normal(size=len(x))
        v /= np.linalg.norm(v)
        finite = (p.gradient(x + 1e-5 * v)[0] - p.gradient(x - 1e-5 * v)[0]) / 2e-5
        np.testing.assert_allclose(h @ v, finite, rtol=2e-5, atol=1e-7)
        assert np.unique(rows * len(x) + cols).size == len(rows)
        assert np.any((rows >= p.surface_size) & (cols < p.surface_size))


def test_constrained_full_lagrangian_hessian_matches_independent_derivatives(modules):
    import jax

    e, _ = modules
    _, constrained, _, _ = solver_modules()
    with jax.enable_x64():
        p = constrained.ConstrainedSurfaceProblem(e.flat(4), e.data(9))
        rng = np.random.default_rng(12)
        x = p.x0 + rng.normal(0, 0.03, len(p.x0))
        lam = rng.normal(0, 0.1, p.constraint_count)
        rows, cols = p.hessianstructure()
        values = p.hessian(x, lam, 0.7)
        h = np.zeros((len(x), len(x)))
        h[rows, cols] = values
        h[cols, rows] = values
        v = rng.normal(size=len(x))
        v /= np.linalg.norm(v)

        def lag_gradient(point):
            return 0.7 * p.gradient(point) + p.values(point)["gjac"].T @ lam

        finite = (lag_gradient(x + 1e-5 * v) - lag_gradient(x - 1e-5 * v)) / 2e-5
        np.testing.assert_allclose(h @ v, finite, rtol=2e-5, atol=1e-7)
        assert len(p.jrows) < p.constraint_count * len(x)


def test_projection_reuses_public_api_and_refreshes_changed_geometry(
    modules, monkeypatch
):
    import jax
    import nurbspy.jax as nrb

    e, _ = modules
    projection = importlib.import_module("_projection")
    calls = dict(prepare=0, project=0)
    original_prepare = nrb.prepare_surface_projection
    original_project = nrb.PreparedSurfaceProjection.project_points

    def prepare(*args, **kwargs):
        calls["prepare"] += 1
        return original_prepare(*args, **kwargs)

    def project(*args, **kwargs):
        calls["project"] += 1
        return original_project(*args, **kwargs)

    monkeypatch.setattr(nrb, "prepare_surface_projection", prepare)
    monkeypatch.setattr(nrb.PreparedSurfaceProjection, "project_points", project)
    with jax.enable_x64():
        model = e.flat(4)
        points = np.array(
            [
                [-0.2, 0.25, 0.4],
                [0.25, 0.4, 0.3],
                [1.2, 0.75, -0.2],
                [0.6, 0.1, 0.5],
                [0.3, 0.8, -0.2],
            ]
        )
        projector = projection.SurfaceProjector(model, points, batch_size=4)
        uv, _, info = projector.project(model)
        np.testing.assert_allclose(uv, np.clip(points[:, :2], 0, 1), atol=1e-9)
        assert calls["prepare"] and calls["project"] and info["success"]
        model.P[2] = 0.2 * model.P[0]
        moved_uv, distances, info = projector.project(model, uv)
        expected = np.column_stack(
            (np.clip((points[:, 0] + 0.2 * points[:, 2]) / 1.04, 0, 1), points[:, 1])
        )
        np.testing.assert_allclose(moved_uv, expected, atol=1e-9)
        np.testing.assert_allclose(
            distances,
            np.linalg.norm(model.get_value(*expected.T).T - points, axis=1),
            atol=1e-12,
        )
        assert info["success"] and "nhev" not in info


def test_joint_and_alternating_preserve_flat_start_and_true_roles(modules):
    e, _ = modules
    joint, _, _, alternating = solver_modules()
    for solver, options in (
        (joint.joint, dict(max_passes=1)),
        (alternating.alternating, dict(max_cycles=1)),
    ):
        fitted, info, history, uv = solver(
            e.flat(4), e.data(9), max_iterations=3, **options
        )
        assert not info["success"]
        assert (
            history[0]["event_kind"] == "initial"
            and history[-1]["event_kind"] == "solution"
        )
        assert np.all(history[0]["surface"].P[2] == 0)
        assert info["nhev"] > 0 and info["kkt_seconds"] == 0
        assert uv.shape == (81, 2)
        np.testing.assert_array_equal(fitted.P, history[-1]["surface"].P)


def test_outputs_are_anchored_to_the_study_directory(modules):
    _, _ = modules
    root = Path(__file__).resolve().parents[2] / "development/surface_fitting"
    for script, folder in [
        ("demo_01_linear_least_squares", "01_linear"),
        ("demo_02_nonlinear_weights", "02_known_parameters"),
        ("demo_03_joint_parameters", "03_joint_parameters"),
        ("demo_04_alternating_projection", "04_alternating_projection"),
        ("demo_05_constrained_ipm", "05_constrained"),
    ]:
        demo = importlib.import_module(script)
        assert demo.OUTPUT == root / "output" / folder
        assert demo.GRID_SIZES == (4, 8, 16, 32)
        assert demo.SAMPLES_PER_AXIS == 45
