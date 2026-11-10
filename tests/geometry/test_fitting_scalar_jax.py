"""Independent numerical checks for the final sparse-Hessian case-02 solver."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import BSpline


@pytest.fixture
def modules(monkeypatch):
    """Load the development study only when its optional solvers are available."""
    pytest.importorskip("jax")
    pytest.importorskip("pysolver_view")
    pytest.importorskip("pounce")
    folder = Path(__file__).resolve().parents[2] / "development/surface_fitting"
    monkeypatch.syspath_prepend(str(folder))
    return importlib.import_module("_experiment"), importlib.import_module("_nonlinear")


def reference_gradient(surface, observations, x):
    """Differentiate numerator and denominator directly, independently of JAX."""
    count = surface.W.size
    a = BSpline.design_matrix(observations["axis"], surface.U, surface.p).toarray()
    b = BSpline.design_matrix(observations["axis"], surface.V, surface.q).toarray()
    controls = x[: 3 * count].reshape(*surface.W.shape, 3)
    weights = np.r_[1.0, np.exp(x[3 * count :])].reshape(surface.W.shape)
    denominator = a @ weights @ b.T
    prediction = (
        np.einsum("ui,ijc,vj->uvc", a, controls * weights[..., None], b)
        / denominator[..., None]
    )
    error = prediction - observations["points"].reshape(prediction.shape)
    adjoint = (1000.0 / len(observations["points"])) * error / denominator[..., None]
    numerator_gradient = np.einsum("ui,uvc,vj->ijc", a, adjoint, b)
    cp_gradient = weights[..., None] * numerator_gradient
    denominator_gradient = np.sum(adjoint * prediction, axis=-1)
    weight_gradient = weights * (
        np.sum(controls * numerator_gradient, axis=-1) - a.T @ denominator_gradient @ b
    )
    return np.r_[cp_gradient.ravel(), weight_gradient.ravel()[1:]]


@pytest.mark.parametrize("grid", [4, 8])
def test_sparse_exact_hessian_and_gradient_match_independent_derivatives(modules, grid):
    """Check all curvature terms and the recovered sparse matrix at a nonzero residual."""
    import jax

    e, n = modules
    initial, observations = e.flat(grid), e.data(17)
    with jax.enable_x64():
        problem = n.RationalSurfaceProblem(initial, observations)
        rng = np.random.default_rng(42)
        x = problem.x0 + rng.normal(0, 0.04, len(problem.x0))
        model = problem.surface(x)
        u, v = np.meshgrid(observations["axis"], observations["axis"], indexing="ij")
        error = model.get_value(u.ravel(), v.ravel()).T - observations["points"]
        np.testing.assert_allclose(
            problem.fitness(x)[0], 500.0 * np.mean(np.sum(error**2, axis=1)), rtol=2e-13
        )
        gradient = problem.gradient(x)[0]
        np.testing.assert_allclose(
            gradient,
            reference_gradient(initial, observations, x),
            rtol=2e-10,
            atol=2e-12,
        )
        rows, cols = problem.hessian_sparsity()
        hessian = np.zeros((len(x), len(x)))
        hessian[rows, cols] = problem.hessians(x)[0]
        hessian[cols, rows] = hessian[rows, cols]
        for _ in range(3):
            direction = rng.normal(size=len(x))
            direction /= np.linalg.norm(direction)
            step = 1e-5
            finite = (
                reference_gradient(initial, observations, x + step * direction)
                - reference_gradient(initial, observations, x - step * direction)
            ) / (2 * step)
            np.testing.assert_allclose(
                hessian @ direction, finite, rtol=3e-6, atol=3e-8
            )
            fd_value = (
                problem.fitness(x + step * direction)[0]
                - problem.fitness(x - step * direction)[0]
            ) / (2 * step)
            np.testing.assert_allclose(
                gradient @ direction, fd_value, rtol=2e-6, atol=2e-8
            )


def test_sparse_pattern_scales_to_32_without_dense_hessian(modules):
    """A fine cubic grid retains local support and a fixed number of AD colors."""
    _, n = modules
    rows, cols, colors = n.hessian_pattern((32, 32), (3, 3))
    dimension = 4 * 32**2 - 1
    assert len(colors) == dimension and len(np.unique(colors)) == 196
    # Cubic local support bounds storage linearly in the variable count.
    assert len(rows) < 64 * dimension
    assert np.all(rows >= cols) and np.max(rows) < dimension
    assert np.unique(rows * dimension + cols).size == len(rows)


def test_ipm_returns_converged_geometry_and_truthfully_labeled_history(modules):
    """Native convergence owns the final geometry; snapshots are not accepted iterates."""
    e, n = modules
    initial, observations = e.flat(4), e.data(17)
    fitted, info, history = n.nonlinear(initial, observations)
    assert info["success"] and info["status"] == 0
    assert info["native_dual_infeasibility"] < 1e-8
    assert info["nhev"] > 0 and info["kkt_seconds"] == 0.0
    assert (
        info["compile_seconds"] > 0 and info["setup_seconds"] >= info["compile_seconds"]
    )
    assert info["variables"] == 63 and info["bound_violation"] == 0.0
    np.testing.assert_array_equal(history[0]["surface"].P, initial.P)
    assert (
        history[0]["event_kind"] == "initial"
        and history[-1]["event_kind"] == "solution"
    )
    assert all(f["event_kind"] == "native_observation" for f in history[1:-1])
    assert all(f["accepted"] is None for f in info["native_history"])
    np.testing.assert_array_equal(history[-1]["surface"].P, fitted.P)
    assert np.all(initial.W == 1) and np.all(initial.P[2] == 0)
    assert fitted.W[0, 0] == 1 and fitted.W.min() >= 0.1 and fitted.W.max() <= 10.0
    assert history[-1]["training_rms"] < history[0]["training_rms"]


def test_budget_stop_is_not_labeled_converged(modules):
    """A partial native solve still exports its authoritative returned point."""
    e, n = modules
    fitted, info, history = n.nonlinear(e.flat(4), e.data(9), max_iterations=1)
    assert not info["success"] and info["iterations"] == 1
    assert info["status"] != 0
    assert history[-1]["event_kind"] == "solution"
    np.testing.assert_array_equal(fitted.P, history[-1]["surface"].P)


def test_archive_and_rebuild_preserve_solver_timings_and_native_roles(
    modules, monkeypatch, tmp_path
):
    """Round-trip actual solver output without recomputing fits or relabeling snapshots."""
    modules  # Require the optional numerical dependencies before loading the runner.
    study = importlib.import_module("_study")
    report = importlib.import_module("_report")
    captured = []

    def capture(rows, observations, directory, **options):
        captured.append(rows)
        return Path(directory) / "report.html"

    monkeypatch.setattr(study, "export_report", capture)
    records = study.run(
        method="rational", grid_sizes=(4,), samples=9, max_iterations=1, output=tmp_path
    )
    saved_seconds = records[0]["solve_seconds"]
    monkeypatch.setattr(report, "export_report", capture)
    report.rebuild_report(tmp_path)
    rebuilt = captured[-1][0]
    assert rebuilt["solve_seconds"] == saved_seconds
    assert rebuilt["compile_seconds"] == records[0]["compile_seconds"]
    assert rebuilt["history"][0]["event_kind"] == "initial"
    assert rebuilt["history"][-1]["event_kind"] == "solution"
    assert all(
        f["event_kind"] == "native_observation" for f in rebuilt["history"][1:-1]
    )
    np.testing.assert_array_equal(rebuilt["surface"].P, records[0]["surface"].P)
    assert not rebuilt["success"]
