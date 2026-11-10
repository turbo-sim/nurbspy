"""Independent checks of the reusable surface-fitting API and sparse derivatives."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.interpolate import BSpline

import nurbspy as nrb


def patch(cls=nrb.NurbsSurface, *, small=False):
    """Use unequal degrees, nonunit knot domains and nonunit rational weights."""
    if small:
        u_knots = np.r_[np.full(3, -2.0), np.full(3, 3.0)]
        v_knots = np.r_[np.full(4, 4.0), np.full(4, 8.0)]
    else:
        u_knots = np.r_[np.full(3, -2.0), -0.35, np.full(3, 3.0)]
        v_knots = np.r_[np.full(4, 4.0), 5.1, 6.75, np.full(4, 8.0)]
    nu, nv = len(u_knots) - 3, len(v_knots) - 4
    gu = np.array([u_knots[i + 1 : i + 3].mean() for i in range(nu)])
    gv = np.array([v_knots[j + 1 : j + 4].mean() for j in range(nv)])
    u, v = np.meshgrid(gu, gv, indexing="ij")
    controls = np.stack(
        (0.6 * u + 0.2 * v, -0.3 * u + 0.5 * v, 0.2 * np.sin(u) + 0.15 * np.cos(v))
    )
    weights = 2.7 * np.exp(0.3 * np.sin(u) * np.cos(v))
    return cls(
        controls,
        weights=weights,
        u_degree=2,
        v_degree=3,
        u_knots=u_knots,
        v_knots=v_knots,
    )


def samples(count=143, *, boundaries=True):
    """Scattered paired labels, including the four actual domain corners."""
    uv = np.random.default_rng(174).uniform([-2.0, 4.0], [3.0, 8.0], (count, 2))
    if boundaries:
        uv[:4] = [[-2, 4], [-2, 8], [3, 4], [3, 8]]
    return uv


def flat_like(surface):
    controls = np.array(surface.P, copy=True)
    controls[-1] = 0
    return nrb.NurbsSurface(
        controls,
        weights=np.asarray(surface.W),
        u_degree=surface.p,
        v_degree=surface.q,
        u_knots=np.asarray(surface.U),
        v_knots=np.asarray(surface.V),
    )


def require_nonlinear():
    for name in ("jax", "equinox", "pysolver_view", "pounce"):
        pytest.importorskip(name)


def test_fixed_nonunit_weights_recover_scattered_rational_surface_without_mutation():
    target = patch()
    initial = flat_like(target)
    before = (initial.P.copy(), initial.W.copy())
    uv = samples()
    points = target.get_value(*uv.T).T
    result = nrb.fit_surface(points, initial, parameters=uv, record_history=True)
    assert result.success and result.iterations == 0
    np.testing.assert_allclose(result.surface.P, target.P, rtol=1e-10, atol=2e-10)
    np.testing.assert_array_equal(result.surface.W, target.W)
    np.testing.assert_array_equal(result.parameters, uv)
    np.testing.assert_array_equal(initial.P, before[0])
    np.testing.assert_array_equal(initial.W, before[1])
    assert result.surface is not initial and result.rms < 1e-11
    assert result.history[0]["event_kind"] == "initial"
    assert result.history[-1]["event_kind"] == "solution"


def test_linear_fit_keeps_fixed_coordinate_components():
    target = patch()
    initial = flat_like(target)
    free = np.zeros(initial.P.shape, dtype=bool)
    free[2] = True
    uv = samples()
    result = nrb.fit_surface(
        target.get_value(*uv.T).T, initial, parameters=uv, control_mask=free
    )
    assert result.success
    np.testing.assert_array_equal(result.surface.P[:2], initial.P[:2])
    np.testing.assert_allclose(result.surface.P[2], target.P[2], atol=1e-10)
    np.testing.assert_array_equal(result.surface.W, initial.W)


def test_linear_fit_preserves_selected_geometry_backend(surface_backend):
    target = patch(surface_backend.NurbsSurface, small=True)
    initial = surface_backend.NurbsSurface(
        np.asarray(target.P) * np.array([1, 1, 0])[:, None, None],
        weights=target.W,
        u_degree=target.p,
        v_degree=target.q,
        u_knots=target.U,
        v_knots=target.V,
    )
    uv = samples(61)
    result = nrb.fit_surface(
        np.asarray(target.get_value(*uv.T)).T, initial, parameters=uv
    )
    assert isinstance(result.surface, surface_backend.NurbsSurface)
    np.testing.assert_allclose(result.surface.P, target.P, atol=2e-10)


def test_linear_fit_does_not_import_optional_optimization_dependencies():
    """A fresh interpreter catches accidental import-time JAX/Pounce dependencies."""
    script = """
import importlib.abc
import sys
class NoOptionalSolver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'equinox', 'pounce', 'pysolver_view'}:
            raise AssertionError('Linear fitting imported ' + fullname)
sys.meta_path.insert(0, NoOptionalSolver())
import numpy as np
import nurbspy as nrb
p = np.array([[[0.,0.],[1.,1.]], [[0.,1.],[0.,1.]], [[0.,0.],[0.,0.]]])
initial = nrb.NurbsSurface(p)
u,v = np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,6),indexing='ij')
uv = np.column_stack((u.ravel(),v.ravel()))
points = np.column_stack((uv, 0.2 + 0.3*uv[:,0] - 0.1*uv[:,1]))
result = nrb.fit_surface(points, initial, parameters=uv)
assert result.success and result.rms < 1e-12
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("fixed_weight", [False, True])
def test_weight_gauge_preserves_the_initial_surface(fixed_weight):
    from nurbspy.fitting._common import FitLayout

    initial = patch(small=True)
    mask = np.ones(initial.W.shape, dtype=bool)
    if fixed_weight:
        mask[-1, -1] = False
    layout = FitLayout(initial, optimize_weights=True, weight_mask=mask)
    represented = layout.surface(layout.x0)
    uv = samples(31)
    np.testing.assert_allclose(
        represented.get_value(*uv.T), initial.get_value(*uv.T), atol=3e-14
    )
    if fixed_weight:
        np.testing.assert_allclose(represented.W, initial.W, rtol=1e-15)
        np.testing.assert_array_equal(represented.W[~mask], initial.W[~mask])
    else:
        assert represented.W[0, 0] == 1
        np.testing.assert_allclose(represented.W, initial.W / initial.W[0, 0])


@pytest.mark.parametrize(
    "change",
    [
        "missing_parameters",
        "wrong_count",
        "outside_domain",
        "nonfinite_points",
        "unknown_mode",
    ],
)
def test_invalid_fitting_data_are_rejected(change):
    initial = patch(small=True)
    uv = samples(17)
    points = initial.get_value(*uv.T).T
    options = dict(parameters=uv)
    if change == "missing_parameters":
        options.clear()
    elif change == "wrong_count":
        options["parameters"] = uv[:-1]
    elif change == "outside_domain":
        options["parameters"] = uv.copy()
        options["parameters"][0, 0] = -2.01
    elif change == "nonfinite_points":
        points[0, 0] = np.nan
    else:
        options["correspondence"] = "unknown"
    with pytest.raises((ValueError, TypeError)):
        nrb.fit_surface(points, initial, **options)


def reference_gradient(
    problem, layout, points, x, scale, c_regularization, w_regularization, joint
):
    """Analytic rational gradient independent of the compiled AD implementation."""
    c, w = layout.unpack(x[: layout.size])
    uv = problem.parameters(x)
    initial = layout.initial
    a = BSpline.design_matrix(uv[:, 0], initial.U, initial.p).toarray()
    b = BSpline.design_matrix(uv[:, 1], initial.V, initial.q).toarray()
    basis = (a[:, :, None] * b[:, None, :]).reshape(len(uv), -1)
    rational = basis * w[None, :]
    rational /= rational.sum(axis=1, keepdims=True)
    prediction = rational @ c
    error = prediction - points
    gc = rational.T @ error / len(points)
    gc += c_regularization / layout.count * (c - layout.controls0)
    gw = np.sum(
        rational[:, :, None]
        * error[:, None, :]
        * (c[None, :, :] - prediction[:, None, :]),
        axis=(0, 2),
    ) / len(points)
    wi = layout.weight_indices
    gw[wi] += (
        w_regularization
        / max(len(wi), 1)
        * (np.log(w[wi]) - np.log(layout.weights0[wi]))
    )
    gradient = np.r_[gc.ravel()[layout.control_indices], gw[wi]]
    if joint:
        model = layout.surface(x[: layout.size])
        du = model.get_derivative(*uv.T, 1, 0).T
        dv = model.get_derivative(*uv.T, 0, 1).T
        gradient = np.r_[
            gradient,
            np.column_stack(
                (np.sum(du * error, axis=1), np.sum(dv * error, axis=1))
            ).ravel()
            / len(points),
        ]
    return scale * gradient


@pytest.mark.parametrize("joint", [False, True], ids=["fixed", "joint"])
def test_sparse_hessian_matches_independent_rational_derivatives(joint):
    require_nonlinear()
    import jax
    from nurbspy.fitting._common import FitLayout
    from nurbspy.fitting._nonlinear import SurfaceFitProblem
    from nurbspy.fitting._joint import JointSurfaceFitProblem

    initial = patch(small=not joint)
    if not joint:
        initial = nrb.NurbsSurface(
            initial.P[:2],
            weights=initial.W,
            u_degree=initial.p,
            v_degree=initial.q,
            u_knots=initial.U,
            v_knots=initial.V,
        )
    uv = samples(19, boundaries=False)
    points = initial.get_value(*uv.T).T + np.linspace(0.03, -0.02, initial.ndim)
    mask = np.ones(initial.P.shape, dtype=bool)
    mask[0, 0, 0] = False
    scale, cr, wr = 37.0, 0.003, 0.002
    with jax.enable_x64():
        layout = FitLayout(initial, optimize_weights=True, control_mask=mask)
        cls = JointSurfaceFitProblem if joint else SurfaceFitProblem
        problem = cls(
            layout,
            points,
            uv,
            objective_scale=scale,
            control_regularization=cr,
            weight_regularization=wr,
        )
        rng = np.random.default_rng(284)
        x = problem.x0.copy()
        x[: layout.size] += rng.normal(0, 0.03, layout.size)
        if joint:
            lower, upper = problem.get_bounds()
            x[layout.size :] = 0.43 * lower[layout.size :] + 0.57 * upper[layout.size :]
        c, w = layout.unpack(x[: layout.size])
        parameters = problem.parameters(x)
        error = layout.surface(x[: layout.size]).get_value(*parameters.T).T - points
        expected_value = 0.5 * np.mean(np.sum(error**2, axis=1))
        expected_value += 0.5 * cr / layout.count * np.sum((c - layout.controls0) ** 2)
        wi = layout.weight_indices
        expected_value += (
            0.5
            * wr
            / max(len(wi), 1)
            * np.sum((np.log(w[wi]) - np.log(layout.weights0[wi])) ** 2)
        )
        np.testing.assert_allclose(
            problem.fitness(x)[0], scale * expected_value, rtol=2e-12
        )
        gradient = reference_gradient(problem, layout, points, x, scale, cr, wr, joint)
        np.testing.assert_allclose(
            problem.gradient(x)[0], gradient, rtol=2e-10, atol=2e-11
        )
        rows, cols = problem.hessian_sparsity()
        values = problem.hessians(x)[0]
        assert np.all(rows >= cols)
        assert np.unique(rows * len(x) + cols).size == len(rows)
        hessian = np.zeros((len(x), len(x)))
        hessian[rows, cols] = values
        hessian[cols, rows] = values
        for _ in range(2):
            direction = rng.normal(size=len(x))
            direction /= np.linalg.norm(direction)
            step = 1e-5
            finite = (
                reference_gradient(
                    problem, layout, points, x + step * direction, scale, cr, wr, joint
                )
                - reference_gradient(
                    problem, layout, points, x - step * direction, scale, cr, wr, joint
                )
            ) / (2 * step)
            np.testing.assert_allclose(
                hessian @ direction, finite, rtol=3e-5, atol=2e-7
            )
        if joint:
            assert np.any((rows >= layout.size) & (cols < layout.size))


def test_nonlinear_budget_stop_has_truthful_history_and_returned_geometry():
    require_nonlinear()
    target = patch(small=True)
    initial = flat_like(target)
    uv = samples(35)
    result = nrb.fit_surface(
        target.get_value(*uv.T).T,
        initial,
        parameters=uv,
        optimize_weights=True,
        weight_bounds=(0.1, 10),
        objective_scale=100,
        max_iterations=1,
        record_history=True,
    )
    assert not result.success and result.iterations == 1
    assert result.history[0]["event_kind"] == "initial"
    assert result.history[-1]["event_kind"] == "solution"
    observations = [
        row for row in result.history if row["event_kind"] == "native_observation"
    ]
    assert observations
    assert all(row.get("accepted") is None for row in observations)
    actual_rms = np.sqrt(
        np.mean(
            np.sum(
                (result.surface.get_value(*uv.T).T - target.get_value(*uv.T).T) ** 2,
                axis=1,
            )
        )
    )
    np.testing.assert_allclose(result.rms, actual_rms, rtol=1e-11)
    assert result.timings["compile_seconds"] > 0
    assert result.timings["solve_seconds"] > 0


@pytest.mark.parametrize("mode", ["joint", "alternating"])
def test_unknown_parameters_can_be_fitted_on_a_completely_fixed_patch(mode):
    require_nonlinear()
    initial = patch(small=True)
    truth = samples(11, boundaries=False)
    points = initial.get_value(*truth.T).T
    guesses = truth + [0.03, -0.02]
    result = nrb.fit_surface(
        points,
        initial,
        correspondence=mode,
        initial_parameters=guesses,
        control_mask=False,
        optimize_weights=False,
        objective_scale=100,
        tolerance=1e-8,
        max_iterations=100,
        max_cycles=3,
    )
    assert result.success, result.message
    assert result.rms < 1e-7
    np.testing.assert_allclose(result.parameters, truth, atol=1e-7)
    np.testing.assert_array_equal(result.surface.P, initial.P)
    np.testing.assert_array_equal(result.surface.W, initial.W)


@pytest.mark.parametrize("mode", ["joint", "alternating"])
def test_geometric_fit_initializes_with_public_projection_in_actual_domain(
    mode, monkeypatch
):
    """World x/y are neither known labels nor guesses for a general patch."""
    pytest.importorskip("jax")
    pytest.importorskip("equinox")
    import nurbspy.jax as jnrb

    calls = []
    original = jnrb.prepare_surface_projection

    def prepare(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(jnrb, "prepare_surface_projection", prepare)
    initial = patch(small=True)
    before = initial.P.copy(), initial.W.copy()
    truth = samples(13, boundaries=False)
    points = initial.get_value(*truth.T).T
    assert np.max(abs(points[:, :2] - truth)) > 1
    result = nrb.fit_surface(
        points,
        initial,
        correspondence=mode,
        control_mask=False,
        optimize_weights=False,
        record_history=True,
    )
    assert result.success, result.message
    assert calls, "Initialization must reuse the public prepared projection API"
    assert result.rms < 1e-7
    np.testing.assert_allclose(result.parameters, truth, atol=1e-7)
    np.testing.assert_allclose(result.history[0]["parameters"], truth, atol=1e-7)
    np.testing.assert_array_equal(result.surface.P, before[0])
    np.testing.assert_array_equal(result.surface.W, before[1])
    np.testing.assert_array_equal(initial.P, before[0])
    np.testing.assert_array_equal(initial.W, before[1])


def test_joint_windows_cross_repeated_knots_and_preserve_full_curvature():
    """C1 knot repetitions must not trap a point on either side of a zero span."""
    require_nonlinear()
    import jax
    from nurbspy.fitting._common import FitLayout
    from nurbspy.fitting._joint import JointSurfaceFitProblem, parameter_windows

    u_knots = np.r_[np.full(4, -2.0), 0.4, 0.4, np.full(4, 3.0)]
    v_knots = np.r_[np.full(3, 4.0), 5.7, np.full(3, 8.0)]
    gu = np.array([u_knots[i + 1 : i + 4].mean() for i in range(6)])
    gv = np.array([v_knots[i + 1 : i + 3].mean() for i in range(4)])
    u, v = np.meshgrid(gu, gv, indexing="ij")
    initial = nrb.NurbsSurface(
        np.stack(
            (0.6 * u + 0.2 * v, -0.3 * u + 0.5 * v, 0.2 * np.sin(u) + 0.1 * np.cos(v))
        ),
        weights=2.2 * np.exp(0.1 * np.sin(u) * np.cos(v)),
        u_degree=3,
        v_degree=2,
        u_knots=u_knots,
        v_knots=v_knots,
    )
    uv = np.column_stack(
        (
            0.4 + np.array([-1, 1, -1, 1, -1, 1]) * 1e-4,
            np.array([4.6, 5.2, 6.1, 7.2, 5.1, 6.8]),
        )
    )
    lower, upper, _, indices, valid = parameter_windows(initial, uv)
    # There are exactly two nonempty u spans. The repeated knot separates them,
    # and points close to it must be able to move into either span in one pass.
    np.testing.assert_array_equal(lower[:, 0], -2.0)
    np.testing.assert_array_equal(upper[:, 0], 3.0)
    for row, pair in enumerate(uv):
        for probe_u in (0.4 - 0.06, 0.4 + 0.06):
            a = BSpline.design_matrix([probe_u], u_knots, 3).toarray()[0]
            b = BSpline.design_matrix([pair[1]], v_knots, 2).toarray()[0]
            support = set(np.flatnonzero(np.outer(a, b).ravel()))
            assert support <= set(indices[row][valid[row]])
    points = initial.get_value(*uv.T).T + [0.01, -0.02, 0.03]
    scale = 23.0
    with jax.enable_x64():
        layout = FitLayout(initial, optimize_weights=True)
        problem = JointSurfaceFitProblem(layout, points, uv, objective_scale=scale)
        rng = np.random.default_rng(471)
        x = problem.x0.copy()
        x[: layout.size] += rng.normal(0, 0.01, layout.size)
        across = uv.copy()
        across[:, 0] = 0.4 + np.where(uv[:, 0] < 0.4, 0.06, -0.06)
        x[layout.size :] = across.ravel()
        error = layout.surface(x[: layout.size]).get_value(*across.T).T - points
        np.testing.assert_allclose(
            problem.fitness(x)[0],
            0.5 * scale * np.mean(np.sum(error**2, axis=1)),
            rtol=3e-12,
        )
        expected = reference_gradient(problem, layout, points, x, scale, 0.0, 0.0, True)
        np.testing.assert_allclose(
            problem.gradient(x)[0], expected, rtol=2e-10, atol=2e-11
        )
        rows, cols = problem.hessian_sparsity()
        values = problem.hessians(x)[0]
        assert np.unique(rows * len(x) + cols).size == len(rows)
        hessian = np.zeros((len(x), len(x)))
        hessian[rows, cols] = values
        hessian[cols, rows] = values
        direction = rng.normal(size=len(x))
        direction /= np.linalg.norm(direction)
        step = 1e-5
        finite = (
            reference_gradient(
                problem, layout, points, x + step * direction, scale, 0.0, 0.0, True
            )
            - reference_gradient(
                problem, layout, points, x - step * direction, scale, 0.0, 0.0, True
            )
        ) / (2 * step)
        np.testing.assert_allclose(hessian @ direction, finite, rtol=4e-5, atol=3e-7)
        # Values and first derivatives remain unambiguous at the C1 knot and
        # at the closed window's domain endpoints; second differences stay off it.
        for probe_u in (-2.0, 0.4, 3.0):
            edge = x.copy()
            edge_parameters = problem.parameters(edge).copy()
            edge_parameters[:, 0] = probe_u
            edge[layout.size :] = edge_parameters.ravel()
            expected = reference_gradient(
                problem, layout, points, edge, scale, 0.0, 0.0, True
            )
            np.testing.assert_allclose(
                problem.gradient(edge)[0], expected, rtol=3e-10, atol=3e-10
            )


def test_alternating_history_keeps_cumulative_seconds_across_cycles():
    """Restarted inner solves must not reset the public animation clock."""
    require_nonlinear()
    initial = nrb.NurbsSurface(
        np.array(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
    )
    u, v = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 7), indexing="ij")
    uv = np.column_stack((u.ravel(), v.ravel()))
    points = np.column_stack(
        (uv, 0.2 * np.sin(np.pi * uv[:, 0]) * uv[:, 1] + 0.08 * uv[:, 0])
    )
    free = np.zeros(initial.P.shape, dtype=bool)
    free[2] = True
    result = nrb.fit_surface(
        points,
        initial,
        correspondence="alternating",
        initial_parameters=uv,
        control_mask=free,
        optimize_weights=False,
        objective_scale=100,
        max_cycles=2,
        record_history=True,
    )
    assert result.diagnostics["outer_iterations"] == 2
    seconds = np.array([row["seconds"] for row in result.history])
    assert np.all(np.diff(seconds) >= 0)
    assert seconds[-1] == pytest.approx(result.timings["solve_seconds"])
    native = result.diagnostics["native_history"]
    assert {row["outer_iteration"] for row in native} == {1, 2}
    assert np.all(np.diff([row["seconds"] for row in native]) >= 0)


@pytest.mark.parametrize("unsupported", ["C0", "degree_zero"])
def test_joint_fit_rejects_structurally_nonsmooth_parameterizations(unsupported):
    """The smooth joint stationarity test needs positive degrees and C1 knots."""
    require_nonlinear()
    if unsupported == "C0":
        degree = 3
        u_knots = np.r_[np.full(4, -2.0), np.full(3, 0.4), np.full(4, 3.0)]
        gu = np.array([u_knots[i + 1 : i + 4].mean() for i in range(7)])
    else:
        degree = 0
        u_knots = np.array([-2.0, 0.4, 3.0])
        gu = np.array([-0.8, 1.7])
    u, v = np.meshgrid(gu, [4.0, 6.0, 8.0], indexing="ij")
    initial = nrb.NurbsSurface(
        np.stack((u, v, 0.04 * np.arange(len(gu))[:, None] ** 2 + 0.02 * v)),
        u_degree=degree,
        v_degree=2,
        u_knots=u_knots,
        v_knots=np.r_[np.full(3, 4.0), np.full(3, 8.0)],
    )
    uv = np.array([[-1.0, 5.0], [0.2, 6.5], [1.1, 7.0]])
    with pytest.raises(ValueError, match="C1|positive degrees"):
        nrb.fit_surface(
            initial.get_value(*uv.T).T,
            initial,
            correspondence="joint",
            initial_parameters=uv,
            max_iterations=1,
        )


def test_tensor_grid_kernels_match_paired_loss_gradient_and_sparse_hessian():
    """Shuffled Cartesian samples retain rational curvature in the fast path."""
    require_nonlinear()
    import jax
    from nurbspy.fitting._common import FitLayout
    from nurbspy.fitting._nonlinear import SurfaceFitProblem

    initial = patch(small=True)
    u, v = np.meshgrid(
        np.linspace(-2.0, 3.0, 6), np.linspace(4.0, 8.0, 7), indexing="ij"
    )
    uv = np.column_stack((u.ravel(), v.ravel()))
    uv = uv[np.random.default_rng(84).permutation(len(uv))]
    points = initial.get_value(*uv.T).T + [0.02, -0.03, 0.04]
    mask = np.ones(initial.P.shape, dtype=bool)
    mask[0, 0, 0] = mask[-1, -1, -1] = False
    with jax.enable_x64():
        layout = FitLayout(initial, optimize_weights=True, control_mask=mask)
        options = dict(
            objective_scale=31.0,
            control_regularization=0.002,
            weight_regularization=0.001,
        )
        paired = SurfaceFitProblem(layout, points, uv, **options)
        tensor = SurfaceFitProblem(layout, points, uv, use_tensor_grid=True, **options)
        assert tensor.use_tensor_grid and not paired.use_tensor_grid
        x = layout.x0 + np.random.default_rng(59).normal(0, 0.02, layout.size)
        np.testing.assert_allclose(
            tensor.fitness(x), paired.fitness(x), rtol=2e-12, atol=1e-13
        )
        np.testing.assert_allclose(
            tensor.gradient(x), paired.gradient(x), rtol=3e-11, atol=1e-12
        )
        for actual, expected in zip(
            tensor.hessian_sparsity(), paired.hessian_sparsity()
        ):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(
            tensor.hessians(x), paired.hessians(x), rtol=3e-10, atol=2e-11
        )


def test_bounded_control_fit_respects_active_coordinate_limits():
    require_nonlinear()
    initial = nrb.NurbsSurface(
        np.array(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
    )
    u, v = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 6), indexing="ij")
    uv = np.column_stack((u.ravel(), v.ravel()))
    points = np.column_stack((uv, np.full(len(uv), 0.7)))
    free = np.zeros(initial.P.shape, dtype=bool)
    free[2] = True
    lower, upper = np.full(initial.P.shape, -np.inf), np.full(initial.P.shape, np.inf)
    lower[2], upper[2] = -0.1, 0.2
    result = nrb.fit_surface(
        points,
        initial,
        parameters=uv,
        control_mask=free,
        control_bounds=(lower, upper),
        objective_scale=100,
        tolerance=1e-8,
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.surface.P[2], 0.2, atol=2e-6)
    np.testing.assert_array_equal(result.surface.P[:2], initial.P[:2])


def test_alternating_fit_accepts_verified_one_sided_minimum_at_c0_knot():
    """At a crease, opposite one-sided slopes replace smooth stationarity."""
    require_nonlinear()
    u, v = np.meshgrid([0.0, 0.5, 1.0], [0.0, 1.0], indexing="ij")
    z = np.broadcast_to(np.array([0.0, 1.0, 0.0])[:, None], u.shape)
    initial = nrb.NurbsSurface(np.stack((u, v, z)), u_degree=1, v_degree=1)
    query = np.array([[0.5, 0.5, 1.2]])
    parameters = np.array([[0.5, 0.5]])
    error = initial.get_value(*parameters.T)[:, 0] - query[0]
    left = initial.get_derivative(np.nextafter(0.5, -np.inf), 0.5, 1, 0)[:, 0]
    right = initial.get_derivative(0.5, 0.5, 1, 0)[:, 0]
    assert left @ error < 0 < right @ error
    free = np.zeros(initial.P.shape, dtype=bool)
    # This free coordinate has zero basis support at the query. The fit block
    # is stationary, while the projection block is a valid nonsmooth minimum.
    free[1, 0, 0] = True
    result = nrb.fit_surface(
        query,
        initial,
        correspondence="alternating",
        initial_parameters=parameters,
        control_mask=free,
        optimize_weights=False,
        max_cycles=1,
    )
    assert result.success, result.message
    assert result.rms == pytest.approx(0.2, abs=1e-12)
    np.testing.assert_allclose(result.parameters, parameters, atol=1e-12)
    assert result.diagnostics["parameter_block_optimality"] <= 1e-8
