"""Linear control-point least squares with fixed parameters and rational weights."""

from time import perf_counter
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import lsmr
from ._common import basis_matrix, control_penalty


def solve_linear(
    layout,
    points,
    parameters,
    *,
    control_regularization=0.0,
    tolerance=1e-10,
    max_iterations=1000,
    print_convergence=False,
):
    """Use a tensor solve when applicable and sparse paired rows otherwise."""
    begun = perf_counter()
    regularization = control_penalty(control_regularization, layout)
    matrix = basis_matrix(layout.initial, parameters, layout.weights0)
    residual = points - matrix @ layout.controls0
    u, iu = np.unique(parameters[:, 0], return_inverse=True)
    v, iv = np.unique(parameters[:, 1], return_inverse=True)
    tensor = (
        len(u) * len(v) == len(parameters)
        and len(np.unique(iu * len(v) + iv)) == len(parameters)
        and layout.ncontrol_variables == layout.count * layout.ndim
        and not regularization.any()
        and np.all(layout.weights0 == layout.weights0[0])
    )
    a = (
        BSpline.design_matrix(
            u, np.asarray(layout.initial.U), int(layout.initial.p)
        ).toarray()
        if tensor
        else None
    )
    b = (
        BSpline.design_matrix(
            v, np.asarray(layout.initial.V), int(layout.initial.q)
        ).toarray()
        if tensor
        else None
    )
    setup = perf_counter() - begun
    begun = perf_counter()
    controls = layout.controls0.copy()
    condition = None
    linear_iterations, optimality, ranks = 0, 0.0, []
    success = True
    if tensor:
        grid = np.empty((len(u), len(v), layout.ndim))
        grid[iu, iv] = residual
        first, _, rank_u, singular_u = np.linalg.lstsq(
            a, grid.reshape(len(u), -1), rcond=None
        )
        first = first.reshape(layout.shape[0], len(v), layout.ndim).transpose(1, 0, 2)
        second, _, rank_v, singular_v = np.linalg.lstsq(
            b, first.reshape(len(v), -1), rcond=None
        )
        controls += (
            second.reshape(layout.shape[1], layout.shape[0], layout.ndim)
            .transpose(1, 0, 2)
            .reshape(layout.count, layout.ndim)
        )
        ranks = [int(rank_u * rank_v)]
        condition = float(
            (singular_u[0] / singular_u[-1]) * (singular_v[0] / singular_v[-1])
        )
        method = "Direct tensor-product linear least squares"
    else:
        method = "Linear least squares at paired parameters"
        for component in range(layout.ndim):
            free = (
                layout.control_indices[
                    layout.control_indices % layout.ndim == component
                ]
                // layout.ndim
            )
            if not len(free):
                continue
            design = matrix[:, free] / np.sqrt(len(points))
            rhs = residual[:, component] / np.sqrt(len(points))
            penalty = regularization[free, component] / layout.count
            if penalty.any():
                design = vstack((design, diags(np.sqrt(penalty))), format="csr")
                rhs = np.r_[rhs, np.zeros(len(free))]
            if len(free) <= 256:
                delta, _, rank, _ = np.linalg.lstsq(design.toarray(), rhs, rcond=None)
                ranks.append(int(rank))
            else:
                answer = lsmr(
                    design, rhs, atol=tolerance, btol=tolerance, maxiter=max_iterations
                )
                delta = answer[0]
                linear_iterations = max(linear_iterations, int(answer[2]))
                success &= answer[1] in (0, 1, 2, 4, 5)
            controls[free, component] += delta
            optimality = max(
                optimality, float(np.max(abs(design.T @ (design @ delta - rhs))))
            )
    x = controls.ravel()[layout.control_indices]
    elapsed = perf_counter() - begun
    audit_started = perf_counter()
    gradient = matrix.T @ (matrix @ controls - points) / len(points)
    gradient += regularization * (controls - layout.controls0) / layout.count
    optimality = float(
        np.max(abs(gradient.ravel()[layout.control_indices]), initial=0.0)
    )
    audit_seconds = perf_counter() - audit_started
    message = (
        method
        if success
        else "Linear least-squares iteration/conditioning limit reached"
    )
    if print_convergence:
        print(f"{method}: {len(points)} points, {layout.size} variables", flush=True)
        print(
            f"  Linear optimality {optimality:.3e}; solve {elapsed:.6f} s; success={success}",
            flush=True,
        )
    return x, dict(
        success=bool(success),
        message=message,
        status=0 if success else 1,
        iterations=0,
        linear_iterations=linear_iterations,
        nfev=1,
        njev=0,
        nhev=0,
        setup_seconds=setup,
        compile_seconds=0.0,
        warmup_seconds=0.0,
        solve_seconds=elapsed,
        audit_seconds=audit_seconds,
        kkt_seconds=0.0,
        optimality=optimality,
        rank=ranks[0] if tensor else ranks,
        condition_number=condition,
        solver="linear",
        variables=layout.size,
    )
