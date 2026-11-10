"""Run five related fitting formulations and archive authoritative results."""

from pathlib import Path
from time import perf_counter
import hashlib
import importlib.metadata
import json
import os
import platform
import webbrowser
import numpy as np
import nurbspy as nrb
from _experiment import GRIDS, SAMPLES, data, flat, linear, metrics, target
from _report import export_report

CASES = {
    "linear": (1, "01_linear"),
    "rational": (2, "02_known_parameters"),
    "joint": (3, "03_joint_parameters"),
    "alternating": (4, "04_alternating_projection"),
    "constrained": (5, "05_constrained"),
}


def geometric_reference(fitted, baseline, observations, correspondence):
    """Score local geometric distances using learned correspondences as initial guesses.

    Nearest observed XYZ positions supply their fitted u/v labels to new reference
    points. This does not use generating labels and avoids discarding the learned
    parameterization when evaluating a strongly reparameterized coarse surface.
    """
    from scipy.spatial import cKDTree
    from _projection import SurfaceProjector
    import jax

    axis = np.linspace(0, 1, 65)
    u, v = np.meshgrid(axis, axis, indexing="ij")
    points = target(u, v).reshape(-1, 3)
    indices = cKDTree(observations["points"]).query(points)[1]
    parameters = correspondence[indices]
    begun = perf_counter()
    with jax.enable_x64():
        projector = SurfaceProjector(fitted, points, initial_parameters=parameters)
        _, distances, info = projector.project(fitted, parameters)
        _, base_distances, base_info = projector.project(baseline, points[:, :2])
    return dict(
        clean_rms=float(np.sqrt(np.mean(distances**2))),
        clean_max=float(distances.max()),
        reference_projection_seconds=perf_counter() - begun,
        reference_projection_success=info["success"],
        bspline_geometric_reference_rms=float(np.sqrt(np.mean(base_distances**2))),
        baseline_projection_success=base_info["success"],
        reference_initialization="NurbsPy surface sampling plus learned correspondence samples",
    )


def run(
    *,
    method="linear",
    grid_sizes=GRIDS,
    samples=SAMPLES,
    sigma=0.0,
    max_iterations=1000,
    tolerance=1e-8,
    max_cycles=100,
    output=None,
    show=False,
):
    """Start each grid flat, save fit/history/timings, then export the HTML report."""
    if method not in CASES:
        raise ValueError("Unknown fitting study")
    if any(n not in GRIDS for n in grid_sizes):
        raise ValueError("Supported control grids are 4, 8, 16 and 32")
    if samples < max(grid_sizes):
        raise ValueError("Use at least as many observations as controls per direction")
    study, folder = CASES[method]
    output = (
        Path(output)
        if output is not None
        else Path(__file__).resolve().parent / "output" / folder
    )
    output.mkdir(parents=True, exist_ok=True)
    observations = data(samples, sigma)
    np.savez_compressed(output / "observations.npz", **observations)
    records = []
    for count in grid_sizes:
        print(
            f"\nCase {study:02d} / {method} / {count}x{count}: flat start", flush=True
        )
        initial = flat(count)
        initial.to_yaml(output / f"initial_{count:02d}.yaml")
        correspondence = None
        imported = perf_counter()
        if method == "linear":
            fitted, info = linear(initial, observations)
            history = [
                dict(
                    iteration=0,
                    event_kind="initial",
                    surface=initial,
                    training_rms=metrics(initial, observations)["training_rms"],
                ),
                dict(
                    iteration=0,
                    event_kind="solution",
                    surface=fitted,
                    training_rms=metrics(fitted, observations)["training_rms"],
                ),
            ]
        else:
            if method == "rational":
                from _nonlinear import nonlinear as solver
            elif method == "joint":
                from _joint import joint as solver
            elif method == "alternating":
                from _alternating import alternating as solver
            else:
                from _constrained import constrained as solver
            module_seconds = perf_counter() - imported
            options = dict(max_iterations=max_iterations, tolerance=tolerance)
            if method == "alternating":
                options["max_cycles"] = max_cycles
            result = solver(initial, observations, **options)
            fitted, info, history = result[:3]
            if len(result) == 4:
                correspondence = result[3]
            info["setup_seconds"] += module_seconds
            (output / f"native_history_{count:02d}.json").write_text(
                json.dumps(info.pop("native_history"), indent=2) + "\n",
                encoding="utf-8",
            )
        fitted.to_yaml(output / f"fit_{count:02d}.yaml")
        archive = dict(
            control_points=np.stack([f["surface"].P for f in history]),
            weights=np.stack([f["surface"].W for f in history]),
            iteration=[f["iteration"] for f in history],
            training_rms=[f["training_rms"] for f in history],
            seconds=[f.get("seconds", 0.0) for f in history],
            event_kind=[f.get("event_kind", "native_observation") for f in history],
            phase=[
                f.get("phase", "joint_fit" if method == "joint" else "surface_fit")
                for f in history
            ],
        )
        for key in ("area", "constraint_violation", "outer_iteration", "local_pass"):
            archive[key] = [f.get(key, np.nan) for f in history]
        if correspondence is not None:
            archive["parameters"] = np.stack([f["parameters"] for f in history])
            np.savez_compressed(
                output / f"correspondences_{count:02d}.npz", parameters=correspondence
            )
        np.savez_compressed(output / f"iterations_{count:02d}.npz", **archive)
        row = dict(
            grid=int(count),
            surface=fitted,
            history=history,
            **info,
            **metrics(fitted, observations),
        )
        if method != "linear":
            baseline, _ = linear(initial, observations)
            row["bspline_reference_rms"] = metrics(baseline, observations)["clean_rms"]
        if correspondence is not None:
            row["parametric_reference_rms"] = row["clean_rms"]
            row["training_rms"] = history[-1]["training_rms"]
        (output / f"fit_metrics_{count:02d}.json").write_text(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("surface", "history")},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if correspondence is not None:
            row.update(
                geometric_reference(fitted, baseline, observations, correspondence)
            )
        records.append(row)
        print(
            f"  RMS train {row['training_rms']:.7g}; reference {row['clean_rms']:.7g}; "
            f"{row['variables']} unknowns; converged={row['success']}",
            flush=True,
        )
        if method != "linear":
            print(
                f"  compile {row['compile_seconds']:.3f} s; setup {row['setup_seconds']:.3f} s; "
                f"warm solve {row['solve_seconds']:.3f} s; {row['iterations']} IPM iterations",
                flush=True,
            )
        (output / "metrics.json").write_text(
            json.dumps(
                [
                    {k: v for k, v in r.items() if k not in ("surface", "history")}
                    for r in records
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    packages = ("nurbspy", "numpy", "scipy") + (
        ("jax", "pysolver_view", "pounce-solver") if study > 1 else ()
    )
    experiment = dict(
        target="Six analytic Gaussian hills and valleys",
        method=method,
        fitting_api=(
            "nurbspy.fit_surface" if study < 5 else "Custom constrained Pounce problem"
        ),
        study=study,
        grid_sizes=list(grid_sizes),
        samples_per_axis=samples,
        noise_sigma=sigma,
        seed=observations["seed"],
        initial_guess="Flat z=0, Greville x/y controls, unit weights; independent for every grid",
        fixed_parameters=method not in ("joint", "alternating"),
        alternating_max_cycles=max_cycles if study == 4 else None,
        joint_max_passes=8 if study == 3 else None,
        point_projection=(
            dict(
                backend="NurbsPy JAX prepared surface projection (existing public API)",
                samples_per_span=8,
                maxiter=200,
                gtol=1e-9,
                ftol=1e-12,
                xtol=1e-12,
            )
            if study in (3, 4)
            else None
        ),
        weight_bounds=[1, 1] if study == 1 else ([0.5, 2] if study == 5 else [0.1, 10]),
        weight_gauge=(
            "All weights=1"
            if study == 1
            else ("Boundary weights=1" if study == 5 else "W00=1")
        ),
        nonlinear_solver=(
            None
            if study == 1
            else dict(
                backend="Pounce interior point",
                hessian="Exact sparse Hessian; full constraint curvature in 05",
                tolerance=tolerance,
                native_tolerance=(
                    min(tolerance * 0.1, 1e-9) if study == 3 else tolerance
                ),
                max_iterations=max_iterations,
                separate_kkt_checks=False,
            )
        ),
        timing="Setup includes compilation and kernel warmup. Warm solve excludes geometry/report export "
        "and final audits. Case 04 sums warm surface-fit IPM and existing library projection. "
        "Reference projections are postprocessing, reported separately.",
        environment=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            processor=platform.processor(),
            packages={name: importlib.metadata.version(name) for name in packages},
            threads={
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "RAYON_NUM_THREADS",
                )
            },
        ),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).resolve().parent.glob("*.py")
        },
        fitting_source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(nrb.__file__).resolve().parent / "fitting").glob("*.py")
        },
    )
    (output / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    report = export_report(
        records, observations, output, nonlinear=study > 1, study=study
    )
    if show:
        webbrowser.open(report.resolve().as_uri())
    return records
