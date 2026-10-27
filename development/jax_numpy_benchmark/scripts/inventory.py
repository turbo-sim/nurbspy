"""Audit public source symbols without importing either geometry backend."""
import ast
import sys

sys.dont_write_bytecode = True
from paths import ROOT, STUDY as HERE


def classification(module, owner, name):
    if name.startswith("plot") or name == "rescale_plot":
        return "Display smoke only", "Curve/surface default rendering; individual styling and plot helpers not separately timed."
    if name == "get_normal_2D":
        return "Direct", "normal2d.rotated_tangent; JAX uses an explicit rotated-tangent adapter."
    if name in ("compute_all_basis_polynomials_derivatives", "compute_basis_polynomials_derivatives_jax"):
        return "Not separately timed", "Alternative basis helper; main analytic basis path is compared."
    if module == "nurbs_basis_functions":
        return "Direct", "basis value and derivative workloads."
    if name.startswith("compute_"):
        return "Indirect", "Exercised by public geometry methods; polynomial/rational helper variants not individually timed."
    if owner == "PointToCurveProjectionProblem" or owner == "PointToSurfaceProjectionProblem":
        return "Adapter / indirect", "NumPy projection exercises these objective helpers; JAX compatibility wrapper is not separately timed."
    if name in ("intersect_lines", "project_point_to_line", "make_nurbs_surface", "make_nurbs_circular_arc"):
        return "Construction / indirect", "Called by tested constructors; degenerate helper inputs are outside this study."
    if name == "get_derivatives":
        return "Indirect", "Prepared individual derivative methods exercise the same derivative tables."
    if name == "prepare_surface_projection" or owner == "PreparedSurfaceProjection":
        return "JAX-only", "prepared_projection.batch; scalar adapter not separately timed."
    if name == "project_point":
        return "Indirect", "Batched projection is timed; scalar adapter not separately timed."
    if module == "prepared":
        return "JAX-only", "prepared curve/surface derivative, curvature, value and grid workloads."
    if module == "arclength":
        return "JAX-only / partial", "Length, prepared quadrature and inverse-map workloads; diagnostic return_info/status APIs not separately exercised."
    if name == "project_point_to_curve" or name == "project_point_to_surface" or name == "project_points":
        return "Direct / adapter", "Projection workload uses native batches for JAX and loops over NumPy scalar calls."
    if name.startswith("NurbsSurface") or name in ("NurbsCurve", "CircularArc"):
        return "Direct", "Constructed by deterministic geometry/generator workloads."
    if name in ("merge_nurbs_curves", "make_offset_surface") or name.startswith("attach_nurbs") or name.startswith("get_isocurve"):
        return "Direct", "construction workloads."
    if name.startswith("reparametrize"):
        return "JAX-only", "Inverse-map construction and independent inverse accuracy checks."
    if name.startswith("get_arclength"):
        return "Direct", "Native default and JAX explicit method workloads."
    if name in ("get_value", "get_derivative", "get_tangent", "get_normal", "get_binormal", "get_curvature", "get_torsion", "get_normals"):
        return "Direct", "Curve/surface numerical workloads and independent references."
    return "Not separately exercised", "Listed explicitly rather than assuming coverage."


def write():
    rows = []
    for backend in ("numpy", "jax"):
        for path in sorted((ROOT/f"src/nurbspy/{backend}").glob("*.py")):
            if path.name.startswith("_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                    symbols = [(None, node)]
                    if isinstance(node, ast.ClassDef):
                        symbols += [(node.name, child) for child in node.body if isinstance(child, ast.FunctionDef) and not child.name.startswith("_")]
                    for owner, symbol in symbols:
                        status, note = classification(path.stem, owner, symbol.name)
                        rows.append((backend, f"{path.stem}.{owner+'.' if owner else ''}{symbol.name}", status, note))
    text = "# Public API coverage inventory\n\nGenerated from public top-level symbols and public class methods in backend source modules. Private modules/symbols, imports/re-exports and special methods are omitted. This is an inventory of API coverage, not a claim that all input combinations are tested.\n\n"
    text += "Important limits: no complex-valued sensitivity study, no GPU/TPU timings, no exhaustive malformed-input matrix, no global projection certificate, no individually timed plot styling, and no separate diagnostic-status stress matrix. Surface area and energy workflows are derived calculations. Most domains are [0,1]; shifted curve evaluation/derivative cases use [-2,3].\n\n"
    text += "| Backend | Source symbol | Coverage | Detail |\n| --- | --- | --- | --- |\n"
    text += "\n".join(f"| {b} | `{s}` | {c} | {n} |" for b, s, c, n in rows)+"\n"
    (HERE/"coverage.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    write()
