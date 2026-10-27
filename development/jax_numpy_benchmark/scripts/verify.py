"""Fast study self-checks and saved-result integrity checks, without retiming."""
import json
import hashlib
import sys

sys.dont_write_bytecode = True
import numpy as np
from cases import curve_data, surface_data
from references import Curve, Surface, basis
from workloads import registry, surface_area
from benchmark import compare
from paths import RESULTS, ROOT


def main():
    P, W, U, u, p = curve_data("circle", 17)
    c = Curve(P, W, U, p)
    np.testing.assert_allclose(np.linalg.norm(c.get_value(u)[:2], axis=0), 1, atol=2e-14)
    np.testing.assert_allclose(c.get_curvature(u), 1, atol=2e-14)
    np.testing.assert_allclose(c.get_arclength(), np.pi/2, atol=2e-12)
    np.testing.assert_array_equal(basis(U, p, u, p+1), np.zeros((3, 17)))
    for kind in ("plane", "paraboloid"):
        P, W, U, V, a, b, p, q = surface_data(kind, 7)
        s = Surface(P, W, U, V, p, q)
        if kind == "plane":
            np.testing.assert_allclose(surface_area(s, np), 1, atol=2e-14)
            np.testing.assert_allclose(s.get_value(a, b), np.array([a, b, np.zeros_like(a)]), atol=2e-14)
            H, K = s.get_curvature(a, b)
            np.testing.assert_allclose(H, 0, atol=2e-14)
            np.testing.assert_allclose(K, 0, atol=2e-14)
        else:
            np.testing.assert_allclose(s.get_value(a, b), np.array([a, b, a*a+b*b]), atol=2e-14)
            H, K = s.get_curvature(a, b)
            d = 1+4*a*a+4*b*b
            np.testing.assert_allclose(H, (2+4*a*a+4*b*b)/d**1.5, atol=2e-14)
            np.testing.assert_allclose(K, 4/d**2, atol=2e-14)
    row = dict(group="curve")
    assert not compare(np.array([np.nan]), np.array([np.nan]), row)["passed"]
    assert not compare(np.array([1.]), np.array([[1.]]), row)["passed"]
    assert compare(np.array([1.]), np.array([1.+1e-12]), row)["passed"]
    expected = registry()
    environment = json.loads((RESULTS/"environment.json").read_text())
    for relative, digest in environment["source_sha256"].items():
        path = ROOT/relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Source changed after measurement: {relative}"
    for profile in ("numpy", "numba", "jax"):
        data = json.loads((RESULTS/f"{profile}.json").read_text())
        if profile == "jax":
            assert data["context"]["x64"]
        else:
            assert data["context"].get("numba_acceleration") == (profile == "numba")
        rows = data["records"]
        wanted = {r["id"] for r in expected if profile == "jax" or not r["only_jax"]}
        assert len(rows) == len(wanted) == len({r["id"] for r in rows}), profile
        assert {r["id"] for r in rows} == wanted
        with np.load(RESULTS/f"{profile}.npz") as raw:
            for r in rows:
                if r["status"] != "ok":
                    raise AssertionError(f"Unexpected API/compiler failure after the basis fix: {r['id']}: {r.get('error')}")
                    continue
                assert r["warm_median_s"] > 0
                assert len(r["warm_samples_s"]) >= 3
                assert np.isclose(r["warm_median_s"], np.median(r["warm_samples_s"]))
                assert r["cold_s"] >= r["first_s"] > 0
                key = r["id"].replace(".", "_")
                for i, shape in enumerate(r["shapes"]):
                    assert raw[f"{key}__out_{i}"].shape == tuple(shape)
    summary = json.loads((RESULTS/"summary.json").read_text())
    assert len(summary) == len(expected)
    reviews = [r["id"] for r in summary if r["status"] in ("ERROR", "REVIEW")]
    print(f"Reference self-checks and result integrity passed: {len(expected)} workloads; {len(reviews)} flagged for review.")


if __name__ == "__main__":
    main()
