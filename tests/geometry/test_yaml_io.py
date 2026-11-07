"""Geometry persistence preserves mathematical state across both backends."""
from io import StringIO

import numpy as np
import pytest
import yaml


def geometry(backend, surface=False, dtype=np.float64, rational=True):
    P = np.array([[0, 1, 2, 3], [0, 2, -1, 0]], dtype=dtype)
    W = np.array([1, .5, 1.5, 1], dtype=np.float64) if rational else None
    if np.issubdtype(dtype, np.complexfloating):
        P[1, 1] += 1e-20j
    if surface:
        P = np.stack((P, P + 1, P + 2), axis=-1)
        W = np.repeat(W[:, None], 3, axis=1) if W is not None else None
        return backend.NurbsSurface(P, weights=W, u_degree=2, v_degree=1)
    return backend.NurbsCurve(P, weights=W, degree=2)


@pytest.mark.parametrize("surface", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
@pytest.mark.parametrize("rational", [False, True])
def test_round_trip_preserves_arrays_values_and_independence(curve_backend, surface, dtype, rational):
    original = geometry(curve_backend, surface, dtype, rational)
    text = original.to_yaml()
    assert "!!python" not in text
    restored = type(original).from_yaml(StringIO(text))
    for field in (("P", "W", "U", "V") if surface else ("P", "W", "U")):
        expected, actual = np.asarray(getattr(original, field)), np.asarray(getattr(restored, field))
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert not np.shares_memory(actual, expected)
    assert restored.p == original.p
    if surface:
        assert restored.q == original.q
        np.testing.assert_allclose(restored.get_value(np.array([.1, .5]), np.array([.2, .8])), original.get_value(np.array([.1, .5]), np.array([.2, .8])))
    else:
        np.testing.assert_allclose(restored.get_derivative(np.array([.2, .7]), 1), original.get_derivative(np.array([.2, .7]), 1))


@pytest.mark.parametrize("surface", [False, True])
def test_path_and_stream_io(curve_backend, tmp_path, surface):
    original = geometry(curve_backend, surface)
    path = tmp_path / "geometry.yaml"
    assert original.to_yaml(path) == path
    np.testing.assert_array_equal(type(original).from_yaml(str(path)).P, original.P)
    stream = StringIO()
    assert original.to_yaml(stream) is None
    stream.seek(0)
    np.testing.assert_array_equal(type(original).from_yaml(stream).W, original.W)


def test_cross_backend_loading_and_jax_gradient(numpy_backend, jax_backend):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    original = geometry(numpy_backend)
    restored = jax_backend.NurbsCurve.from_yaml(StringIO(original.to_yaml()))
    assert isinstance(restored.P, jax.Array)
    gradient = eqx.filter_grad(lambda model: jnp.sum(model.get_value(.37)))(restored)
    assert np.all(np.isfinite(gradient.P))
    restored_numpy = numpy_backend.NurbsCurve.from_yaml(StringIO(restored.to_yaml()))
    assert isinstance(restored_numpy.P, np.ndarray)
    np.testing.assert_array_equal(restored_numpy.P, original.P)


def test_nonstandard_domain_is_preserved(numpy_backend):
    original = geometry(numpy_backend)
    original.U = 2 + 3 * original.U
    restored = type(original).from_yaml(StringIO(original.to_yaml()))
    np.testing.assert_allclose(restored.get_value(np.array([2., 3., 5.])), original.get_value(np.array([2., 3., 5.])))


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(version=99),
    lambda r: r.update(version=True),
    lambda r: r.update(type="NurbsSurface"),
    lambda r: r.update(extra="unknown"),
    lambda r: r.update(parameters=[]),
    lambda r: r["parameters"].update(degree=True),
    lambda r: r["parameters"].update(degree=9),
    lambda r: r["parameters"].update(degree=1.5),
    lambda r: r["parameters"].update(knots={"dtype": "float64", "values": [1, 0]}),
    lambda r: r["parameters"].update(weights={"dtype": "float64", "values": [1]}),
    lambda r: r["parameters"].update(weights={"dtype": "object", "values": [1, 1, 1, 1]}),
    lambda r: r["parameters"]["control_points"].update(values=[[0, float("nan"), 2, 3], [0, 1, 0, 0]]),
    lambda r: r["parameters"]["control_points"].update(values=[[0, 1], [2]]),
    lambda r: r["parameters"]["control_points"].update(dtype="int8", values=[[0, 128, 2, 3], [0, 1, 0, 0]]),
    lambda r: r["parameters"]["control_points"].update(dtype="int64", values=[[0, .5, 2, 3], [0, 1, 0, 0]]),
])
def test_reject_malformed_documents(numpy_backend, mutation):
    original = geometry(numpy_backend)
    record = yaml.safe_load(original.to_yaml())
    mutation(record)
    with pytest.raises(ValueError):
        type(original).from_yaml(StringIO(yaml.safe_dump(record)))


@pytest.mark.parametrize("text", ["", "[]", "!!python/object/apply:os.system ['never execute']", "{invalid"])
def test_only_safe_versioned_yaml_is_loaded(numpy_backend, text):
    with pytest.raises(ValueError):
        numpy_backend.NurbsCurve.from_yaml(StringIO(text))


def test_nonfinite_geometry_is_not_written(numpy_backend, tmp_path):
    original = geometry(numpy_backend)
    original.P[0, 0] = np.inf
    path = tmp_path / "bad.yaml"
    with pytest.raises(ValueError, match="finite"):
        original.to_yaml(path)
    assert not path.exists()


@pytest.mark.parametrize("dimension", [1, 4])
def test_arbitrary_dimension_and_complex_weights(curve_backend, dimension):
    P = np.arange(dimension * 4, dtype=float).reshape(dimension, 4)
    W = np.array([1, .5 + 1e-20j, 2, 1])
    original = curve_backend.NurbsCurve(P.astype(complex), W, degree=2)
    restored = type(original).from_yaml(StringIO(original.to_yaml()))
    np.testing.assert_array_equal(restored.P, original.P)
    np.testing.assert_array_equal(restored.W, original.W)
    np.testing.assert_allclose(restored.get_value(.3), original.get_value(.3))
