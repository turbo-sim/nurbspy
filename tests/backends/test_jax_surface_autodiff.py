"""JAX surface derivatives and transformations against independent references."""

from importlib.util import find_spec

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import BSpline

if find_spec("jax") is None:
    pytest.skip(
        "Install nurbspy[jax] for surface autodiff checks.", allow_module_level=True
    )

import jax
import jax.numpy as jnp
import nurbspy.jax as nrb
from tests.cases import (
    rational_surface_data,
    surface_generator_case,
    paraboloid_control_points,
)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_surface_parameter_derivatives(mode):
    """Differentiate values twice, including mixed partials and clamped boundaries."""
    surface = nrb.NurbsSurface(**rational_surface_data())
    samples = jnp.array([[0.0, 1.0], [0.17, 0.29], [0.71, 0.83], [1.0, 0.0]])

    def value(uv):
        """Expose one parameter pair for automatic Jacobians and Hessians."""
        return surface.get_value(uv[0], uv[1])[:, 0]

    first = jax.jit(jax.vmap(mode(value)))(samples)
    second = jax.jit(jax.vmap(mode(mode(value))))(samples)
    for axis, orders in enumerate([(1, 0), (0, 1)]):
        assert_allclose(
            first[:, :, axis].T,
            surface.get_derivative(*samples.T, *orders),
            rtol=1e-11,
            atol=1e-10,
        )
    for a, b, orders in [
        (0, 0, (2, 0)),
        (0, 1, (1, 1)),
        (1, 0, (1, 1)),
        (1, 1, (0, 2)),
    ]:
        assert_allclose(
            second[:, :, a, b].T,
            surface.get_derivative(*samples.T, *orders),
            rtol=1e-10,
            atol=1e-9,
        )


@pytest.mark.parametrize("argument", ["control_points", "weights"])
def test_surface_shape_gradients(argument):
    """Compare both AD modes with independent SciPy tensor-basis sensitivities."""
    data = rational_surface_data()
    points, weights = data["control_points"], data["weights"]
    u, v = np.array([0.13, 0.39, 0.77]), np.array([0.21, 0.58, 0.89])
    bu = BSpline(data["u_knots"], np.eye(weights.shape[0]), data["u_degree"])(u).T
    bv = BSpline(data["v_knots"], np.eye(weights.shape[1]), data["v_degree"])(v).T
    basis = np.einsum("ik,jk->ijk", bu, bv)
    denominator = np.einsum("ij,ijk->k", weights, basis)
    reference = np.einsum("dij,ijk->dk", points * weights, basis) / denominator
    direction = np.linspace(-0.2, 0.3, data[argument].size).reshape(
        data[argument].shape
    )
    if argument == "control_points":
        derivative = np.einsum("dij,ijk->dk", direction * weights, basis) / denominator
    else:
        derivative = (
            np.einsum("dij,ijk->dk", points * direction, basis)
            - reference * np.einsum("ij,ijk->k", direction, basis)
        ) / denominator

    def loss(value):
        """Form a scalar geometry objective with one variable family exposed."""
        inputs = dict(data, **{argument: value})
        return jnp.sum(nrb.NurbsSurface(**inputs).get_value(u, v) ** 2)

    expected = np.sum(2 * reference * derivative)
    _, tangent = jax.jit(lambda x, dx: jax.jvp(loss, (x,), (dx,)))(
        jnp.asarray(data[argument]), jnp.asarray(direction)
    )
    reverse = jnp.sum(jax.jit(jax.grad(loss))(jnp.asarray(data[argument])) * direction)
    assert_allclose([tangent, reverse], expected, rtol=1e-11, atol=1e-11)


def test_surface_jit_and_batched_geometries():
    """Batch whole surface objects without losing array-valued geometry inputs."""
    data = rational_surface_data()
    surface = nrb.NurbsSurface(**data)
    u, v = jnp.array([0.17, 0.67]), jnp.array([0.31, 0.83])

    def evaluate(points):
        """Build and evaluate a surface inside the compiled batch."""
        return nrb.NurbsSurface(**dict(data, control_points=points)).get_value(u, v)

    result = jax.jit(jax.vmap(evaluate))(jnp.stack([surface.P, 2 * surface.P + 0.3]))
    expected = jnp.stack([surface.get_value(u, v), 2 * surface.get_value(u, v) + 0.3])
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kind", ["bilinear", "ruled", "extruded", "revolution", "coons"]
)
def test_generator_input_gradients(kind):
    """Differentiate generator geometry in both modes against NumPy finite differences."""
    import nurbspy.numpy as numpy_backend

    parameters = np.array([1.2, 0.8, 0.4, 1.1])
    u, v = np.array([0.17, 0.61]), np.array([0.23, 0.82])

    def loss(values):
        """Expose generator inputs while constructing the surface inside JIT."""
        patch = surface_generator_case(nrb, kind, values, jnp)
        return jnp.sum(patch.get_value(u, v) ** 2)

    def reference(values):
        """Use the original NumPy generator for the finite-difference reference."""
        return np.sum(
            surface_generator_case(numpy_backend, kind, values).get_value(u, v) ** 2
        )

    step = 1e-5
    expected = [
        (reference(parameters + step * d) - reference(parameters - step * d))
        / (2 * step)
        for d in np.eye(4)
    ]
    assert_allclose(
        jax.jit(jax.jacfwd(loss))(jnp.asarray(parameters)),
        expected,
        rtol=2e-7,
        atol=2e-8,
    )
    assert_allclose(
        jax.jit(jax.jacrev(loss))(jnp.asarray(parameters)),
        expected,
        rtol=2e-7,
        atol=2e-8,
    )


def test_normals_and_curvature_shape_gradients():
    """Differentiate a paraboloid's regular geometry against exact formulas."""
    base = jnp.asarray(paraboloid_control_points())
    u, v = 0.31, 0.67

    def geometry(a):
        """Return a normal and both curvatures as functions of the paraboloid scale."""
        points = (
            base * jnp.stack((jnp.asarray(1.0), jnp.asarray(1.0), a))[:, None, None]
        )
        surface = nrb.NurbsSurface(points)
        normal = surface.get_normals(u, v)[:, 0]
        mean, gaussian = surface.get_curvature(u, v)
        return jnp.concatenate((normal, mean, gaussian))

    def reference(a):
        """Evaluate the classical graph-surface formulas independently."""
        radius_squared = u * u + v * v
        metric = 1 + 4 * a * a * radius_squared
        normal = np.array([-2 * a * u, -2 * a * v, 1.0]) / np.sqrt(metric)
        mean = (2 * a + 4 * a**3 * radius_squared) / metric**1.5
        gaussian = 4 * a * a / metric**2
        return np.r_[normal, mean, gaussian]

    scale, step = 0.2, 1e-5
    expected = (reference(scale + step) - reference(scale - step)) / (2 * step)
    assert_allclose(geometry(scale), reference(scale), rtol=1e-12, atol=1e-12)
    assert_allclose(
        jax.jit(jax.jacfwd(geometry))(jnp.asarray(scale)),
        expected,
        rtol=1e-8,
        atol=1e-9,
    )
    assert_allclose(
        jax.jit(jax.jacrev(geometry))(jnp.asarray(scale)),
        expected,
        rtol=1e-8,
        atol=1e-9,
    )


@pytest.mark.parametrize("axis", ["u", "v"])
def test_isocurve_parameter_gradients(axis):
    """A section's fixed parameter retains the transverse surface sensitivity."""
    surface = nrb.NurbsSurface(**rational_surface_data())
    samples, fixed = jnp.array([0.17, 0.69]), jnp.asarray(0.37)

    def section(parameter):
        """Extract and evaluate a curve inside JIT and differentiation."""
        curve = (
            surface.get_isocurve_u(parameter)
            if axis == "u"
            else surface.get_isocurve_v(parameter)
        )
        return curve.get_value(samples)

    fixed_samples = jnp.full_like(samples, fixed)
    expected = (
        surface.get_derivative(fixed_samples, samples, 1, 0)
        if axis == "u"
        else surface.get_derivative(samples, fixed_samples, 0, 1)
    )
    assert_allclose(
        jax.jit(jax.jacfwd(section))(fixed), expected, rtol=1e-11, atol=1e-11
    )
    assert_allclose(
        jax.jit(jax.jacrev(section))(fixed), expected, rtol=1e-11, atol=1e-11
    )


def test_attachment_and_offset_gradients():
    """Compiled operations retain shape and signed-offset sensitivities."""
    points = jnp.array(
        [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    plane = nrb.NurbsSurface(points)

    def attached_value(height):
        """Translate the second patch while evaluating inside its half-domain."""
        second = nrb.NurbsSurface(
            points
            + jnp.stack((jnp.asarray(1.0), jnp.asarray(0.0), height))[:, None, None]
        )
        return plane.attach_nurbs_udir(second).get_value(0.73, 0.39)[:, 0]

    def offset_value(distance):
        """Evaluate the sampled offset construction inside differentiation."""
        return nrb.make_offset_surface(plane, distance).get_value(0.31, 0.67)[:, 0]

    for mode in (jax.jacfwd, jax.jacrev):
        assert_allclose(
            jax.jit(mode(attached_value))(jnp.asarray(0.2)), [0.0, 0.0, 1.0], atol=1e-11
        )
        assert_allclose(
            jax.jit(mode(offset_value))(jnp.asarray(0.2)), [0.0, 0.0, -1.0], atol=1e-11
        )


def test_repeated_knots_and_interior_knot_gradients():
    """Repeated-knot evaluations and a movable simple knot retain finite AD."""
    data = rational_surface_data()
    # Five quadratic u-control points permit one repeated interior knot.
    repeated = dict(
        data, u_degree=2, u_knots=jnp.array([0.0, 0.0, 0.0, 0.4, 0.4, 1.0, 1.0, 1.0])
    )
    surface = nrb.NurbsSurface(**repeated)
    samples = jnp.array([[0.0, 0.2], [0.17, 0.4], [0.4, 0.6], [0.73, 0.8], [1.0, 0.9]])

    def value(uv):
        """Expose span-selected derivatives, including the repeated knot itself."""
        return surface.get_value(*uv)[:, 0]

    for mode in (jax.jacfwd, jax.jacrev):
        actual = jax.jit(jax.vmap(mode(value)))(samples)
        assert np.isfinite(actual).all()
        # At the repeated knot these are the implementation's right-sided
        # span derivatives; the two-sided derivative need not exist.
        assert_allclose(
            actual[:, :, 0].T, surface.get_derivative(*samples.T, 1, 0), atol=1e-10
        )

    u, v = np.array([0.17, 0.79]), np.array([0.31, 0.67])

    def loss(knot):
        """Move a simple interior knot without changing its order or multiplicity."""
        knots = jnp.asarray(data["u_knots"]).at[4].set(knot)
        return jnp.sum(
            nrb.NurbsSurface(**dict(data, u_knots=knots)).get_value(u, v) ** 2
        )

    def reference(knot):
        """Compute the same rational values using SciPy tensor-product bases."""
        knots = np.array(data["u_knots"])
        knots[4] = knot
        bu = BSpline(knots, np.eye(5), 3)(u).T
        bv = BSpline(data["v_knots"], np.eye(4), 3)(v).T
        tensor = np.einsum("ik,jk->ijk", bu, bv)
        values = np.einsum(
            "dij,ijk->dk", data["control_points"] * data["weights"], tensor
        ) / np.einsum("ij,ijk->k", data["weights"], tensor)
        return np.sum(values**2)

    step = 1e-5
    expected = (reference(0.5 + step) - reference(0.5 - step)) / (2 * step)
    assert_allclose(
        jax.jit(jax.jacfwd(loss))(jnp.asarray(0.5)), expected, rtol=1e-7, atol=1e-8
    )
    assert_allclose(
        jax.jit(jax.jacrev(loss))(jnp.asarray(0.5)), expected, rtol=1e-7, atol=1e-8
    )
