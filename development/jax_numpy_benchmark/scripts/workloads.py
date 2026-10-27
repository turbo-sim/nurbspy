"""Shared adapters: identical inputs, explicit derived workflows and API gaps."""
import numpy as np
from cases import curve_data, surface_data, generator, gauss, gauss_grid
from references import Curve, Surface, basis


def registry():
    rows = []
    def add(group, op, kind="rational", size=257, host=False, only_jax=False):
        rows.append(dict(id=f"{group}.{op}.{kind}.{size}", group=group, op=op,
                         kind=kind, size=size, host=host, only_jax=only_jax))
    for k in ("rational", "repeated"):
        for op in ("value", "d1", "d2", "d3", "d4"):
            add("basis", op, k)
    for k in ("rational", "polynomial", "planar", "repeated"):
        for op in ("value", "d1", "d2", "d3", "d4", "speed", "tangent", "normal", "binormal", "curvature"):
            add("curve", op, k)
        if k != "planar":
            add("curve", "torsion", k)
    for n in (1, 4097):
        add("curve", "value", size=n)
    for k in ("rational", "circle", "repeated", "line"):
        add("length", "default", k, 1, host=True)
        add("length", "adaptive", k, 1, only_jax=True)
        add("length", "ode", k, 1, only_jax=True)
        add("length", "fixed", k, 1, only_jax=True)
    for k in ("rational", "plane", "paraboloid"):
        for op in ("value", "d10", "d01", "d20", "d11", "d02", "d21", "normals", "curvature", "area"):
            add("surface", op, k, 17)
    for k in ("rational", "line"):
        add("projection_curve", "batch", k, 4)
    for k in ("rational", "plane", "paraboloid"):
        add("projection_surface", "batch", k, 4)
    for op in ("isocurve_u", "isocurve_v", "attach_u", "attach_v", "offset"):
        add("construction", op, size=17, host=True)
    add("construction", "merge", size=257, host=True)
    add("construction", "circle", size=257, host=True)
    for k in ("bilinear", "ruled", "extruded", "revolution", "coons"):
        add("generator", "value", k, 17, host=True)
    for op in ("curve", "surface"):
        add("workflow", op, size=65 if op == "curve" else 9)
        add("sensitivity", op, size=65 if op == "curve" else 9, host=True)
        add("plot", op, size=17, host=True)
    for op in ("curve_value", "curve_d2", "curve_curvature", "surface_value", "surface_d11", "surface_curvature", "surface_grid", "length"):
        add("prepared", op, size=257 if op.startswith("curve") else 17, only_jax=True)
    add("map", "arclength", "circle", 65, host=True, only_jax=True)
    add("map", "coordinate", "rational", 65, host=True, only_jax=True)
    for op in ("normal", "curvature", "torsion"):
        add("degenerate", op, "collapsed", 17)
    for op in ("value", "d1", "d2"):
        add("curve", op, "evolution", 65)
    for op in ("value", "d11", "curvature"):
        add("surface", op, "polynomial", 17)
    add("normal2d", "rotated_tangent", "planar", 257)
    for op in ("curve_weights", "surface_weights"):
        add("sensitivity", op, size=65 if op.startswith("curve") else 9, host=True)
    add("prepared_projection", "batch", size=4, only_jax=True)
    add("map_query", "arclength", "circle", 65, only_jax=True)
    for kind in ("quintic", "constant"):
        for op in ("value", "d1", "d4"):
            add("basis", op, kind, 65)
            add("curve", op, kind, 65)
    for kind in ("shifted", "weight_stress"):
        for op in ("value", "d1", "d2", "curvature"):
            add("curve", op, kind, 257)
    for op in ("value", "d10", "d01", "d20", "d11", "d02", "normals", "curvature"):
        add("surface", op, "repeated", 17)
    add("projection_curve", "batch", size=16)
    add("projection_surface", "batch", size=16)
    for op in ("value", "d1", "d2", "d3"):
        add("basis", op, "nonclamped", 257)
        add("curve", op, "nonclamped", 257)
    for kind in ("circle", "rational", "repeated"):
        add("map_query", "unit_speed", kind, 65, only_jax=True)
    for kind in ("rational", "repeated"):
        add("map_query", "unit_speed_tight", kind, 65, only_jax=True)
    for op in ("curve", "surface"):
        add("sensitivity_projection", op, size=4, host=True)
    return rows


def surface_area(s, xp, order=24):
    """Derived tensor Gauss workflow, not a native surface-area API.

    Cases have one span per direction. Extending cases to multiple spans must
    also split this rule at every nonempty knot span.
    """
    u, v, weights = gauss_grid(order)
    a, b = s.get_derivative(xp.asarray(u), xp.asarray(v), 1, 0), s.get_derivative(xp.asarray(u), xp.asarray(v), 0, 1)
    return xp.sum(xp.linalg.norm(xp.cross(a.T, b.T), axis=-1)*xp.asarray(weights))


def make(row, profile):
    """Return callable, dynamic arguments, independent reference, notes.

    Geometry and sample arrays are arguments, so JAX cannot constant-fold the
    entire workload. Static topology and prepared sample tables are intentional.
    """
    if profile == "jax":
        import jax
        import jax.numpy as xp
        import nurbspy.jax as backend
    else:
        xp = np
        import nurbspy.numpy as backend
    g, op, kind, n = (row[x] for x in ("group", "op", "kind", "size"))
    P, W, U, u, p = curve_data(kind if g not in ("generator",) else "rational", n)
    c = lambda P, W, U: backend.NurbsCurve(P, W, p, U)
    rc = Curve(P, W, U, p)
    convert = lambda *xs: tuple(xp.asarray(x) for x in xs)
    notes = ""
    if g == "normal2d":
        def f(P, W, U, u):
            obj = c(P, W, U)
            if profile != "jax":
                return obj.get_normal_2D(u)
            tangent = obj.get_tangent(u)
            return xp.stack((-tangent[1], tangent[0]))
        d = rc.get_tangent(u)
        return f, convert(P, W, U, u), np.stack((-d[1], d[0])), "NumPy get_normal_2D versus explicit rotated JAX tangent; derived adapter because JAX lacks this named API."
    if g == "basis":
        order = 0 if op == "value" else int(op[1:])
        def f(U, u):
            if order == 0:
                return backend.compute_basis_polynomials(P.shape[1]-1, p, U, u)
            return backend.compute_basis_polynomials_derivatives(P.shape[1]-1, p, U, u, order)
        return f, convert(U, u), basis(U, p, u, order), notes
    if g in ("curve", "degenerate"):
        def evaluate(obj, x, namespace):
            if op.startswith("d"):
                return obj.get_derivative(x, int(op[1:]))
            if op == "speed":
                return namespace.linalg.norm(obj.get_derivative(x, 1), axis=0)
            return getattr(obj, "get_"+op)(x)
        f = lambda P, W, U, u: evaluate(c(P, W, U), u, xp)
        with np.errstate(all="ignore"):
            ref = evaluate(rc, u, np)
        return f, convert(P, W, U, u), ref, notes
    if g == "length":
        def f(P, W, U):
            obj = c(P, W, U)
            if op == "default":
                return obj.get_arclength()
            return getattr(obj, "get_arclength_"+{"adaptive": "quad_adaptive", "ode": "ode", "fixed": "quad_fixed"}[op])()
        notes = "Native defaults differ: NumPy global Gauss-10; JAX knot-aware adaptive 1e-6. Fixed JAX uses global Clenshaw-Curtis-41."
        return f, convert(P, W, U), rc.get_arclength(), notes
    if g == "projection_curve" or (g == "sensitivity_projection" and op == "curve"):
        parameters = np.array([.07, .33, .7, .96]) if n == 4 else np.linspace(.03, .97, n)
        points = rc.get_value(parameters).T+np.array([.05, .13, -.09])
        def f(P, W, U, points):
            obj = c(P, W, U)
            if profile == "jax":
                uv = obj.project_points(points.T, tol=1e-10)
            else:
                uv = xp.asarray([obj.project_point_to_curve(x, ftol=1e-12, gtol=1e-10) for x in points]).reshape(-1)
            # Objective distance handles multiple equally good minimizers.
            d2 = xp.sum((obj.get_value(uv).T-points)**2, axis=1)
            return d2, uv
        uv = rc.project(points)
        ref = (np.sum((rc.get_value(uv).T-points)**2, axis=1), uv)
        if g == "sensitivity_projection":
            return projection_sensitivity(f, convert(P, W, U, points), rc.get_value(uv).T-points, profile, xp)
        return f, convert(P, W, U, points), ref, f"{n} queries; NumPy scalar loop vs native JAX batch; compare squared distances and domain, parameters informational."
    if g in ("surface", "projection_surface", "prepared_projection", "sensitivity_projection", "construction", "workflow", "sensitivity", "plot", "prepared"):
        S, T, A, B, a, b, p_s, q_s = surface_data(kind, n)
        s = lambda S, T, A, B: backend.NurbsSurface(S, T, p_s, q_s, A, B)
        rs = Surface(S, T, A, B, p_s, q_s)
    if g == "surface":
        def evaluate(obj, a, b, namespace, order=24):
            if op == "area":
                return surface_area(obj, namespace, order)
            if op.startswith("d"):
                return obj.get_derivative(a, b, int(op[1]), int(op[2]))
            return getattr(obj, "get_"+op)(a, b)
        f = lambda S, T, A, B, a, b: evaluate(s(S, T, A, B), a, b, xp)
        ref = evaluate(rs, a, b, np, 64)
        if op == "area":
            notes = f"Derived Gauss-24 area; independent Gauss-64 reference; |64-48|={abs(float(ref)-float(surface_area(rs, np, 48))):.3g}."
        return f, convert(S, T, A, B, a, b), ref, notes
    if g in ("projection_surface", "prepared_projection", "sensitivity_projection"):
        parameters = np.array([.05, .3, .7, .95]) if n == 4 else np.linspace(.03, .97, n)
        points = rs.get_value(parameters, 1-parameters).T+np.array([.07, -.09, .18])
        prepared_projection = backend.prepare_surface_projection(s(xp.asarray(S), xp.asarray(T), xp.asarray(A), xp.asarray(B))) if g == "prepared_projection" else None
        def f(S, T, A, B, points):
            obj = s(S, T, A, B)
            if profile == "jax":
                uv = (prepared_projection or obj).project_points(points.T, ftol=1e-12, gtol=1e-10).T
            else:
                uv = xp.asarray([obj.project_point_to_surface(x, ftol=1e-12, gtol=1e-10) for x in points])
            d2 = xp.sum((obj.get_value(uv[:, 0], uv[:, 1]).T-points)**2, axis=1)
            return d2, uv
        uv = rs.project(points)
        ref = (np.sum((rs.get_value(uv[:, 0], uv[:, 1]).T-points)**2, axis=1), uv)
        if g == "sensitivity_projection":
            return projection_sensitivity(f, convert(S, T, A, B, points), rs.get_value(uv[:, 0], uv[:, 1]).T-points, profile, xp)
        return f, convert(S, T, A, B, points), ref, "Independent 25-start SciPy search plus edges/corners; numerical reference, not a global certificate."
    if g == "construction":
        if op == "circle":
            def f(scale):
                obj = backend.CircularArc(xp.asarray([0., 0, 0]), xp.asarray([1., 0, 0]), xp.asarray([0., 1, 0]), scale, 0., np.pi/2).NurbsCurve
                return obj.get_value(xp.asarray(u))
            _, _, _, _, _ = curve_data("circle", n)
            P0, W0, U0, x0, p0 = curve_data("circle", n)
            return f, (xp.asarray(1.),), Curve(P0, W0, U0, p0).get_value(x0), "Host constructor chooses segment count from angle."
        if op == "merge":
            def f(P, W, U):
                obj = c(P, W, U)
                other = c(P+(P[:, -1]-P[:, 0])[:, None], W, U)
                return backend.merge_nurbs_curves([obj, other]).get_value(xp.asarray(u))
            return f, convert(P, W, U), None, "Cross-backend comparison; existing epsilon-separated join convention."
        def f(S, T, A, B):
            obj = s(S, T, A, B)
            if op.startswith("isocurve"):
                return getattr(obj, "get_"+op)(.37).get_value(xp.asarray(a))
            if op.startswith("attach"):
                axis = 1 if op == "attach_u" else 2
                shift = (S.take(-1, axis=axis)-S.take(0, axis=axis))
                # Match the homogeneous boundary, not only its control points.
                # Unequal boundary weights would describe a different seam.
                other_weights = xp.concatenate((T[-1:, :], T[1:, :]), axis=0) if axis == 1 else xp.concatenate((T[:, -1:], T[:, 1:]), axis=1)
                other = s(S+xp.expand_dims(shift, axis=axis), other_weights, A, B)
                obj = getattr(obj, "attach_nurbs_"+("udir" if axis == 1 else "vdir"))(other)
            else:
                obj = backend.make_offset_surface(obj, .1)
            return obj.get_value(xp.asarray(a), xp.asarray(b))
        ref = rs.get_value(np.full_like(a, .37), a) if op == "isocurve_u" else rs.get_value(a, np.full_like(a, .37)) if op == "isocurve_v" else None
        if op.startswith("attach"):
            axis = 1 if op == "attach_u" else 2
            shift = S.take(-1, axis=axis)-S.take(0, axis=axis)
            other_weights = np.concatenate((T[-1:, :], T[1:, :]), axis=0) if axis == 1 else np.concatenate((T[:, -1:], T[:, 1:]), axis=1)
            other = Surface(S+np.expand_dims(shift, axis=axis), other_weights, A, B, p_s, q_s)
            t = a if axis == 1 else b
            first = rs.get_value(np.minimum(2*a, 1), b) if axis == 1 else rs.get_value(a, np.minimum(2*b, 1))
            second = other.get_value(np.maximum(2*a-1, 0), b) if axis == 1 else other.get_value(a, np.maximum(2*b-1, 0))
            ref = np.where(t[None, :] < .5, first, second)
        if op == "offset":
            nu, nv = 10*S.shape[1], 10*S.shape[2]
            x, y = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="ij")
            x, y = x.ravel(), y.ravel()
            net = (rs.get_value(x, y)-.1*rs.get_normals(x, y)).reshape((3, nu, nv))
            knots = lambda count, degree: np.r_[np.zeros(degree), np.linspace(0, 1, count-degree+1), np.ones(degree)]
            offset_reference = Surface(net, np.ones((nu, nv)), knots(nu, p_s), knots(nv, q_s), p_s, q_s)
            ref = offset_reference.get_value(a, b)
        return f, convert(S, T, A, B), ref, "Offset is an approximate sampled control net, not an exact offset." if op == "offset" else ""
    if g == "generator":
        a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
        f = lambda scale: generator(backend, kind, xp, scale).get_value(xp.asarray(a.ravel()), xp.asarray(b.ravel()))
        return f, (xp.asarray(1.2),), None, "Cross-backend rational generator comparison; construction included."
    if g in ("workflow", "sensitivity"):
        def curve_work(P, W, U, u):
            obj = c(P, W, U)
            # Fixed common sample quadrature makes the entire workflow equivalent.
            x, w = gauss(32)
            speed = xp.linalg.norm(obj.get_derivative(xp.asarray(x), 1), axis=0)
            return xp.sum(obj.get_value(u)**2)+.01*xp.sum(obj.get_curvature(u)**2)+xp.sum(speed*xp.asarray(w))
        def surface_work(S, T, A, B, a, b):
            obj = s(S, T, A, B)
            H, K = obj.get_curvature(a, b)
            return xp.sum(obj.get_value(a, b)**2)+.01*xp.sum(H*H+K*K)+surface_area(obj, xp)
        f, args = (curve_work, convert(P, W, U, u)) if op.startswith("curve") else (surface_work, convert(S, T, A, B, a, b))
        if g == "sensitivity":
            base = f
            argnum = 1 if op.endswith("weights") else 0
            if profile == "jax":
                f = jax.jit(jax.grad(base, argnums=argnum))
            else:
                def f(*args):
                    grad = np.empty_like(args[argnum]); step = 2e-5
                    for index in np.ndindex(grad.shape):
                        hi, lo = args[argnum].copy(), args[argnum].copy()
                        hi[index] += step; lo[index] -= step
                        high_args, low_args = list(args), list(args)
                        high_args[argnum], low_args[argnum] = hi, lo
                        grad[index] = (base(*high_args)-base(*low_args))/(2*step)
                    return grad
            notes = "JAX reverse AD vs NumPy central finite differences (h=2e-5); accuracy and computational work differ."
        return f, args, None, notes or "Complete sampled energy/geometry/integration workflow; excludes projection, measured separately."
    if g == "plot":
        def f(*args):
            import matplotlib.pyplot as plt
            obj = c(*args) if op == "curve" else s(*args)
            fig, ax = obj.plot()
            fig.canvas.draw()
            result = np.asarray([len(ax.lines), len(ax.collections), len(ax.patches)])
            plt.close(fig)
            return result
        return f, convert(P, W, U) if op == "curve" else convert(S, T, A, B), None, "Agg render smoke test and artist counts only; not image pixel parity."
    if g == "prepared":
        if op.startswith("curve"):
            prepared = backend.prepare_curve(c(xp.asarray(P), xp.asarray(W), xp.asarray(U)), u, max_order=2)
            method = {"curve_value": "get_value", "curve_d2": "get_derivative", "curve_curvature": "get_curvature"}[op]
            f = lambda P, W: getattr(prepared, method)(P, W, **({"order": 2} if op == "curve_d2" else {}))
            ref = rc.get_derivative(u, 2) if op == "curve_d2" else getattr(rc, "get_"+op.split("_")[1])(u)
            return f, convert(P, W), ref, "Setup measured separately; sample basis reused with dynamic control points/weights."
        if op == "length":
            prepared = backend.prepare_arclength(c(xp.asarray(P), xp.asarray(W), xp.asarray(U)), order=16, subdivisions=4)
            return lambda P, W: prepared.value(P, W), convert(P, W), rc.get_arclength(), "Prepared span Gauss-16, four subdivisions; fixed resolution, no adaptive error bound."
        grid = op == "surface_grid"
        x = np.linspace(0, 1, n)
        prepared = backend.prepare_surface(s(xp.asarray(S), xp.asarray(T), xp.asarray(A), xp.asarray(B)), x if grid else a, x if grid else b, max_order_u=2, max_order_v=2, grid=grid)
        method = {"surface_value": "get_value", "surface_grid": "get_value", "surface_d11": "get_derivative", "surface_curvature": "get_curvature"}[op]
        f = lambda S, T: getattr(prepared, method)(S, T, **({"order_u": 1, "order_v": 1} if op == "surface_d11" else {}))
        ref = rs.get_derivative(a, b, 1, 1) if op == "surface_d11" else rs.get_curvature(a, b) if op == "surface_curvature" else rs.get_value(a, b)
        if grid:
            ref = ref.reshape((3, n, n))
        return f, convert(S, T), ref, "Prepared grid factorizes tensor evaluation." if grid else "Prepared sample tables; setup measured separately."
    if g in ("map", "map_query"):
        obj = c(xp.asarray(P), xp.asarray(W), xp.asarray(U))
        if op in ("arclength", "unit_speed", "unit_speed_tight"):
            def f(P, W, U):
                inverse, length = c(P, W, U).reparametrize_by_arclength()
                values = xp.linspace(0, length, n)
                return inverse(values), length
            # Invert independent cumulative length, not the spline parameter.
            from scipy.optimize import brentq
            length = np.pi/2 if kind == "circle" else rc.get_arclength()
            ref_u = np.array([0 if x == 0 else 1 if x == length else brentq(lambda t: rc.get_arclength(0, t)-x, 0, 1) for x in np.linspace(0, length, n)])
            if g == "map_query":
                options = {"tol": 1e-9, "steps_per_span": 64} if op == "unit_speed_tight" else {}
                inverse, total = obj.reparametrize_by_arclength(**options)
                if op.startswith("unit_speed"):
                    def query(s):
                        parameter = inverse(s)
                        derivative = jax.vmap(jax.grad(inverse))(s)
                        speed = xp.linalg.norm(obj.get_derivative(parameter, 1), axis=0)
                        return parameter, speed*derivative
                    return query, (xp.linspace(0, total, n),), (ref_u, np.ones(n)), "Dense inverse and unit-speed check via AD; independent cumulative-length inversion; "+("tol=1e-9, steps_per_span=64." if options else "default tol=1e-6, steps_per_span=16.")
                return inverse, (xp.linspace(0, total, n),), ref_u, "Reusable inverse-map query; geometry snapshot built once during setup."
            return f, convert(P, W, U), (ref_u, length), "Host map construction and dense query included; inverse accuracy checked independently."
        def f(P, W, U):
            inverse = c(P, W, U).reparametrize_by_coordinate(dim=0)
            return inverse(xp.linspace(P[0, 0], P[0, -1], n))
        from scipy.optimize import brentq
        ref = np.array([brentq(lambda t: rc.get_value(t)[0, 0]-x, 0, 1) for x in np.linspace(P[0, 0], P[0, -1], n)])
        return f, convert(P, W, U), ref, "Monotone x-coordinate inverse on this fixture."
    raise ValueError(row)


def projection_sensitivity(project, inputs, residuals, profile, xp):
    """Uniform z-translation derivative of the minimized squared distance.

    The envelope derivative is 2*sum(z residual) at a converged minimizer,
    including stable constrained minima. The independent reference uses the
    SciPy minimizer, not the production projection or its derivatives.
    """
    def loss(shift):
        translation = xp.asarray([0., 0., 1.])*shift
        geometry = inputs[0]+translation.reshape((3,)+(1,)*(inputs[0].ndim-1))
        return xp.sum(project(geometry, *inputs[1:])[0])
    if profile == "jax":
        import jax
        function = jax.jit(jax.grad(loss))
    else:
        def function(shift):
            step = 2e-5
            return (loss(shift+step)-loss(shift-step))/(2*step)
    return function, (xp.asarray(0.),), 2*np.sum(residuals[:, 2]), "Projection-loss z-translation sensitivity: JAX AD vs NumPy central differences; independent envelope reference."
