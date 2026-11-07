"""Function-oriented static rendering shared by NumPy/JAX; interactive UI is separate."""
import numpy as np
import matplotlib as mpl
from ..graphics import style_3d_axes, close_3d_box, _prepare_plot_axes


def _host_geometry(geometry):
    """Copy JAX geometry to NumPy at the visualization boundary, outside JIT."""
    if isinstance(geometry.P, np.ndarray):
        return geometry
    from ..numpy import NurbsCurve, NurbsSurface
    if np.ndim(geometry.P) == 2:
        return NurbsCurve(np.asarray(geometry.P), np.asarray(geometry.W), geometry.p, np.asarray(geometry.U))
    return NurbsSurface(np.asarray(geometry.P), np.asarray(geometry.W), geometry.p, geometry.q, np.asarray(geometry.U), np.asarray(geometry.V))


def curve_plot(geometry, fig=None, ax=None, curve=True, control_points=True, frenet_serret=False, axis_off=False, ticks_off=False, rescale=True):
    """Plot geometry on new or supplied axes; return ``(fig, ax)``."""
    geometry = _host_geometry(geometry)
    if geometry.ndim not in (1, 2, 3):
        raise ValueError('The number of dimensions must be 1, 2 or 3')
    new_axes = ax is None
    fig, ax = _prepare_plot_axes(fig, ax, '3d' if geometry.ndim == 3 else None)
    if new_axes:
        if geometry.ndim == 3:
            style_3d_axes(ax, elev=30, azim=-120)
        labels = ('$u$ parameter', 'NURBS curve value') if geometry.ndim == 1 else ('$x$ axis', '$y$ axis')
        ax.set_xlabel(labels[0], fontsize=12, labelpad=12)
        ax.set_ylabel(labels[1], fontsize=12, labelpad=12)
        if geometry.ndim == 3:
            ax.set_zlabel('$z$ axis', fontsize=12, labelpad=12)
    if ticks_off:
        ax.set_xticks([])
        ax.set_yticks([])
        if geometry.ndim == 3:
            ax.set_zticks([])
    if axis_off:
        ax.axis('off')
    if curve:
        geometry.plot_curve(fig, ax)
    if control_points:
        geometry.plot_control_points(fig, ax)
    if frenet_serret:
        geometry.plot_frenet_serret(fig, ax)
    if rescale:
        geometry.rescale_plot(fig, ax)
    return (fig, ax)

def curve_plot_curve(geometry, fig, ax, linewidth=1.5, linestyle='-', color='black', u1=0.0, u2=1.0):
    """ Plot the coordinates of the NURBS curve """
    geometry = _host_geometry(geometry)
    if geometry.ndim == 1:
        u = np.linspace(u1, u2, 501)
        X = np.real(geometry.get_value(u))
        line, = ax.plot(u, X[0, :])
        line.set_linewidth(linewidth)
        line.set_linestyle(linestyle)
        line.set_color(color)
        line.set_marker(' ')
    elif geometry.ndim == 2:
        u = np.linspace(u1, u2, 501)
        X, Y = np.real(geometry.get_value(u))
        line, = ax.plot(X, Y)
        line.set_linewidth(linewidth)
        line.set_linestyle(linestyle)
        line.set_color(color)
        line.set_marker(' ')
    elif geometry.ndim == 3:
        u = np.linspace(u1, u2, 501)
        X, Y, Z = np.real(geometry.get_value(u))
        line, = ax.plot(X, Y, Z)
        line.set_linewidth(linewidth)
        line.set_linestyle(linestyle)
        line.set_color(color)
        line.set_marker(' ')
    else:
        raise Exception('The number of dimensions must be 1, 2 or 3')
    return (fig, ax)

def curve_plot_control_points(geometry, fig, ax, linewidth=1.25, linestyle='-.', color='red', markersize=5, markerstyle='o'):
    """ Plot the control points of the NURBS curve """
    geometry = _host_geometry(geometry)
    if geometry.ndim == 1:
        Px = np.real(geometry.P)
        u = np.linspace(0, 1, Px.size)
        line, = ax.plot(u, Px[0, :])
        line.set_linewidth(linewidth)
        line.set_linestyle(linestyle)
        line.set_color(color)
        line.set_marker(markerstyle)
        line.set_markersize(markersize)
        line.set_markeredgewidth(linewidth)
        line.set_markeredgecolor(color)
        line.set_markerfacecolor('w')
        line.set_zorder(4)
    elif geometry.ndim == 2:
        Px, Py = np.real(geometry.P)
        line, = ax.plot(Px, Py)
        line.set_linewidth(linewidth)
        line.set_linestyle(linestyle)
        line.set_color(color)
        line.set_marker(markerstyle)
        line.set_markersize(markersize)
        line.set_markeredgewidth(linewidth)
        line.set_markeredgecolor(color)
        line.set_markerfacecolor('w')
        line.set_zorder(4)
    elif geometry.ndim == 3:
        Px, Py, Pz = np.real(geometry.P)
        line, = ax.plot(Px, Py, Pz)
        line.set_linewidth(linewidth)
        line.set_linestyle(linestyle)
        line.set_color(color)
        line.set_marker(markerstyle)
        line.set_markersize(markersize)
        line.set_markeredgewidth(linewidth)
        line.set_markeredgecolor(color)
        line.set_markerfacecolor('w')
        line.set_zorder(4)
    else:
        raise Exception('The number of dimensions must be 2 or 3')
    return (fig, ax)

def curve_plot_frenet_serret(geometry, fig, ax, frame_number=5, frame_scale=0.1):
    """Plot defined frame vectors; omit undefined directions at singular points."""
    geometry = _host_geometry(geometry)
    if geometry.ndim not in (2, 3):
        raise ValueError('Frenet-Serret frames require a two- or three-dimensional curve')
    if not isinstance(frame_number, (int, np.integer)) or isinstance(frame_number, bool) or frame_number < 1:
        raise ValueError('frame_number must be a positive integer')
    if not np.isfinite(frame_scale):
        raise ValueError('frame_scale must be finite')
    scale = frame_scale * np.real(geometry.get_arclength(0, 1))
    u = np.linspace(1e-12, 1 - 1e-12, frame_number)
    position = np.real(geometry.get_value(u))
    with np.errstate(divide='ignore', invalid='ignore'):
        vectors = [np.real(geometry.get_tangent(u)), np.real(geometry.get_normal(u))]
        if geometry.ndim == 3:
            vectors.append(np.real(geometry.get_binormal(u)))
    for k in range(frame_number):
        if not np.all(np.isfinite(position[:, k])):
            continue
        for vector, color in zip(vectors, ('red', 'blue', 'green')):
            direction = vector[:, k]
            if not np.all(np.isfinite(direction)) or not np.any(direction) or scale == 0:
                continue
            if geometry.ndim == 2:
                ax.quiver(*position[:, k], *scale * direction, color=color, angles='xy', scale_units='xy', scale=1)
            else:
                ax.quiver(*position[:, k], *direction, color=color, length=scale, normalize=True)
    valid = np.all(np.isfinite(position), axis=0)
    ax.plot(*position[:, valid], linestyle=' ', marker='o', markersize=5, markeredgewidth=1.25, markeredgecolor='k', markerfacecolor='w', zorder=4)
    return (fig, ax)

def curve_rescale_plot(geometry, fig, ax):
    """ Adjust the aspect ratio of the figure """
    geometry = _host_geometry(geometry)
    if geometry.ndim == 2:
        ax.set_aspect(1.0)
        fig.tight_layout(pad=5.0, w_pad=None, h_pad=None)
    if geometry.ndim == 3:
        ax.autoscale(enable=True)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        z_min, z_max = ax.get_zlim()
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2
        L = np.max((x_max - x_min, y_max - y_min, z_max - z_min)) / 2
        ax.set_xlim3d(x_mid - 1.0 * L, x_mid + 1.0 * L)
        ax.set_ylim3d(y_mid - 1.0 * L, y_mid + 1.0 * L)
        ax.set_zlim3d(z_mid - 1.0 * L, z_mid + 1.0 * L)
        close_3d_box(ax)
        fig.tight_layout(pad=5.0, w_pad=None, h_pad=None)

def curve_plot_curvature(geometry, fig=None, ax=None, color='black', linestyle='-'):
    geometry = _host_geometry(geometry)
    fig, ax = _prepare_plot_axes(fig, ax)
    ax.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
    ax.set_ylabel('Curvature', fontsize=12, color='k', labelpad=12)
    for t in ax.xaxis.get_major_ticks():
        t.label1.set_fontsize(12)
    for t in ax.yaxis.get_major_ticks():
        t.label1.set_fontsize(12)
    u = np.linspace(0, 1, 1000)
    curvature = np.real(geometry.get_curvature(u))
    line, = ax.plot(u, curvature)
    line.set_linewidth(1.25)
    line.set_linestyle(linestyle)
    line.set_color(color)
    line.set_marker(' ')
    line.set_markersize(3.5)
    line.set_markeredgewidth(1)
    line.set_markeredgecolor('k')
    line.set_markerfacecolor('w')
    fig.tight_layout(pad=5.0, w_pad=None, h_pad=None)
    return (fig, ax)

def curve_plot_torsion(geometry, fig=None, ax=None, color='black', linestyle='-'):
    geometry = _host_geometry(geometry)
    fig, ax = _prepare_plot_axes(fig, ax)
    ax.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
    ax.set_ylabel('Torsion', fontsize=12, color='k', labelpad=12)
    for t in ax.xaxis.get_major_ticks():
        t.label1.set_fontsize(12)
    for t in ax.yaxis.get_major_ticks():
        t.label1.set_fontsize(12)
    u = np.linspace(0, 1, 1000)
    torsion = np.real(geometry.get_torsion(u))
    line, = ax.plot(u, torsion)
    line.set_linewidth(1.25)
    line.set_linestyle(linestyle)
    line.set_color(color)
    line.set_marker(' ')
    line.set_markersize(3.5)
    line.set_markeredgewidth(1)
    line.set_markeredgecolor('k')
    line.set_markerfacecolor('w')
    ratio = 1.0
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    span_y = abs(y2 - y1)
    ax.set_aspect(abs(x2 - x1) / span_y * ratio if span_y > 0 else 'auto')
    fig.tight_layout(pad=5.0, w_pad=None, h_pad=None)
    return (fig, ax)

def surface_plot(geometry, fig=None, ax=None, surface=True, surface_color='blue', colorbar=False, boundary=True, control_points=False, normals=False, axis_off=False, ticks_off=False, Nu=50, Nv=50, isocurves_u=None, isocurves_v=None):
    """Plot a scalar, planar, or spatial patch; return ``(fig, ax)``."""
    geometry = _host_geometry(geometry)
    if geometry.ndim not in (1, 2, 3):
        raise ValueError('The number of dimensions must be 1, 2 or 3')
    projection = '3d' if geometry.ndim in (1, 3) else None
    new_axes = ax is None
    fig, ax = _prepare_plot_axes(fig, ax, projection)
    if new_axes:
        if projection:
            style_3d_axes(ax, elev=30, azim=-105)
        labels = ('$u$ parameter', '$v$ parameter', 'NURBS value') if geometry.ndim == 1 else ('$x$ axis', '$y$ axis', '$z$ axis')
        ax.set_xlabel(labels[0], fontsize=11, labelpad=12)
        ax.set_ylabel(labels[1], fontsize=11, labelpad=12)
        if projection:
            ax.set_zlabel(labels[2], fontsize=11, labelpad=12)
    if ticks_off:
        ax.set_xticks([])
        ax.set_yticks([])
        if projection:
            ax.set_zticks([])
    if axis_off:
        ax.axis('off')
    if surface:
        geometry.plot_surface(fig, ax, color=surface_color, colorbar=colorbar, Nu=Nu, Nv=Nv)
    if control_points:
        geometry.plot_control_points(fig, ax)
    if geometry.ndim > 1:
        if boundary:
            geometry.plot_boundary(fig, ax)
        if isocurves_u:
            geometry.plot_isocurve_u(fig, ax, np.linspace(0, 1, isocurves_u))
        if isocurves_v:
            geometry.plot_isocurve_v(fig, ax, np.linspace(0, 1, isocurves_v))
    if normals:
        geometry.plot_normals(fig, ax)
    if geometry.ndim == 3:
        geometry.rescale_plot(fig, ax)
    elif geometry.ndim == 2:
        ax.set_aspect(1.0)
    else:
        ax.autoscale(enable=True)
        close_3d_box(ax)
    return (fig, ax)

def surface_plot_surface(geometry, fig, ax, color='blue', alpha=0.3, colorbar=False, Nu=50, Nv=50):
    """Draw geometry, optionally coloured by mean or Gaussian curvature."""
    geometry = _host_geometry(geometry)
    for name, count in (('Nu', Nu), ('Nv', Nv)):
        if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 2:
            raise ValueError(f'{name} must be an integer of at least 2')
    u, v = np.meshgrid(np.linspace(0, 1, Nu), np.linspace(0, 1, Nv), indexing='ij')
    coordinates = np.real(geometry.get_value(u.ravel(), v.ravel())).reshape((geometry.ndim, Nu, Nv))
    if geometry.ndim == 2:
        x, y = coordinates
        perimeter = lambda grid: np.concatenate((grid[:, 0], grid[-1, 1:], grid[-2::-1, -1], grid[0, -2:0:-1]))
        ax.fill(perimeter(x), perimeter(y), color=color, alpha=alpha)
        return
    if geometry.ndim == 1:
        X, Y, Z = (u, v, coordinates[0])
    elif geometry.ndim == 3:
        X, Y, Z = coordinates
    else:
        raise ValueError('The number of dimensions must be 1, 2 or 3')
    if isinstance(color, str) and color in ('mean_curvature', 'gaussian_curvature'):
        if geometry.ndim != 3:
            raise ValueError('Curvature colouring requires a three-dimensional surface')
        curvatures = np.real(geometry.get_curvature(u.ravel(), v.ravel()))
        curvature = curvatures[0 if color == 'mean_curvature' else 1].reshape((Nu, Nv))
        finite = curvature[np.isfinite(curvature)]
        if not finite.size:
            raise ValueError('Curvature is undefined at all sampled points')
        low, high = (finite.min(), finite.max())
        if low == high:
            padding = max(1.0, abs(low)) * 1e-06
            low, high = (low - padding, high + padding)
        norm = mpl.colors.Normalize(vmin=low, vmax=high)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
        mapper.set_array(curvature)
        ax.plot_surface(X, Y, Z, facecolors=mapper.to_rgba(curvature), linewidth=0.75, alpha=1, shade=False, antialiased=True, zorder=2, rcount=Nu, ccount=Nv)
        if colorbar:
            cbar = fig.colorbar(mapper, ax=ax, orientation='vertical', pad=0.15, fraction=0.03, aspect=20)
            cbar.set_label(color.replace('_', ' '))
    else:
        ax.plot_surface(X, Y, Z, color=color, linewidth=0, alpha=alpha, shade=False, antialiased=True, zorder=0, rcount=Nu, ccount=Nv)

def surface_plot_boundary(geometry, fig, ax, color='black', linewidth=1.0, linestyle='-', south=True, north=True, east=True, west=True):
    """ Plot the isoparametric curves at the boundary """
    geometry = _host_geometry(geometry)
    if east:
        geometry.get_isocurve_u(u0=0.0).plot_curve(fig, ax, linestyle=linestyle, linewidth=linewidth, color=color)
    if west:
        geometry.get_isocurve_u(u0=1.0).plot_curve(fig, ax, linestyle=linestyle, linewidth=linewidth, color=color)
    if south:
        geometry.get_isocurve_v(v0=0.0).plot_curve(fig, ax, linestyle=linestyle, linewidth=linewidth, color=color)
    if north:
        geometry.get_isocurve_v(v0=1.0).plot_curve(fig, ax, linestyle=linestyle, linewidth=linewidth, color=color)

def surface_plot_isocurve_u(geometry, fig, ax, u_values, color='black', linewidth=1.0, linestyle='-'):
    """ Plot isoparametric curves in the u-direction """
    geometry = _host_geometry(geometry)
    for u in u_values:
        geometry.get_isocurve_u(u0=u).plot_curve(fig, ax, color=color, linewidth=linewidth, linestyle=linestyle)

def surface_plot_isocurve_v(geometry, fig, ax, v_values, color='black', linewidth=1.0, linestyle='-'):
    """ Plot isoparametric curves in the v-direction """
    geometry = _host_geometry(geometry)
    for v in v_values:
        geometry.get_isocurve_v(v0=v).plot_curve(fig, ax, color=color, linewidth=linewidth, linestyle=linestyle)

def surface_plot_control_points(geometry, fig, ax, color='red', linewidth=1.0, linestyle='-', markersize=5, markerstyle='o'):
    """ Plot the control points """
    geometry = _host_geometry(geometry)
    if geometry.ndim == 1:
        Px = np.linspace(0, 1, np.shape(geometry.P)[1])
        Py = np.linspace(0, 1, np.shape(geometry.P)[2])
        Px, Py = np.meshgrid(Px, Py, indexing='ij')
        Pz = np.real(geometry.P)[0, :, :]
        ax.plot_wireframe(Px, Py, Pz, edgecolor=color, linewidth=linewidth, linestyles=linestyle, alpha=1.0, antialiased=True, zorder=1)
        points, = ax.plot(Px.flatten(), Py.flatten(), Pz.flatten())
        points.set_linewidth(linewidth)
        points.set_linestyle(' ')
        points.set_marker(markerstyle)
        points.set_markersize(markersize)
        points.set_markeredgewidth(linewidth)
        points.set_markeredgecolor(color)
        points.set_markerfacecolor('w')
        points.set_zorder(4)
    if geometry.ndim == 2:
        Px, Py = np.real(geometry.P)
        ax.plot(Px, Py, color=color, linewidth=linewidth, linestyle='-', marker=markerstyle, markersize=markersize, markeredgewidth=linewidth, markeredgecolor=color, markerfacecolor='w', zorder=4)
        ax.plot(Px.transpose(), Py.transpose(), color=color, linewidth=linewidth, linestyle='-', marker=markerstyle, markersize=markersize, markeredgewidth=linewidth, markeredgecolor=color, markerfacecolor='w', zorder=4)
    if geometry.ndim == 3:
        Px, Py, Pz = np.real(geometry.P)
        ax.plot_wireframe(Px, Py, Pz, edgecolor=color, linewidth=linewidth, linestyles=linestyle, alpha=1.0, antialiased=True, zorder=1)
        points, = ax.plot(Px.flatten(), Py.flatten(), Pz.flatten())
        points.set_linewidth(linewidth)
        points.set_linestyle(' ')
        points.set_marker(markerstyle)
        points.set_markersize(markersize)
        points.set_markeredgewidth(linewidth)
        points.set_markeredgecolor(color)
        points.set_markerfacecolor('w')
        points.set_zorder(4)

def surface_plot_normals(geometry, fig, ax, number_u=10, number_v=10, scale=0.075):
    """Plot normals with a length proportional to the surface dimensions."""
    geometry = _host_geometry(geometry)
    if geometry.ndim != 3:
        raise ValueError('Normal vectors require a three-dimensional surface')
    if not np.isfinite(scale):
        raise ValueError('scale must be finite')
    for name, count in (('number_u', number_u), ('number_v', number_v)):
        if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 1:
            raise ValueError(f'{name} must be a positive integer')
    u, v = np.meshgrid(np.linspace(1e-06, 1 - 1e-06, number_u), np.linspace(1e-06, 1 - 1e-06, number_v), indexing='ij')
    u, v = (u.ravel(), v.ravel())
    S = np.real(geometry.get_value(u, v))
    tangent_u = np.real(geometry.get_derivative(u, v, 1, 0))
    tangent_v = np.real(geometry.get_derivative(u, v, 0, 1))
    N = np.cross(tangent_u, tangent_v, axis=0)
    magnitude = np.linalg.norm(N, axis=0)
    valid = (magnitude > 0) & np.isfinite(magnitude) & np.all(np.isfinite(S), axis=0)
    if not valid.any() or scale == 0:
        return
    Lu = geometry.get_isocurve_u(u0=0.5).get_arclength()
    Lv = geometry.get_isocurve_v(v0=0.5).get_arclength()
    length = scale * np.real(np.max([Lu, Lv]))
    N = N[:, valid] / magnitude[valid] * length
    S = S[:, valid]
    ax.quiver(*S, *N, color='black', length=1, normalize=False)

def surface_rescale_plot(geometry, fig, ax):
    """ Adjust the aspect ratio of the figure """
    geometry = _host_geometry(geometry)
    ax.autoscale(enable=True)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2
    L = np.max((x_max - x_min, y_max - y_min, z_max - z_min)) / 2
    ax.set_xlim3d(x_mid - 1.0 * L, x_mid + 1.0 * L)
    ax.set_ylim3d(y_mid - 1.0 * L, y_mid + 1.0 * L)
    ax.set_zlim3d(z_mid - 1.0 * L, z_mid + 1.0 * L)
    close_3d_box(ax)
    fig.tight_layout(pad=5.0, w_pad=None, h_pad=None)

def surface_plot_curvature(geometry, fig=None, ax=None, curvature_type='mean'):
    geometry = _host_geometry(geometry)
    fig, ax = _prepare_plot_axes(fig, ax, '3d')
    style_3d_axes(ax, elev=30, azim=-105)
    ax.set_xlabel('$x$ axis', fontsize=11, color='k', labelpad=18)
    ax.set_ylabel('$y$ axis', fontsize=11, color='k', labelpad=18)
    ax.set_zlabel('$z$ axis', fontsize=11, color='k', labelpad=18)
    for t in ax.xaxis.get_major_ticks():
        t.label1.set_fontsize(8)
    for t in ax.yaxis.get_major_ticks():
        t.label1.set_fontsize(8)
    for t in ax.zaxis.get_major_ticks():
        t.label1.set_fontsize(8)
    Nu, Nv = (50, 50)
    u = np.linspace(0.0, 1.0, Nu)
    v = np.linspace(0.0, 1.0, Nv)
    [uu, vv] = np.meshgrid(u, v, indexing='ij')
    u = uu.flatten()
    v = vv.flatten()
    if curvature_type == 'mean':
        curvature, _ = np.real(geometry.get_curvature(u, v))
    elif curvature_type == 'gaussian':
        _, curvature = np.real(geometry.get_curvature(u, v))
    else:
        raise Exception("Choose a valid curvature type: 'mean' or 'gaussian'")
    ax.set_xlabel('$u$', fontsize=11, color='k', labelpad=10)
    ax.set_ylabel('$v$', fontsize=11, color='k', labelpad=10)
    ax.set_zlabel('$\\kappa$' + ' ' + curvature_type, fontsize=11, color='k', labelpad=20)
    curvature = np.reshape(curvature, (Nu, Nv))
    ax.plot_surface(uu, vv, curvature, color='blue', linewidth=0.75, alpha=0.5, shade=False, antialiased=True, zorder=0, ccount=50, rcount=50)
    ax.autoscale(enable=True)
    close_3d_box(ax)
    fig.tight_layout(pad=5.0, w_pad=None, h_pad=None)
    return (fig, ax)

def plot(geometry, *args, **kwargs):
    """Return Matplotlib (figure, axes); use show/edit for interactive geometry."""
    return (curve_plot if np.ndim(geometry.P)==2 else surface_plot)(geometry,*args,**kwargs)
