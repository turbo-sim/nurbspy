# Public API coverage inventory

Generated from public top-level symbols and public class methods in backend source modules. Private modules/symbols, imports/re-exports and special methods are omitted. This is an inventory of API coverage, not a claim that all input combinations are tested.

Important limits: no complex-valued sensitivity study, no GPU/TPU timings, no exhaustive malformed-input matrix, no global projection certificate, no individually timed plot styling, and no separate diagnostic-status stress matrix. Surface area and energy workflows are derived calculations. Most domains are [0,1]; shifted curve evaluation/derivative cases use [-2,3].

| Backend | Source symbol | Coverage | Detail |
| --- | --- | --- | --- |
| numpy | `nurbs_basis_functions.compute_basis_polynomials` | Direct | basis value and derivative workloads. |
| numpy | `nurbs_basis_functions.compute_basis_polynomials_derivatives` | Direct | basis value and derivative workloads. |
| numpy | `nurbs_curve.NurbsCurve` | Direct | Constructed by deterministic geometry/generator workloads. |
| numpy | `nurbs_curve.NurbsCurve.get_value` | Direct | Curve/surface numerical workloads and independent references. |
| numpy | `nurbs_curve.NurbsCurve.compute_nurbs_coordinates` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| numpy | `nurbs_curve.NurbsCurve.compute_bspline_coordinates` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| numpy | `nurbs_curve.NurbsCurve.get_derivative` | Direct | Curve/surface numerical workloads and independent references. |
| numpy | `nurbs_curve.NurbsCurve.compute_nurbs_derivatives` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| numpy | `nurbs_curve.NurbsCurve.compute_bspline_derivatives` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| numpy | `nurbs_curve.NurbsCurve.get_tangent` | Direct | Curve/surface numerical workloads and independent references. |
| numpy | `nurbs_curve.NurbsCurve.get_normal` | Direct | Curve/surface numerical workloads and independent references. |
| numpy | `nurbs_curve.NurbsCurve.get_binormal` | Direct | Curve/surface numerical workloads and independent references. |
| numpy | `nurbs_curve.NurbsCurve.get_normal_2D` | Direct | normal2d.rotated_tangent; JAX uses an explicit rotated-tangent adapter. |
| numpy | `nurbs_curve.NurbsCurve.get_curvature` | Direct | Curve/surface numerical workloads and independent references. |
| numpy | `nurbs_curve.NurbsCurve.get_torsion` | Direct | Curve/surface numerical workloads and independent references. |
| numpy | `nurbs_curve.NurbsCurve.get_arclength` | Direct | Native default and JAX explicit method workloads. |
| numpy | `nurbs_curve.NurbsCurve.plot` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_curve.NurbsCurve.plot_curve` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_curve.NurbsCurve.plot_control_points` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_curve.NurbsCurve.plot_frenet_serret` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_curve.NurbsCurve.rescale_plot` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_curve.NurbsCurve.plot_curvature` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_curve.NurbsCurve.plot_torsion` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_curve.NurbsCurve.project_point_to_curve` | Direct / adapter | Projection workload uses native batches for JAX and loops over NumPy scalar calls. |
| numpy | `nurbs_curve.merge_nurbs_curves` | Direct | construction workloads. |
| numpy | `nurbs_curve_circular_arc.CircularArc` | Direct | Constructed by deterministic geometry/generator workloads. |
| numpy | `nurbs_curve_circular_arc.CircularArc.make_nurbs_circular_arc` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| numpy | `nurbs_curve_circular_arc.CircularArc.intersect_lines` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| numpy | `nurbs_curve_circular_arc.CircularArc.plot` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_surface.NurbsSurface` | Direct | Constructed by deterministic geometry/generator workloads. |
| numpy | `nurbs_surface.NurbsSurface.get_value` | Direct | Curve/surface numerical workloads and independent references. |
| numpy | `nurbs_surface.NurbsSurface.compute_nurbs_coordinates` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| numpy | `nurbs_surface.NurbsSurface.compute_bspline_coordinates` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| numpy | `nurbs_surface.NurbsSurface.get_derivative` | Direct | Curve/surface numerical workloads and independent references. |
| numpy | `nurbs_surface.NurbsSurface.compute_nurbs_derivatives` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| numpy | `nurbs_surface.NurbsSurface.compute_bspline_derivatives` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| numpy | `nurbs_surface.NurbsSurface.get_isocurve_u` | Direct | construction workloads. |
| numpy | `nurbs_surface.NurbsSurface.get_isocurve_v` | Direct | construction workloads. |
| numpy | `nurbs_surface.NurbsSurface.attach_nurbs_udir` | Direct | construction workloads. |
| numpy | `nurbs_surface.NurbsSurface.attach_nurbs_vdir` | Direct | construction workloads. |
| numpy | `nurbs_surface.NurbsSurface.get_normals` | Direct | Curve/surface numerical workloads and independent references. |
| numpy | `nurbs_surface.NurbsSurface.get_curvature` | Direct | Curve/surface numerical workloads and independent references. |
| numpy | `nurbs_surface.NurbsSurface.plot` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_surface.NurbsSurface.plot_surface` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_surface.NurbsSurface.plot_boundary` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_surface.NurbsSurface.plot_isocurve_u` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_surface.NurbsSurface.plot_isocurve_v` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_surface.NurbsSurface.plot_control_points` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_surface.NurbsSurface.plot_normals` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_surface.NurbsSurface.rescale_plot` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_surface.NurbsSurface.plot_curvature` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| numpy | `nurbs_surface.NurbsSurface.project_point_to_surface` | Direct / adapter | Projection workload uses native batches for JAX and loops over NumPy scalar calls. |
| numpy | `nurbs_surface.make_offset_surface` | Direct | construction workloads. |
| numpy | `nurbs_surface_bilinear.NurbsSurfaceBilinear` | Direct | Constructed by deterministic geometry/generator workloads. |
| numpy | `nurbs_surface_bilinear.NurbsSurfaceBilinear.make_nurbs_surface` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| numpy | `nurbs_surface_coons.NurbsSurfaceCoons` | Direct | Constructed by deterministic geometry/generator workloads. |
| numpy | `nurbs_surface_coons.NurbsSurfaceCoons.make_nurbs_surface` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| numpy | `nurbs_surface_extruded.NurbsSurfaceExtruded` | Direct | Constructed by deterministic geometry/generator workloads. |
| numpy | `nurbs_surface_extruded.NurbsSurfaceExtruded.make_Nurbs_surface` | Not separately exercised | Listed explicitly rather than assuming coverage. |
| numpy | `nurbs_surface_revolution.NurbsSurfaceRevolution` | Direct | Constructed by deterministic geometry/generator workloads. |
| numpy | `nurbs_surface_revolution.NurbsSurfaceRevolution.make_nurbs_surface` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| numpy | `nurbs_surface_revolution.NurbsSurfaceRevolution.intersect_lines` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| numpy | `nurbs_surface_revolution.NurbsSurfaceRevolution.project_point_to_line` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| numpy | `nurbs_surface_ruled.NurbsSurfaceRuled` | Direct | Constructed by deterministic geometry/generator workloads. |
| numpy | `nurbs_surface_ruled.NurbsSurfaceRuled.make_nurbs_surface` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| jax | `arclength.ArcLengthInfo` | JAX-only / partial | Length, prepared quadrature and inverse-map workloads; diagnostic return_info/status APIs not separately exercised. |
| jax | `arclength.PreparedArcLength` | JAX-only / partial | Length, prepared quadrature and inverse-map workloads; diagnostic return_info/status APIs not separately exercised. |
| jax | `arclength.PreparedArcLength.value` | JAX-only / partial | Length, prepared quadrature and inverse-map workloads; diagnostic return_info/status APIs not separately exercised. |
| jax | `arclength.prepare_arclength` | JAX-only / partial | Length, prepared quadrature and inverse-map workloads; diagnostic return_info/status APIs not separately exercised. |
| jax | `arclength.ArcLengthMap` | JAX-only / partial | Length, prepared quadrature and inverse-map workloads; diagnostic return_info/status APIs not separately exercised. |
| jax | `arclength.ArcLengthMap.total_length` | JAX-only / partial | Length, prepared quadrature and inverse-map workloads; diagnostic return_info/status APIs not separately exercised. |
| jax | `nurbs_basis_functions.compute_basis_polynomials` | Direct | basis value and derivative workloads. |
| jax | `nurbs_basis_functions.compute_all_basis_polynomials_derivatives` | Not separately timed | Alternative basis helper; main analytic basis path is compared. |
| jax | `nurbs_basis_functions.compute_basis_polynomials_derivatives` | Direct | basis value and derivative workloads. |
| jax | `nurbs_basis_functions.compute_basis_polynomials_derivatives_jax` | Not separately timed | Alternative basis helper; main analytic basis path is compared. |
| jax | `nurbs_curve.binomial_coeff` | Not separately exercised | Listed explicitly rather than assuming coverage. |
| jax | `nurbs_curve.compute_nurbs_coordinates` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| jax | `nurbs_curve.compute_bspline_coordinates` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| jax | `nurbs_curve.compute_all_bspline_derivatives` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| jax | `nurbs_curve.compute_all_nurbs_derivatives` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| jax | `nurbs_curve.NurbsCurve` | Direct | Constructed by deterministic geometry/generator workloads. |
| jax | `nurbs_curve.NurbsCurve.get_value` | Direct | Curve/surface numerical workloads and independent references. |
| jax | `nurbs_curve.NurbsCurve.get_derivative` | Direct | Curve/surface numerical workloads and independent references. |
| jax | `nurbs_curve.NurbsCurve.get_tangent` | Direct | Curve/surface numerical workloads and independent references. |
| jax | `nurbs_curve.NurbsCurve.get_normal` | Direct | Curve/surface numerical workloads and independent references. |
| jax | `nurbs_curve.NurbsCurve.get_binormal` | Direct | Curve/surface numerical workloads and independent references. |
| jax | `nurbs_curve.NurbsCurve.get_curvature` | Direct | Curve/surface numerical workloads and independent references. |
| jax | `nurbs_curve.NurbsCurve.get_torsion` | Direct | Curve/surface numerical workloads and independent references. |
| jax | `nurbs_curve.NurbsCurve.get_arclength` | Direct | Native default and JAX explicit method workloads. |
| jax | `nurbs_curve.NurbsCurve.get_arclength_quad_adaptive` | Direct | Native default and JAX explicit method workloads. |
| jax | `nurbs_curve.NurbsCurve.get_arclength_ode` | Direct | Native default and JAX explicit method workloads. |
| jax | `nurbs_curve.NurbsCurve.get_arclength_quad_fixed` | Direct | Native default and JAX explicit method workloads. |
| jax | `nurbs_curve.NurbsCurve.reparametrize_by_arclength` | JAX-only | Inverse-map construction and independent inverse accuracy checks. |
| jax | `nurbs_curve.NurbsCurve.reparametrize_by_coordinate` | JAX-only | Inverse-map construction and independent inverse accuracy checks. |
| jax | `nurbs_curve.NurbsCurve.project_points` | Direct / adapter | Projection workload uses native batches for JAX and loops over NumPy scalar calls. |
| jax | `nurbs_curve.NurbsCurve.plot` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_curve.NurbsCurve.plot_curve` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_curve.NurbsCurve.plot_control_points` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_curve.NurbsCurve.plot_frenet_serret` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_curve.NurbsCurve.rescale_plot` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_curve.NurbsCurve.plot_curvature` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_curve.NurbsCurve.plot_torsion` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_curve.merge_nurbs_curves` | Direct | construction workloads. |
| jax | `nurbs_curve_circular_arc.CircularArc` | Direct | Constructed by deterministic geometry/generator workloads. |
| jax | `nurbs_curve_circular_arc.CircularArc.make_nurbs_circular_arc` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| jax | `nurbs_curve_circular_arc.CircularArc.intersect_lines` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| jax | `nurbs_curve_circular_arc.CircularArc.plot` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_surface.compute_bspline_coordinates` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| jax | `nurbs_surface.compute_nurbs_coordinates` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| jax | `nurbs_surface.compute_bspline_derivatives` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| jax | `nurbs_surface.compute_nurbs_derivatives` | Indirect | Exercised by public geometry methods; polynomial/rational helper variants not individually timed. |
| jax | `nurbs_surface.NurbsSurface` | Direct | Constructed by deterministic geometry/generator workloads. |
| jax | `nurbs_surface.NurbsSurface.U_values` | Not separately exercised | Listed explicitly rather than assuming coverage. |
| jax | `nurbs_surface.NurbsSurface.U_mults` | Not separately exercised | Listed explicitly rather than assuming coverage. |
| jax | `nurbs_surface.NurbsSurface.V_values` | Not separately exercised | Listed explicitly rather than assuming coverage. |
| jax | `nurbs_surface.NurbsSurface.V_mults` | Not separately exercised | Listed explicitly rather than assuming coverage. |
| jax | `nurbs_surface.NurbsSurface.get_value` | Direct | Curve/surface numerical workloads and independent references. |
| jax | `nurbs_surface.NurbsSurface.get_derivative` | Direct | Curve/surface numerical workloads and independent references. |
| jax | `nurbs_surface.NurbsSurface.get_normals` | Direct | Curve/surface numerical workloads and independent references. |
| jax | `nurbs_surface.NurbsSurface.get_curvature` | Direct | Curve/surface numerical workloads and independent references. |
| jax | `nurbs_surface.NurbsSurface.get_isocurve_u` | Direct | construction workloads. |
| jax | `nurbs_surface.NurbsSurface.get_isocurve_v` | Direct | construction workloads. |
| jax | `nurbs_surface.NurbsSurface.attach_nurbs_udir` | Direct | construction workloads. |
| jax | `nurbs_surface.NurbsSurface.attach_nurbs_vdir` | Direct | construction workloads. |
| jax | `nurbs_surface.NurbsSurface.project_point_to_surface` | Direct / adapter | Projection workload uses native batches for JAX and loops over NumPy scalar calls. |
| jax | `nurbs_surface.NurbsSurface.project_points` | Direct / adapter | Projection workload uses native batches for JAX and loops over NumPy scalar calls. |
| jax | `nurbs_surface.NurbsSurface.plot` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_surface.NurbsSurface.plot_surface` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_surface.NurbsSurface.plot_boundary` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_surface.NurbsSurface.plot_isocurve_u` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_surface.NurbsSurface.plot_isocurve_v` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_surface.NurbsSurface.plot_control_points` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_surface.NurbsSurface.plot_normals` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_surface.NurbsSurface.rescale_plot` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_surface.NurbsSurface.plot_curvature` | Display smoke only | Curve/surface default rendering; individual styling and plot helpers not separately timed. |
| jax | `nurbs_surface.make_offset_surface` | Direct | construction workloads. |
| jax | `nurbs_surface_bilinear.NurbsSurfaceBilinear` | Direct | Constructed by deterministic geometry/generator workloads. |
| jax | `nurbs_surface_bilinear.NurbsSurfaceBilinear.make_nurbs_surface` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| jax | `nurbs_surface_coons.NurbsSurfaceCoons` | Direct | Constructed by deterministic geometry/generator workloads. |
| jax | `nurbs_surface_coons.NurbsSurfaceCoons.make_nurbs_surface` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| jax | `nurbs_surface_extruded.NurbsSurfaceExtruded` | Direct | Constructed by deterministic geometry/generator workloads. |
| jax | `nurbs_surface_extruded.NurbsSurfaceExtruded.make_Nurbs_surface` | Not separately exercised | Listed explicitly rather than assuming coverage. |
| jax | `nurbs_surface_revolution.NurbsSurfaceRevolution` | Direct | Constructed by deterministic geometry/generator workloads. |
| jax | `nurbs_surface_revolution.NurbsSurfaceRevolution.make_nurbs_surface` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| jax | `nurbs_surface_revolution.NurbsSurfaceRevolution.project_point_to_line` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| jax | `nurbs_surface_revolution.NurbsSurfaceRevolution.intersect_lines` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| jax | `nurbs_surface_ruled.NurbsSurfaceRuled` | Direct | Constructed by deterministic geometry/generator workloads. |
| jax | `nurbs_surface_ruled.NurbsSurfaceRuled.make_nurbs_surface` | Construction / indirect | Called by tested constructors; degenerate helper inputs are outside this study. |
| jax | `prepared.PreparedCurveBasis` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared.PreparedCurveBasis.get_value` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared.PreparedCurveBasis.get_derivatives` | Indirect | Prepared individual derivative methods exercise the same derivative tables. |
| jax | `prepared.PreparedCurveBasis.get_derivative` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared.PreparedCurveBasis.get_tangent` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared.PreparedCurveBasis.get_curvature` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared.PreparedSurfaceBasis` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared.PreparedSurfaceBasis.get_value` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared.PreparedSurfaceBasis.get_derivatives` | Indirect | Prepared individual derivative methods exercise the same derivative tables. |
| jax | `prepared.PreparedSurfaceBasis.get_derivative` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared.PreparedSurfaceBasis.get_normals` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared.PreparedSurfaceBasis.get_curvature` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared.prepare_curve` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared.prepare_surface` | JAX-only | prepared curve/surface derivative, curvature, value and grid workloads. |
| jax | `prepared_projection.PreparedSurfaceProjection` | Not separately exercised | Listed explicitly rather than assuming coverage. |
| jax | `prepared_projection.PreparedSurfaceProjection.project_point` | JAX-only | prepared_projection.batch; scalar adapter not separately timed. |
| jax | `prepared_projection.PreparedSurfaceProjection.project_points` | JAX-only | prepared_projection.batch; scalar adapter not separately timed. |
| jax | `prepared_projection.prepare_surface_projection` | JAX-only | prepared_projection.batch; scalar adapter not separately timed. |
