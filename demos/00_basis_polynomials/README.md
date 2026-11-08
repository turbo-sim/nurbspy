# 00 — Basis polynomials

```powershell
poetry run python demos/00_basis_polynomials/01_basis_polynomials.py
```

Start with basis functions and first/second derivatives in a single Matplotlib
figure with three subplots and a shared legend. `nrb.set_plot_options()` supplies
the standard fonts, colours, ticks, grid, and line widths. Change `degree`,
`control_count`, or `knots` in `main()` to explore other bases.

Running the demo displays the figure without writing files. Call
`main(show=False)` to return the Matplotlib figure for use in other Python code.

Continue with [curves](../01_curves/README.md).
