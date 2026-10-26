"""Run published examples and verify the geometric properties they illustrate."""

from pathlib import Path
import runpy
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose


def test_documented_curvature_continuity():
    """Verify that the published two-curve example has a G2-continuous join.

    Execute the actual documentation demo and compare the joining positions,
    unit tangents, and curvature vectors. Matching scalar curvature alone
    would miss a change in bending direction, so compare the full vectors.
    Also check that the endpoint parameter speeds differ: the example is meant
    to illustrate geometric continuity without requiring C1 parametrization.

    Suppress the demo's display call and close its figures so this check can
    run unattended. The same function can be called directly during debugging
    without needing pytest to supply a fixture.
    """
    script = (
        Path(__file__).resolve().parents[2]
        / "demos/documentation/curvature_continuity.py"
    )
    try:
        with patch.object(plt, "show", return_value=None):
            example = runpy.run_path(str(script))
        assert_allclose(
            example["left"].get_value(1.0), example["right"].get_value(0.0), atol=1e-14
        )
        assert_allclose(example["ta"], example["tb"], atol=1e-14)
        assert_allclose(example["ka"], example["kb"], atol=1e-14)
        assert not np.isclose(example["sa"], example["sb"])
    finally:
        plt.close("all")
