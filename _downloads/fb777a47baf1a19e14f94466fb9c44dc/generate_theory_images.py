"""Generate numerical theory illustrations: PNG (300 dpi), SVG and PDF.

Run ``python docs/generate_theory_images.py`` in the project environment with
the JAX and documentation dependencies installed. Use ``--only frames`` to
regenerate one topic. Every plot uses actual nurbspy geometry/operations;
independent analytic/SciPy checks are run before completion. No timing study
or package-source changes are involved.
"""

import argparse
import importlib
import json
from pathlib import Path

from theory_figures.common import configure


TOPICS = ("arclength", "parameterization", "frames", "isocurves",
          "surface_integrals", "surface_geometry", "projection")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "source/theory/images"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=TOPICS,
                        help="Generate selected topics only.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "validation.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    for topic in args.only or TOPICS:
        print(f"Generating {topic} ...", flush=True)
        module = importlib.import_module(f"theory_figures.{topic}")
        metrics[topic] = module.generate(args.output_dir)
        print(json.dumps(metrics[topic], indent=2), flush=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Figures and validation results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
