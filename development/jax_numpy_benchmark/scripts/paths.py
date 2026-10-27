"""One layout for readable reports, source scripts and ignored local output."""
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STUDY = SCRIPTS.parent
ROOT = STUDY.parents[1]
OUTPUTS = STUDY / "outputs"
RESULTS = OUTPUTS / "results"
LOGS = OUTPUTS / "logs"
FIGURES = OUTPUTS / "figures"
QA = OUTPUTS / "qa"
CACHE = OUTPUTS / "cache"


def ensure_outputs():
    for directory in (RESULTS, LOGS, FIGURES, QA, CACHE):
        directory.mkdir(parents=True, exist_ok=True)
