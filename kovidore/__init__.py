"""KoVidore Benchmark Package"""

__version__ = "0.1.0"

from .main import main
from .evaluate import run_benchmark, ALL_TASKS

__all__ = ["main", "run_benchmark", "ALL_TASKS"]