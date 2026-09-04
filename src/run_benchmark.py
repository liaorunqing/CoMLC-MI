"""Canonical entry point for the CoMLC-MI benchmark.

Run with::

    python -m src.run_benchmark --config configs/comlc_mi_benchmark.json

The implementation is provided by :mod:`src.benchmark_pipeline`.
"""

from .benchmark_pipeline import main


if __name__ == "__main__":
    main()

