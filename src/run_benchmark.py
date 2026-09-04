"""Canonical entry point for the CoMLC-MI benchmark.

Run with::

    python -m src.run_benchmark --config configs/comlc_mi_benchmark.json

The implementation remains in :mod:`src.run_revision` so historical commands
continue to work without maintaining two analysis engines.
"""

from .run_revision import main


if __name__ == "__main__":
    main()
