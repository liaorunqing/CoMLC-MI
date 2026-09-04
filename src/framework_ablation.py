#!/usr/bin/env python3
"""Retired synthetic "framework ablation" entry point.

The earlier version assembled a table from unrelated outputs and hard-coded
claims about LDS-guided model selection, SHAP validation, an aggregate DeLong
statistic, and four significant labels. That table was not a model ablation and
is not part of the revised analysis.

Use these auditable analyses instead:

* ``python src/comlc_mi_opt.py`` for the 18-configuration neural ablation;
* ``python src/final_statistical_tests.py`` for paired per-label inference;
* ``python src/calibration_analysis.py`` for the calibration audit.
"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "framework_ablation.py is retired because its former output combined "
        "non-comparable components and unsupported hard-coded claims. Run "
        "comlc_mi_opt.py, final_statistical_tests.py, and calibration_analysis.py "
        "to reproduce the revised evidence instead."
    )


if __name__ == "__main__":
    main()
