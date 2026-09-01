import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statistical_utils import benjamini_hochberg, delong_roc_test


def test_bh_adjusted_values_are_monotone_in_rank():
    p_values = np.array([0.20, 0.01, 0.15, 0.02])
    reject, adjusted = benjamini_hochberg(p_values, alpha=0.05)
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert np.allclose(adjusted, [0.20, 0.04, 0.20, 0.04])
    assert reject.tolist() == [False, True, False, True]


def test_delong_is_symmetric_under_model_swap():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model_a = np.array([0.1, 0.4, 0.3, 0.2, 0.9, 0.8, 0.6, 0.7])
    model_b = np.array([0.1, 0.5, 0.4, 0.2, 0.7, 0.6, 0.8, 0.5])
    z_ab, p_ab, auc_a, auc_b = delong_roc_test(y_true, model_a, model_b)
    z_ba, p_ba, auc_b_again, auc_a_again = delong_roc_test(y_true, model_b, model_a)
    assert np.isclose(z_ab, -z_ba)
    assert np.isclose(p_ab, p_ba)
    assert np.isclose(auc_a, auc_a_again)
    assert np.isclose(auc_b, auc_b_again)

