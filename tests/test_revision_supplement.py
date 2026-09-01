from src.generate_r1_supplement import esc, number


def test_latex_cell_escaping_and_missing_values():
    assert esc("A_B & C") == r"A\_B \& C"
    assert number(float("nan")) == r"\NA"
    assert esc(number(float("nan"))) == r"\NA"


def test_latex_numeric_format_is_deterministic():
    assert number(0.123456, 4) == "0.1235"
