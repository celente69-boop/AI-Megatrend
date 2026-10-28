"""Focused unit tests for intrinsic_value_calculator.py."""
from __future__ import annotations

import pandas as pd
import pytest

import intrinsic_value_calculator as iv


TEST_INPUTS = {
    "EPS_GROWTH": 18.4,
    "SECTOR_PE": 22.1,
    "BETA": 1.12,
    "MKT_RETURN": 5.9,
    "TREASURY": 49.88,
    "STOCK_PRICE": 182.5,
    "STOCK_EPS": 6.41,
    "YEARS": 5,
}


def test_part_c_formula_example_exact():
    result = iv.calculate_intrinsic_value(TEST_INPUTS)
    assert result["proj_growth"] == pytest.approx(16.56)
    assert result["proj_pe"] == pytest.approx(18.785)
    assert result["risk_free"] == pytest.approx(4.988)
    assert result["exp_return"] == pytest.approx(6.00944)
    # Part G's stated ~$210.4 does not follow from its own Part C formulas.
    # The verbatim Part C formula yields $193.505904..., so the correct cent is $193.51.
    assert result["intrinsic_value"] == pytest.approx(193.50590438861792, abs=1e-10)
    assert round(result["intrinsic_value"], 2) == 193.51
    assert result["margin_of_safety"] == pytest.approx(0.05687632335246351)
    assert result["iv_5yr"] == pytest.approx(result["intrinsic_value"])


def test_reduce_value_cap_and_no_reduction():
    assert iv.reduce_value(70) == 40.0
    assert iv.reduce_value(65) == 40.0
    assert iv.reduce_value(4) == 4.0


def test_missing_input_refuses_calculation_and_names_field():
    inputs = dict(TEST_INPUTS)
    inputs["BETA"] = None
    with pytest.raises(iv.MissingInputsError) as error:
        iv.calculate_intrinsic_value(inputs)
    assert error.value.missing == ["BETA"]


def test_treasury_yield_times_ten_then_divided_by_ten():
    inputs = dict(TEST_INPUTS, TREASURY=43.0)
    result = iv.calculate_intrinsic_value(inputs)
    assert result["risk_free"] == pytest.approx(4.3)


def test_negative_growth_and_pe_warn_but_calculate():
    inputs = dict(TEST_INPUTS, EPS_GROWTH=-5, SECTOR_PE=-12)
    result = iv.calculate_intrinsic_value(inputs)
    assert len(result["warnings"]) == 2
    assert result["proj_growth"] == -5
    assert result["proj_pe"] == -12


def test_report_contains_sources_and_required_disclaimer():
    result = iv.calculate_intrinsic_value(TEST_INPUTS)
    sources = {
        field: iv.InputRecord(field, value, f"test source {field}", "2026-10-28 14:03:00 UTC")
        for field, value in TEST_INPUTS.items()
    }
    report = iv.format_report("AAPL", TEST_INPUTS, result, sources)
    assert "[source: test source BETA; fetched 2026-10-28 14:03:00 UTC]" in report
    assert "Intrinsic value (5-yr)    : $193.51" in report
    assert "this program is not affiliated with or endorsed by Schwab." in report


def test_auto_fill_sources_are_labeled_and_timestamped(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker
            self.info = {
                "sector": "Technology", "forwardPE": 22.0, "beta": 1.1,
                "earningsGrowth": 0.25, "trailingEps": 5.0, "currency": "USD",
            }
            self.income_stmt = pd.DataFrame(
                [[2.0, 3.0, 4.0, 5.0, 6.0]],
                index=["Diluted EPS"],
                columns=pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]),
            )
            self.fast_info = {"currency": "USD", "last_price": 100.0}

        def history(self, period="1d", interval="1d", **kwargs):
            idx = pd.DatetimeIndex(["2026-10-28 14:00:00"], tz="UTC")
            close = 4.3 if self.ticker == "^TNX" else 100.0
            return pd.DataFrame({"Close": [close]}, index=idx)

    class FakeYF:
        Ticker = FakeTicker

    monkeypatch.setattr(iv, "yf", FakeYF)
    records, ticker, currency = iv.fetch_auto_inputs("TEST", force_refresh=True)
    assert ticker == "TEST" and currency == "USD"
    assert records["EPS_GROWTH"].source.startswith("yfinance annual income_stmt")
    assert records["EPS_GROWTH"].value == pytest.approx(((6 / 2) ** 0.25 - 1) * 100)
    assert records["SECTOR_PE"].value is None
    assert "NOT FETCHED" in records["SECTOR_PE"].source
    assert "not the sector P/E" in records["SECTOR_PE"].note
    assert records["TREASURY"].value == pytest.approx(43.0)
    assert "4.3000%" in records["TREASURY"].note and "43.0000" in records["TREASURY"].note
    assert records["STOCK_PRICE"].value == 100.0
    for field in ("EPS_GROWTH", "SECTOR_PE", "BETA", "TREASURY", "STOCK_PRICE", "STOCK_EPS"):
        assert records[field].source and records[field].fetched_at
    assert records["MKT_RETURN"].value == 5.9 and records["MKT_RETURN"].fetched_at is None
    assert records["YEARS"].value == 5 and records["YEARS"].fetched_at is None
