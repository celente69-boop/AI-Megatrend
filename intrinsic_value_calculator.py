"""Charles Schwab Intrinsic Value Calculator reproduction.

Provides:
  * Auto-fill from free Yahoo Finance data with source/time metadata.
  * Editable values and a Streamlit tab renderer for megatrend.py.
  * Manual CLI prompts and fully non-interactive flags.

Run interactively:
    python intrinsic_value_calculator.py --mode manual
    python intrinsic_value_calculator.py --mode auto --ticker AAPL

Run without prompts:
    python intrinsic_value_calculator.py --mode manual --ticker AAPL \\
      --eps-growth 18.4 --sector-pe 22.1 --beta 1.12 --mkt-return 5.9 \\
      --treasury 49.88 --price 182.5 --stock-eps 6.41 --years 5

Auto-fetch and explicitly accept fetched values in a non-interactive shell:
    python intrinsic_value_calculator.py --mode auto --ticker AAPL --accept-fetched

Dependencies: pip install yfinance pandas (Streamlit is needed only for the app tab).
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # yfinance remains optional for manual-only use
    yf = None


REQUIRED_FIELDS = (
    "EPS_GROWTH",
    "SECTOR_PE",
    "BETA",
    "MKT_RETURN",
    "TREASURY",
    "STOCK_PRICE",
    "STOCK_EPS",
)
INPUT_ORDER = REQUIRED_FIELDS + ("YEARS",)
INPUT_LABELS = {
    "EPS_GROWTH": "EPS growth (5y, GAAP)",
    "SECTOR_PE": "Sector P/E",
    "BETA": "Beta",
    "MKT_RETURN": "Expected market return",
    "TREASURY": "10-yr Treasury (index)",
    "STOCK_PRICE": "Current price",
    "STOCK_EPS": "EPS (TTM)",
    "YEARS": "Years of growth",
}
INPUT_INSTRUCTIONS = {
    "EPS_GROWTH": (
        "Annualized EPS growth for the last five years (GAAP), as a percent (e.g. 25). "
        "On schwab.com: Research → Research Tools → symbol → Stock Summary → Expected Earnings "
        "→ Growth Analysis → Annual → EPS → GAAP."
    ),
    "SECTOR_PE": (
        "Sector P/E. On schwab.com: Research → Research Tools → Stock Compare → symbol "
        "→ Index Comparison → sector P/E."
    ),
    "BETA": "Stock beta. On schwab.com: Research → Stocks → symbol → Summary → Technicals.",
    "MKT_RETURN": (
        "Expected 10-year U.S. large-cap return from Schwab's Long-Term Capital Market "
        "Expectations. Default is 5.9."
    ),
    "TREASURY": (
        "10-year Treasury Index value (yield × 10). On schwab.com: Research → U.S. Markets "
        "→ Indices → 10 Yr Treasury Index. Enter the index directly, or choose yield-percent "
        "below and the calculator will multiply it by 10."
    ),
    "STOCK_PRICE": "Current share price in the same currency as EPS.",
    "STOCK_EPS": "Trailing-twelve-month EPS (TTM), in the same currency as the share price.",
    "YEARS": "Years of projected growth. Default is 5; enter a positive whole number.",
}
DEFAULT_MARKET_RETURN = 5.9
DEFAULT_YEARS = 5
FETCH_CACHE_TTL_SECONDS = 15 * 60
DISCLAIMER = (
    "For personal use only. Not for commercial use. This tool is intended solely to assist in\n"
    "identifying hypothetical equity valuations using limited inputs. No guarantees, implied or\n"
    "otherwise, are made about profitability or risks. This is not investment advice and should not\n"
    "be a primary basis for an investment decision. \"Margin of safety\" here refers only to the price\n"
    "difference between the calculated intrinsic value and the market price; it does not identify\n"
    "price support and is not a measure of safety against losses. May not be appropriate for all\n"
    "investors. Data from third-party sources may be inaccurate, stale, or incomplete. Hypothetical\n"
    "results may be less accurate for stocks with atypically high or low 5-year EPS growth. Past\n"
    "performance is no guarantee of future results. Commissions, transaction costs, and taxes are not\n"
    "considered. Investing involves risk, including loss of principal. Use at your own risk.\n"
    "Calculation methodology adapted from the Charles Schwab Intrinsic Value Calculator (v.9);\n"
    "this program is not affiliated with or endorsed by Schwab."
)


@dataclass(frozen=True)
class InputRecord:
    """A fetched/default/manual field plus provenance shown to the user."""

    field: str
    value: Optional[float]
    source: str
    fetched_at: Optional[str] = None
    note: str = ""


class MissingInputsError(ValueError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__("Missing required inputs: " + ", ".join(missing))


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _finite_number(value: Any) -> Optional[float]:
    try:
        n = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if math.isfinite(n) else None


def _record(field: str, value: Any, source: str, note: str = "") -> InputRecord:
    n = _finite_number(value)
    return InputRecord(field, n, source, _now_utc(), note)


def _missing_record(field: str, source: str, error: str) -> InputRecord:
    return InputRecord(field, None, source, _now_utc(), error or "Fetch did not return a usable value; enter manually.")


def reduce_value(x: float) -> float:
    """Workbook's nested-IF reduction, reproduced verbatim (x is a percent number)."""
    if x >= 64:
        return 40.0
    elif x >= 60:
        return x * 0.65
    elif x >= 55:
        return x * 0.675
    elif x >= 50:
        return x * 0.70
    elif x >= 45:
        return x * 0.725
    elif x >= 40:
        return x * 0.75
    elif x >= 35:
        return x * 0.775
    elif x >= 30:
        return x * 0.80
    elif x >= 20:
        return x * 0.85
    elif x >= 12:
        return x * 0.90
    elif x >= 6:
        return x * 0.95
    else:
        return x


def _validated_values(values: dict[str, Any]) -> tuple[dict[str, float], int]:
    missing: list[str] = []
    numeric: dict[str, float] = {}
    for field in REQUIRED_FIELDS:
        v = _finite_number(values.get(field))
        if v is None:
            missing.append(field)
        else:
            numeric[field] = v
    years_raw = _finite_number(values.get("YEARS"))
    if years_raw is None:
        missing.append("YEARS")
        years = DEFAULT_YEARS
    elif years_raw <= 0 or not years_raw.is_integer():
        raise ValueError("YEARS must be a positive whole number.")
    else:
        years = int(years_raw)
    if missing:
        raise MissingInputsError(missing)
    return numeric, years


def calculate_intrinsic_value(values: dict[str, Any]) -> dict[str, Any]:
    """Calculate workbook outputs exactly from percent-number inputs."""
    v, years = _validated_values(values)
    eps_growth = v["EPS_GROWTH"]
    sector_pe = v["SECTOR_PE"]
    beta = v["BETA"]
    market_return = v["MKT_RETURN"]
    treasury = v["TREASURY"]
    stock_price = v["STOCK_PRICE"]
    stock_eps = v["STOCK_EPS"]

    # Workbook O9, O13, K19, O16.
    proj_growth = reduce_value(eps_growth)
    proj_pe = reduce_value(sector_pe)
    risk_free = treasury / 10.0
    exp_return = risk_free + beta * (market_return - risk_free)

    discount_base = 1.0 + exp_return / 100.0
    if discount_base == 0:
        raise ValueError("CAPM expected return produces a zero discount base; IV cannot be calculated.")

    # Workbook E28 / K28. Percent numbers are divided by 100 only in exponents.
    future_eps = stock_eps * (1.0 + proj_growth / 100.0) ** years
    future_price = future_eps * proj_pe
    intrinsic_value = future_price / discount_base ** years
    margin_of_safety = (
        (intrinsic_value - stock_price) / intrinsic_value
        if intrinsic_value != 0
        else None
    )

    # Workbook E30 / F30; sensitivity is always displayed at 5 and 10 years.
    iv_5yr = (
        stock_eps * (1.0 + proj_growth / 100.0) ** 5 * proj_pe
    ) / (1.0 + exp_return / 100.0) ** 5
    iv_10yr = (
        stock_eps * (1.0 + proj_growth / 100.0) ** 10 * proj_pe
    ) / (1.0 + exp_return / 100.0) ** 10

    warnings: list[str] = []
    if eps_growth < 0:
        warnings.append("EPS_GROWTH is negative; the workbook reduction function passes it through unchanged.")
    if sector_pe < 0:
        warnings.append("SECTOR_PE is negative; the workbook reduction function passes it through unchanged.")
    if intrinsic_value == 0:
        warnings.append("Margin of safety is undefined because intrinsic value is zero.")
    return {
        "inputs": {**v, "YEARS": years},
        "proj_growth": proj_growth,
        "proj_pe": proj_pe,
        "risk_free": risk_free,
        "exp_return": exp_return,
        "future_eps": future_eps,
        "future_price": future_price,
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety,
        "iv_5yr": iv_5yr,
        "iv_10yr": iv_10yr,
        "warnings": warnings,
    }


def _currency_code(info: dict[str, Any], ticker_obj: Any = None) -> str:
    code = info.get("currency") if isinstance(info, dict) else None
    if not code and ticker_obj is not None:
        try:
            fast = ticker_obj.fast_info
            code = fast.get("currency") if hasattr(fast, "get") else getattr(fast, "currency", None)
        except Exception:
            pass
    return str(code).upper() if code else "USD"


def _five_year_diluted_eps_cagr(ticker_obj: Any) -> tuple[float, str]:
    """CAGR from five annual Diluted EPS points (four elapsed intervals)."""
    statement = ticker_obj.income_stmt
    if statement is None or getattr(statement, "empty", True):
        raise ValueError("Yahoo annual income statement is unavailable.")
    row_name = next(
        (idx for idx in statement.index if str(idx).strip().lower() == "diluted eps"),
        None,
    )
    if row_name is None:
        row_name = next(
            (idx for idx in statement.index if "diluted eps" in str(idx).strip().lower()),
            None,
        )
    if row_name is None:
        raise ValueError("Annual income statement has no Diluted EPS row.")
    row = statement.loc[row_name]
    points: list[tuple[pd.Timestamp, float]] = []
    for column, raw in row.items():
        date = pd.to_datetime(column, errors="coerce", utc=True)
        value = _finite_number(raw)
        if pd.isna(date) or value is None:
            continue
        points.append((pd.Timestamp(date), value))
    points.sort(key=lambda x: x[0])
    if len(points) < 5:
        raise ValueError(f"Need 5 annual Diluted EPS values; Yahoo returned {len(points)}.")
    points = points[-5:]
    old_date, old_eps = points[0]
    latest_date, latest_eps = points[-1]
    if old_eps <= 0 or latest_eps <= 0:
        raise ValueError("5-year Diluted EPS CAGR is undefined when either endpoint EPS is zero/negative.")
    cagr_pct = ((latest_eps / old_eps) ** (1.0 / 4.0) - 1.0) * 100.0
    source_note = (
        f"Annual Diluted EPS {old_date.strftime('%Y')}→{latest_date.strftime('%Y')}; "
        "four elapsed annual intervals."
    )
    return cagr_pct, source_note


def _last_close(history: Any) -> tuple[Optional[float], Optional[str]]:
    if history is None or getattr(history, "empty", True) or "Close" not in history:
        return None, None
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if close.empty:
        return None, None
    index = close.index[-1]
    try:
        ts = pd.Timestamp(index)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        label = ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        label = str(index)
    return _finite_number(close.iloc[-1]), label


_FETCH_CACHE: dict[str, tuple[float, dict[str, InputRecord], str]] = {}
_FETCH_CACHE_LOCK = threading.Lock()


def fetch_auto_inputs(ticker: str, *, force_refresh: bool = False) -> tuple[dict[str, InputRecord], str, str]:
    """Fetch workbook fields from Yahoo Finance; cache per process for 15 minutes.

    Returns (records by field, normalized ticker, currency code). Any unavailable
    field has value=None and an explicit FETCH FAILED source/note.
    """
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        raise ValueError("Enter a ticker symbol before fetching.")
    now_mono = time.monotonic()
    with _FETCH_CACHE_LOCK:
        cached = _FETCH_CACHE.get(symbol)
        if cached and not force_refresh and now_mono - cached[0] < FETCH_CACHE_TTL_SECONDS:
            return dict(cached[1]), symbol, cached[2]
    if yf is None:
        raise RuntimeError("yfinance is not installed. Install it with: pip install yfinance")

    records: dict[str, InputRecord] = {}
    try:
        ticker_obj = yf.Ticker(symbol)
    except Exception as exc:
        ticker_obj = None
        ticker_error = f"{type(exc).__name__}: {exc}"
    else:
        ticker_error = ""

    info: dict[str, Any] = {}
    info_error = ticker_error
    if ticker_obj is not None:
        try:
            raw_info = ticker_obj.info
            info = raw_info if isinstance(raw_info, dict) else {}
            if not info:
                info_error = "Yahoo returned an empty info response."
        except Exception as exc:
            info_error = f"{type(exc).__name__}: {exc}"
    sector = str(info.get("sector") or "sector not reported")
    currency = _currency_code(info, ticker_obj)

    # 1) EPS growth: prefer five annual GAAP diluted-EPS observations.
    try:
        cagr, note = _five_year_diluted_eps_cagr(ticker_obj) if ticker_obj is not None else (None, "")
        if cagr is None:
            raise ValueError("5-year EPS CAGR unavailable.")
        records["EPS_GROWTH"] = _record(
            "EPS_GROWTH", cagr, "yfinance annual income_stmt['Diluted EPS'] CAGR", note
        )
    except Exception as cagr_exc:
        growth = _finite_number(info.get("earningsGrowth"))
        if growth is not None:
            records["EPS_GROWTH"] = _record(
                "EPS_GROWTH",
                growth * 100.0,
                "yfinance info['earningsGrowth'] × 100 (1-year proxy)",
                "Fallback only: earningsGrowth is usually 1-year YoY, not the requested 5-year GAAP annualized EPS growth. Override recommended. "
                f"5-year CAGR unavailable: {cagr_exc}",
            )
        else:
            records["EPS_GROWTH"] = _missing_record(
                "EPS_GROWTH", "FETCH FAILED — annual Diluted EPS CAGR and info.earningsGrowth unavailable",
                f"{cagr_exc}; info error: {info_error or 'earningsGrowth not reported'}",
            )

    # 2) Do NOT substitute company-level forwardPE/trailingPE for the
    # workbook's sector-level P/E on current-year EPS. Yahoo does not expose
    # the same sector metric, and the difference can materially change IV.
    records["SECTOR_PE"] = _missing_record(
        "SECTOR_PE",
        "NOT FETCHED — Yahoo does not provide Schwab's sector P/E",
        f"Yahoo reports the company sector as {sector}, but its forwardPE/trailingPE are company-level ratios, not the sector P/E. Enter Schwab's current-year sector P/E manually.",
    )

    # 3) Beta.
    beta = _finite_number(info.get("beta"))
    records["BETA"] = (
        _record("BETA", beta, "yfinance info['beta']")
        if beta is not None
        else _missing_record("BETA", "FETCH FAILED — yfinance info['beta']", info_error or "Beta not reported.")
    )

    # 4) Market return is a stated workbook default; never fetched.
    records["MKT_RETURN"] = InputRecord(
        "MKT_RETURN", DEFAULT_MARKET_RETURN, "Default: Schwab LTCME (editable; not fetched)", None,
        "Confirm or override this workbook default.",
    )

    # 5) Yahoo ^TNX Close is a yield in percent per the workbook specification;
    # multiply by ten for the workbook's Treasury index input.
    try:
        tnx = yf.Ticker("^TNX")
        treasury_history = tnx.history(period="5d", interval="1d")
        yield_pct, bar_time = _last_close(treasury_history)
        if yield_pct is None:
            raise ValueError("Yahoo ^TNX returned no usable Close value.")
        treasury_index = yield_pct * 10.0
        records["TREASURY"] = _record(
            "TREASURY", treasury_index, "yfinance ^TNX history Close (yield % × 10)",
            f"10-year yield={yield_pct:.4f}%; workbook index={treasury_index:.4f}; latest bar={bar_time or 'timestamp unavailable'}.",
        )
    except Exception as exc:
        records["TREASURY"] = _missing_record(
            "TREASURY", "FETCH FAILED — yfinance ^TNX history Close",
            f"{type(exc).__name__}: {exc}. Enter the index (yield × 10), or enter a yield percent and convert it.",
        )

    # 6) Current stock price: intraday close first, then daily close, then fast_info.
    price: Optional[float] = None
    price_source = ""
    price_note = ""
    if ticker_obj is not None:
        for period, interval in (("1d", "1m"), ("5d", "1d")):
            try:
                hist = ticker_obj.history(period=period, interval=interval, auto_adjust=False)
                price, bar_time = _last_close(hist)
                if price is not None:
                    price_source = f"yfinance Ticker.history(period='{period}', interval='{interval}') Close"
                    price_note = f"latest bar={bar_time or 'timestamp unavailable'}; quote currency={currency}."
                    break
            except Exception as exc:
                price_note = f"History fetch issue ({period}/{interval}): {type(exc).__name__}: {exc}."
        if price is None:
            try:
                fast = ticker_obj.fast_info
                raw = fast.get("last_price") if hasattr(fast, "get") else getattr(fast, "last_price", None)
                price = _finite_number(raw)
                if price is not None:
                    price_source = "yfinance Ticker.fast_info['last_price']"
                    price_note = f"Yahoo fast_info exposes no reliable bar timestamp; fetched at request time. Currency={currency}."
            except Exception as exc:
                price_note = f"fast_info fetch issue: {type(exc).__name__}: {exc}."
    records["STOCK_PRICE"] = (
        _record("STOCK_PRICE", price, price_source, price_note)
        if price is not None
        else _missing_record("STOCK_PRICE", "FETCH FAILED — history Close and fast_info.last_price", price_note or info_error or "No quote returned.")
    )

    # 7) TTM EPS.
    stock_eps = _finite_number(info.get("trailingEps"))
    records["STOCK_EPS"] = (
        _record("STOCK_EPS", stock_eps, "yfinance info['trailingEps']")
        if stock_eps is not None
        else _missing_record("STOCK_EPS", "FETCH FAILED — yfinance info['trailingEps']", info_error or "Trailing EPS not reported.")
    )

    # 8) Projection horizon is a stated workbook default; never fetched.
    records["YEARS"] = InputRecord(
        "YEARS", float(DEFAULT_YEARS), "Default: workbook horizon (editable; not fetched)", None,
        "Confirm or change the projected-growth years.",
    )

    with _FETCH_CACHE_LOCK:
        _FETCH_CACHE[symbol] = (time.monotonic(), dict(records), currency)
    return records, symbol, currency


def format_number(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def format_money(value: Optional[float], currency: str = "USD") -> str:
    if value is None:
        return "—"
    code = (currency or "USD").upper()
    if code == "USD":
        return f"${value:,.2f}"
    return f"{code} {value:,.2f}"


def _record_source_text(record: Optional[InputRecord]) -> str:
    if record is None:
        return "source: manual entry"
    text = f"source: {record.source}"
    if record.fetched_at:
        text += f"; fetched {record.fetched_at}"
    if record.note:
        text += f"; {record.note}"
    return text


def format_report(
    ticker: str,
    values: dict[str, Any],
    result: dict[str, Any],
    sources: Optional[dict[str, InputRecord]] = None,
    currency: str = "USD",
) -> str:
    """Format the Part E report and print the required disclaimer every time."""
    sources = sources or {}
    v = result["inputs"]

    def line(field: str, label: str, value: str) -> str:
        record = sources.get(field)
        source = f"[{_record_source_text(record)}]" if record else "[source: manual entry]"
        return f"  {label:<28}: {value:<18} {source}"

    years = int(v["YEARS"])
    mos = result["margin_of_safety"]
    if mos is None:
        mos_text = "undefined (intrinsic value is zero)"
    else:
        mos_pct = mos * 100.0
        if mos_pct > 0:
            mos_text = f"{mos_pct:+.2f} %       (price is {mos_pct:.2f}% below intrinsic value)"
        elif mos_pct < 0:
            mos_text = f"{mos_pct:.2f} %       (price is {abs(mos_pct):.2f}% above intrinsic value)"
        else:
            mos_text = "0.00 %        (price equals intrinsic value)"

    output = [
        f"=== Intrinsic Value Calculator — {ticker or 'MANUAL'} ===",
        "Inputs",
        line("EPS_GROWTH", INPUT_LABELS["EPS_GROWTH"], f"{v['EPS_GROWTH']:.2f} %"),
        line("SECTOR_PE", INPUT_LABELS["SECTOR_PE"], f"{v['SECTOR_PE']:.2f}"),
        line("BETA", INPUT_LABELS["BETA"], f"{v['BETA']:.2f}"),
        line("MKT_RETURN", INPUT_LABELS["MKT_RETURN"], f"{v['MKT_RETURN']:.2f} %"),
        line("TREASURY", INPUT_LABELS["TREASURY"], f"{v['TREASURY']:.2f}"),
        line("STOCK_PRICE", INPUT_LABELS["STOCK_PRICE"], format_money(v["STOCK_PRICE"], currency)),
        line("STOCK_EPS", INPUT_LABELS["STOCK_EPS"], f"{v['STOCK_EPS']:.2f}"),
        line("YEARS", INPUT_LABELS["YEARS"], f"{years}"),
        "",
        "Derived",
        f"  Reduced projected growth  : {result['proj_growth']:.2f} % (from {v['EPS_GROWTH']:.2f})",
        f"  Reduced projected P/E     : {result['proj_pe']:.2f} (from {v['SECTOR_PE']:.2f})",
        f"  Risk-free rate            : {result['risk_free']:.2f} %",
        f"  CAPM expected return      : {result['exp_return']:.2f} %",
        "",
        "Results",
        f"  Intrinsic value ({years}-yr)    : {format_money(result['intrinsic_value'], currency)}",
        f"  Current price             : {format_money(v['STOCK_PRICE'], currency)}",
        f"  Margin of safety          : {mos_text}",
        "",
        "Sensitivity",
        f"  Intrinsic value @ 5 yrs   : {format_money(result['iv_5yr'], currency)}",
        f"  Intrinsic value @ 10 yrs  : {format_money(result['iv_10yr'], currency)}",
    ]
    if result.get("warnings"):
        output.extend(["", "Warnings"])
        output.extend(f"  - {warning}" for warning in result["warnings"])
    output.extend(["", DISCLAIMER])
    return "\n".join(output)


def _display_value(value: Optional[float], field: str) -> str:
    if value is None:
        return "—"
    if field == "YEARS":
        return str(int(value)) if float(value).is_integer() else format_number(value)
    if field == "BETA":
        return format_number(value, 3)
    return format_number(value, 2)


def _record_rows(records: dict[str, InputRecord]) -> list[dict[str, str]]:
    rows = []
    for field in INPUT_ORDER:
        record = records.get(field)
        if record is None:
            continue
        rows.append({
            "Field": field,
            "Auto-fill / default": _display_value(record.value, field),
            "Source": record.source,
            "Fetched at (UTC)": record.fetched_at or "— (default, not fetched)",
            "Note": record.note,
        })
    return rows


def _help_for_field(field: str, record: Optional[InputRecord] = None) -> str:
    help_text = INPUT_INSTRUCTIONS[field]
    if record and record.note:
        help_text += "\n\nAuto-fill note: " + record.note
    return help_text


def _as_input_record_map(records: Any) -> dict[str, InputRecord]:
    if not isinstance(records, dict):
        return {}
    out: dict[str, InputRecord] = {}
    for field, raw in records.items():
        if isinstance(raw, InputRecord):
            out[field] = raw
        elif isinstance(raw, dict):
            try:
                out[field] = InputRecord(**raw)
            except Exception:
                continue
    return out


def render_intrinsic_value_tab() -> None:
    """Streamlit UI rendered inside Mega Trend's Intrinsic Value tab."""
    import streamlit as st

    st.subheader("🧮 Intrinsic Value Calculator")
    st.caption("Workbook-style intrinsic value estimate. Auto-filled values are editable; missing data is never guessed.")
    st.info(
        "Yahoo values can be stale, missing, or wrong—especially for small caps and non-U.S. tickers. "
        "Yahoo does not provide Schwab's sector P/E, so that field is intentionally left blank for manual entry. "
        "Yahoo earningsGrowth is often 1-year YoY, not the requested 5-year GAAP CAGR. Treasury is stored as yield × 10."
    )

    mode = st.radio(
        "Input mode",
        ["Auto — fetch a ticker and review inputs", "Manual — enter the values yourself"],
        horizontal=True,
        key="iv_mode",
    )
    is_auto = mode.startswith("Auto")
    currency = "USD"
    ticker = "MANUAL"
    records: dict[str, InputRecord] = {}
    values: dict[str, Optional[float]] = {}
    source_map: dict[str, InputRecord] = {}

    if is_auto:
        ticker_entry = st.text_input("Ticker", key="iv_auto_ticker_entry", placeholder="e.g. AAPL")
        if st.button("Fetch Yahoo inputs (15-minute cache)", key="iv_auto_fetch"):
            if not ticker_entry.strip():
                st.error("Enter a ticker first.")
            else:
                with st.spinner(f"Fetching Yahoo Finance inputs for {ticker_entry.strip().upper()}…"):
                    try:
                        records, normalized, currency = fetch_auto_inputs(ticker_entry, force_refresh=False)
                        st.session_state["iv_auto_records"] = records
                        st.session_state["iv_auto_loaded_ticker"] = normalized
                        st.session_state["iv_auto_currency"] = currency
                        for field in INPUT_ORDER:
                            key = f"iv_auto_{field.lower()}"
                            st.session_state[key] = records.get(field, InputRecord(field, None, "FETCH FAILED")).value
                    except Exception as exc:
                        st.error(f"Fetch failed: {type(exc).__name__}: {exc}. You can still use Manual mode.")
        records = _as_input_record_map(st.session_state.get("iv_auto_records"))
        loaded_ticker = str(st.session_state.get("iv_auto_loaded_ticker") or "")
        ticker = loaded_ticker or ticker_entry.strip().upper() or "MANUAL"
        currency = str(st.session_state.get("iv_auto_currency") or "USD")
        if records:
            if ticker_entry.strip().upper() != loaded_ticker:
                st.warning(f"Loaded inputs are for {loaded_ticker or 'no ticker'}. Fetch again before calculating for a different ticker.")
            st.markdown("**Fetched/default inputs — review these values, then edit any number below before calculating.**")
            st.dataframe(pd.DataFrame(_record_rows(records)), width="stretch", hide_index=True)
            for field in INPUT_ORDER:
                rec = records.get(field)
                if rec and rec.value is None:
                    st.warning(f"{field}: {rec.source}. Enter this value manually below. {rec.note}")
                elif rec and rec.note and (field in ("EPS_GROWTH", "SECTOR_PE", "TREASURY")):
                    st.caption(f"{field}: {rec.note}")
            for field in INPUT_ORDER:
                rec = records.get(field)
                key = f"iv_auto_{field.lower()}"
                val = st.session_state.get(key)
                kwargs: dict[str, Any] = {
                    "label": f"{field} — {INPUT_LABELS[field]}",
                    "value": None,
                    "key": key,
                    "help": _help_for_field(field, rec),
                    "step": 1 if field == "YEARS" else 0.01,
                }
                if field == "YEARS":
                    kwargs.update(min_value=1, max_value=100, step=1)
                if field == "BETA":
                    kwargs["step"] = 0.01
                st.number_input(**kwargs)
            values = {field: _finite_number(st.session_state.get(f"iv_auto_{field.lower()}")) for field in INPUT_ORDER}
            for field in INPUT_ORDER:
                rec = records.get(field)
                v = values.get(field)
                if rec is None:
                    source_map[field] = InputRecord(field, v, "Manual entry; source metadata unavailable")
                elif v is None:
                    source_map[field] = rec
                elif rec.value is not None and v == rec.value:
                    source_map[field] = rec
                else:
                    origin = rec.source + (f", fetched {rec.fetched_at}" if rec.fetched_at else "")
                    source_map[field] = InputRecord(
                        field, v, f"Manual override (original: {origin})", None,
                        "Value typed/changed by user before calculation.",
                    )
            if st.button("Calculate intrinsic value", key="iv_auto_calculate", type="primary"):
                if ticker_entry.strip().upper() != loaded_ticker:
                    st.error("Fetch inputs for the current ticker before calculating.")
                else:
                    _calculate_and_render(st, ticker, values, source_map, currency)
        else:
            st.caption("Enter a ticker and press Fetch. If Yahoo cannot supply a field, the failed field will be called out and left editable for manual entry.")

    else:
        st.markdown("**Enter the workbook inputs in order. Defaults for market return and years are shown and editable.**")
        manual_keys = {field: f"iv_manual_{field.lower()}" for field in INPUT_ORDER}
        # All fields are blank except the workbook-provided market-return and years defaults.
        defaults: dict[str, Optional[float]] = {
            "EPS_GROWTH": None,
            "SECTOR_PE": None,
            "BETA": None,
            "MKT_RETURN": DEFAULT_MARKET_RETURN,
            "TREASURY": None,
            "STOCK_PRICE": None,
            "STOCK_EPS": None,
            "YEARS": float(DEFAULT_YEARS),
        }
        for field in INPUT_ORDER:
            if manual_keys[field] not in st.session_state:
                st.session_state[manual_keys[field]] = defaults[field]
        for field in INPUT_ORDER:
            if field == "TREASURY":
                st.caption(INPUT_INSTRUCTIONS[field])
                treasury_mode = st.selectbox(
                    "Treasury entry format",
                    ["Workbook index value (yield × 10)", "Yield percentage (calculator multiplies by 10)"],
                    key="iv_manual_treasury_mode",
                    help="The workbook expects the index value. Example: a 4.988% yield becomes 49.88.",
                )
                entry = st.number_input(
                    "TREASURY — 10-year Treasury value",
                    value=None,
                    key=manual_keys[field],
                    step=0.01,
                    help=_help_for_field(field),
                )
                if entry is not None:
                    if treasury_mode.startswith("Yield"):
                        values[field] = float(entry) * 10.0
                        source_map[field] = InputRecord(field, values[field], "Manual yield percent × 10", None,
                                                        f"Entered yield={float(entry):.4f}%; workbook index={values[field]:.4f}.")
                        st.caption(f"Yield {float(entry):.4f}% → workbook Treasury index {values[field]:.4f}.")
                    else:
                        values[field] = float(entry)
                        source_map[field] = InputRecord(field, values[field], "Manual Treasury index value (yield × 10)")
                else:
                    values[field] = None
                    source_map[field] = InputRecord(field, None, "Manual entry")
                continue
            st.caption(INPUT_INSTRUCTIONS[field])
            kwargs = {
                "label": f"{field} — {INPUT_LABELS[field]}", 
                "value": None,
                "key": manual_keys[field],
                "help": _help_for_field(field),
                "step": 1 if field == "YEARS" else 0.01,
            }
            if field == "YEARS":
                kwargs.update(min_value=1, max_value=100, step=1)
            st.number_input(**kwargs)
            value = _finite_number(st.session_state.get(manual_keys[field]))
            values[field] = value
            if value is None:
                source_map[field] = InputRecord(field, None, "Manual entry — missing")
            elif field == "MKT_RETURN" and value == DEFAULT_MARKET_RETURN:
                source_map[field] = InputRecord(field, value, "Default: Schwab LTCME (confirmed or editable)")
            elif field == "YEARS" and value == DEFAULT_YEARS:
                source_map[field] = InputRecord(field, value, "Default: workbook horizon 5 years (confirmed or editable)")
            else:
                source_map[field] = InputRecord(field, value, "Manual entry")
        ticker = st.text_input("Ticker / label for report", key="iv_manual_ticker") or "MANUAL"
        if st.button("Calculate intrinsic value", key="iv_manual_calculate", type="primary"):
            _calculate_and_render(st, ticker.strip().upper(), values, source_map, "USD")


def _calculate_and_render(
    st: Any,
    ticker: str,
    values: dict[str, Optional[float]],
    sources: dict[str, InputRecord],
    currency: str,
) -> None:
    try:
        result = calculate_intrinsic_value(values)
    except MissingInputsError as exc:
        st.error("Cannot calculate. Enter the missing field(s): " + ", ".join(exc.missing))
        return
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        st.error(f"Cannot calculate: {exc}")
        return
    if result["warnings"]:
        for warning in result["warnings"]:
            st.warning(warning)
    report = format_report(ticker, values, result, sources, currency)
    st.markdown("### Results")
    st.code(report, language=None)


def _cli_value(value: Optional[float], field: str) -> str:
    if value is None:
        return "FETCH FAILED — enter manually"
    return f"{value:.4f}" if field in ("BETA", "TREASURY") else f"{value:.2f}"


def _print_fetch_preview(records: dict[str, InputRecord], ticker: str) -> None:
    print(f"Auto-fill preview for {ticker} (all values can be overridden):")
    for field in INPUT_ORDER:
        rec = records[field]
        suffix = f" | fetched {rec.fetched_at}" if rec.fetched_at else ""
        note = f" | {rec.note}" if rec.note else ""
        print(f"  {field:<12} = {_cli_value(rec.value, field)} [source: {rec.source}{suffix}{note}]")


def _parse_cli_float(raw: str, field: str) -> float:
    try:
        val = float(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric; received {raw!r}.") from exc
    if not math.isfinite(val):
        raise ValueError(f"{field} must be a finite number.")
    return val


def _prompt_value(field: str, current: Optional[float], *, default_label: str = "") -> tuple[float, str]:
    label = INPUT_LABELS[field]
    instruction = INPUT_INSTRUCTIONS[field]
    default = f" [{current:g}]" if current is not None else ""
    if default_label:
        print(f"{instruction} {default_label}")
    else:
        print(instruction)
    while True:
        raw = input(f"{field}{default}: ").strip()
        if not raw and current is not None:
            return float(current), "Default/accepted value"
        if not raw:
            print("A value is required; please enter a number.")
            continue
        try:
            return _parse_cli_float(raw, field), "Manual entry"
        except ValueError as exc:
            print(exc)


def _interactive_treasury_prompt() -> tuple[float, str, str]:
    print(INPUT_INSTRUCTIONS["TREASURY"])
    while True:
        fmt = input("Enter as [I]ndex (e.g. 49.88) or [Y]ield percent (e.g. 4.988)? [I]: ").strip().lower() or "i"
        if fmt in ("i", "index"):
            raw = input("TREASURY index value: ").strip()
            try:
                value = _parse_cli_float(raw, "TREASURY")
                return value, "Manual Treasury index value", ""
            except ValueError as exc:
                print(exc)
        elif fmt in ("y", "yield", "percent"):
            raw = input("10-year Treasury yield percent: ").strip()
            try:
                yield_pct = _parse_cli_float(raw, "Treasury yield")
                return yield_pct * 10.0, "Manual yield percent × 10", f"Entered yield={yield_pct:.4f}%; workbook index={yield_pct * 10.0:.4f}."
            except ValueError as exc:
                print(exc)
        else:
            print("Enter I for index or Y for yield percent.")


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Charles Schwab-style Intrinsic Value Calculator")
    parser.add_argument("--mode", choices=("auto", "manual"), default="manual")
    parser.add_argument("--ticker", default="", help="Stock ticker/report label")
    parser.add_argument("--accept-fetched", action="store_true",
                        help="In non-interactive auto mode, explicitly accept the fetched values/proxies after they are printed.")
    parser.add_argument("--eps-growth", type=float, dest="eps_growth")
    parser.add_argument("--sector-pe", type=float, dest="sector_pe")
    parser.add_argument("--beta", type=float)
    parser.add_argument("--mkt-return", type=float, dest="mkt_return")
    treasury = parser.add_mutually_exclusive_group()
    treasury.add_argument("--treasury", type=float, help="Workbook Treasury index value (yield × 10)")
    treasury.add_argument("--treasury-yield", type=float, dest="treasury_yield",
                          help="10-year Treasury yield in percent; program multiplies by 10")
    parser.add_argument("--price", "--stock-price", type=float, dest="stock_price")
    parser.add_argument("--stock-eps", "--eps-ttm", type=float, dest="stock_eps")
    parser.add_argument("--years", type=float, default=None)
    parser.add_argument("--currency", default="USD", help="Currency code for manual mode (default USD)")
    return parser


def cli_main(argv: Optional[list[str]] = None) -> int:
    args = _cli_parser().parse_args(argv)
    interactive = sys.stdin.isatty()
    ticker = args.ticker.strip().upper() or ("MANUAL" if args.mode == "manual" else "")
    values: dict[str, Optional[float]] = {}
    sources: dict[str, InputRecord] = {}
    currency = args.currency.upper()

    if args.mode == "auto":
        if not args.ticker.strip():
            print("--ticker is required with --mode auto.", file=sys.stderr)
            return 2
        try:
            records, ticker, currency = fetch_auto_inputs(ticker, force_refresh=True)
        except Exception as exc:
            print(f"Auto-fetch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not interactive:
                return 2
            records = {field: _missing_record(field, "FETCH FAILED", str(exc)) for field in INPUT_ORDER}
            records["MKT_RETURN"] = InputRecord("MKT_RETURN", DEFAULT_MARKET_RETURN, "Default: Schwab LTCME")
            records["YEARS"] = InputRecord("YEARS", float(DEFAULT_YEARS), "Default: workbook horizon")
        _print_fetch_preview(records, ticker)
        flag_overrides: dict[str, Optional[float]] = {
            "EPS_GROWTH": args.eps_growth,
            "SECTOR_PE": args.sector_pe,
            "BETA": args.beta,
            "MKT_RETURN": args.mkt_return,
            "STOCK_PRICE": args.stock_price,
            "STOCK_EPS": args.stock_eps,
            "YEARS": args.years,
        }
        if args.treasury is not None:
            flag_overrides["TREASURY"] = args.treasury
            sources["TREASURY"] = InputRecord("TREASURY", args.treasury, "CLI override: workbook Treasury index")
        elif args.treasury_yield is not None:
            flag_overrides["TREASURY"] = args.treasury_yield * 10.0
            sources["TREASURY"] = InputRecord(
                "TREASURY", args.treasury_yield * 10.0, "CLI override: yield percent × 10", None,
                f"Entered yield={args.treasury_yield:.4f}%; workbook index={args.treasury_yield * 10.0:.4f}.",
            )
        for field, override in flag_overrides.items():
            if override is not None:
                values[field] = float(override)
                sources[field] = InputRecord(field, float(override), "CLI override")

        if not interactive and not args.accept_fetched:
            unresolved = [f for f in INPUT_ORDER if f not in values and records[f].value is not None]
            if unresolved:
                print("Non-interactive auto mode requires explicit confirmation. Re-run with --accept-fetched "
                      "or supply overrides for: " + ", ".join(unresolved), file=sys.stderr)
                return 2
        for field in INPUT_ORDER:
            if field in values:
                continue
            rec = records[field]
            current = rec.value
            if interactive and not args.accept_fetched:
                instruction = INPUT_INSTRUCTIONS[field]
                print(f"\n{field}: {instruction}")
                src = f"source: {rec.source}" + (f"; fetched {rec.fetched_at}" if rec.fetched_at else "")
                if rec.note:
                    print("  " + src + " — " + rec.note)
                else:
                    print("  " + src)
                while True:
                    default = f" [{current:g}]" if current is not None else ""
                    raw = input("Press Enter to accept or type a replacement: ").strip()
                    if not raw and current is not None:
                        values[field] = float(current)
                        sources[field] = rec
                        break
                    if not raw:
                        print("This field could not be fetched; enter a number manually.")
                        continue
                    try:
                        value = _parse_cli_float(raw, field)
                        values[field] = value
                        sources[field] = InputRecord(field, value, f"Manual override (original: {rec.source})")
                        break
                    except ValueError as exc:
                        print(exc)
            else:
                if current is not None and args.accept_fetched:
                    values[field] = float(current)
                    sources[field] = rec
                elif current is not None and not interactive:
                    # Flags may have supplied some values; without an explicit
                    # accept flag, do not silently consume any other auto value.
                    continue
                elif interactive:
                    value, src = _prompt_value(field, current)
                    values[field] = value
                    sources[field] = InputRecord(field, value, src)
                else:
                    values[field] = None
                    sources[field] = rec
        if args.accept_fetched:
            for field in INPUT_ORDER:
                if field not in values and records[field].value is not None:
                    values[field] = float(records[field].value)
                    sources[field] = records[field]
        for field in INPUT_ORDER:
            if field not in sources:
                sources[field] = records[field]

    else:  # manual CLI mode
        flag_values: dict[str, Optional[float]] = {
            "EPS_GROWTH": args.eps_growth,
            "SECTOR_PE": args.sector_pe,
            "BETA": args.beta,
            "MKT_RETURN": args.mkt_return,
            "STOCK_PRICE": args.stock_price,
            "STOCK_EPS": args.stock_eps,
            "YEARS": args.years,
        }
        for field, value in flag_values.items():
            if value is not None:
                values[field] = float(value)
                sources[field] = InputRecord(field, float(value), "CLI manual input")
        if args.treasury is not None:
            values["TREASURY"] = float(args.treasury)
            sources["TREASURY"] = InputRecord("TREASURY", float(args.treasury), "CLI manual Treasury index")
        elif args.treasury_yield is not None:
            values["TREASURY"] = float(args.treasury_yield) * 10.0
            sources["TREASURY"] = InputRecord(
                "TREASURY", values["TREASURY"], "CLI manual yield percent × 10", None,
                f"Entered yield={args.treasury_yield:.4f}%; workbook index={values['TREASURY']:.4f}.",
            )

        # The interactive order matches the workbook instructions exactly.
        for field in INPUT_ORDER:
            if field in values:
                continue
            if field == "MKT_RETURN":
                current = DEFAULT_MARKET_RETURN
                default_label = "Default: Schwab LTCME."
            elif field == "YEARS":
                current = float(DEFAULT_YEARS)
                default_label = "Default: workbook horizon."
            else:
                current = None
                default_label = ""

            if field == "TREASURY":
                if interactive:
                    val, src, note = _interactive_treasury_prompt()
                    values[field] = val
                    sources[field] = InputRecord(field, val, src, None, note)
                else:
                    values[field] = None
                    sources[field] = InputRecord(field, None, "Manual entry — missing")
            elif interactive:
                val, src = _prompt_value(field, current, default_label=default_label)
                values[field] = val
                if src == "Default/accepted value" and field == "MKT_RETURN":
                    src = "Default: Schwab LTCME"
                elif src == "Default/accepted value" and field == "YEARS":
                    src = "Default: workbook horizon 5 years"
                sources[field] = InputRecord(field, val, src)
            elif field == "MKT_RETURN":
                values[field] = DEFAULT_MARKET_RETURN
                sources[field] = InputRecord(field, DEFAULT_MARKET_RETURN, "Default: Schwab LTCME")
            elif field == "YEARS":
                values[field] = float(DEFAULT_YEARS)
                sources[field] = InputRecord(field, float(DEFAULT_YEARS), "Default: workbook horizon 5 years")
            else:
                values[field] = None
                sources[field] = InputRecord(field, None, "Manual entry — missing")

    try:
        result = calculate_intrinsic_value(values)
    except MissingInputsError as exc:
        print("Cannot calculate. Missing required field(s): " + ", ".join(exc.missing), file=sys.stderr)
        return 2
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        print(f"Cannot calculate: {exc}", file=sys.stderr)
        return 2
    report = format_report(ticker, values, result, sources, currency)
    print(report)
    return 0


def main() -> None:
    raise SystemExit(cli_main())


if __name__ == "__main__":
    main()
