"""
Nadeem Walayat AI Portfolio Monitor - buying ranges & trim levels  (OPTIMIZED)
Streamlit app powered by yfinance (REAL market data - no synthetic fallback).

WHAT THIS FILE IS
-----------------
A drop-in replacement for `strategy_original.py`. All published levels (10X
BRIGADE, STOCKS), the rules text, the zone colours and every visible column are
unchanged; `test_parity.py` asserts that the rendered content is identical.

WHAT CHANGED AND WHY (measured on live Yahoo, 83 monitored tickers)
-------------------------------------------------------------------
Original cold-start cost, measured:
    quotes  5d/30m .........  6.2 s    81/83 tickers (MPW, RDFN delisted)
    ATH     max/1d ......... 11.1 s    16,273 rows / 65 MB in RAM -> 83 floats
    fundamentals .info x83 ..  1.4 s
                             -----
                             18.7 s  EVERY new browser session and EVERY restart

Root causes the original cannot avoid:
  1. `st.cache_data` is in-memory only and keyed on a per-session nonce, so a
     new session or a process restart refetches everything.
  2. All-time highs were recomputed from `period="max"` daily, though only the
     last few sessions can move a running maximum.
  3. Derived fundamentals (EGF proxy, FScore) were recomputed per row per
     rerun: 206 pure-function calls to fill 83 rows.
  4. Fetch failures were swallowed (`except Exception: return {}`) and rendered
     as a silent em-dash, so rate limiting looked like "no data".

Fixes here:
  * Disk-backed store (`.cache/ai_portfolio/`) -> warm start is file I/O, not
    network. Stale-while-revalidate: serve cached data, refresh behind it.
  * ATH kept as a persistent running max, updated incrementally (`1mo` of
    daily bars) instead of a full `period="max"` re-download.
  * Fundamentals cached per ticker with a 24 h TTL, retries + jittered
    backoff, and a visible failure report instead of silent em-dashes.
  * Derived metrics computed once at fetch time, not per row per rerun.
  * Retry/backoff on every network call; Yahoo 429/JSONDecode are retryable.
  * HTML uses CSS classes instead of ~1,079 repeated inline style strings
    (70.8 KB -> ~30 KB per monitor table, sent to the browser every rerun).
  * `html.escape` on every interpolated value (the original was injectable).

Zone thresholds, colours and wording kept. Level data updated only where the
25 Aug sheet / Briefs published a number that was previously blank. Tab layout
expanded in v3 (Market Overview, Crypto, EGF, Big Picture).

VERSION 2 CHANGES (requested after review)
-----------------------------------------
  * MPW and RDFN dropped from the monitored book. Both are delisted — Yahoo
    returns no price history for either (verified against the live API), so
    they could only ever render as "NO DATA".
  * All-time high FIXED: kept as a persisted running maximum, refreshed
    incrementally (one month of daily bars instead of period="max" every day),
    AND folded with the intraday high from the quote snapshot, so a name making
    a new high right now shows ~0% from ATH instead of a stale negative.
  * DCF FIXED and made the default: two-stage — 5 explicit growth years, a
    5-year linear fade to the terminal rate, then a perpetuity. The old
    single-stage formula (which inflated FCF by this year's growth but
    discounted at the terminal rate) is retained only as
    `dcf_fair_value_legacy`, the parity reference for the tests.
  * No invented inputs anywhere. If Yahoo does not report a growth rate, the
    model assumes the terminal rate and claims no growth premium; it does not
    substitute a default. Missing data renders as an em-dash and is counted in
    the status line — it is never back-filled, smoothed or simulated.


VERSION 3 CHANGES
-----------------
  * Stocks sorted alphabetically by ticker in 10X Brigade, Portfolio, and
    Fundamentals (user request).
  * Missing buy / take-profit levels filled from the 25 Aug 2026 portfolio
    sheet and Stocks Briefs where Nadeem published numbers (TMO $400–$450,
    BHP/CCJ/ALB/FCX/OXY/SLB/FSLR/UNH/MSTR/SNPS/PINS, etc.). No invented
    levels — names still without a published range stay blank.
  * MPW / RDFN remain excluded (delisted). No top-of-page warning for
    names still missing buy zones (user will fill as found).
  * New tabs: Market Overview (upcoming earnings, week lookback, sentiment),
    Crypto, EGF, Big Picture — each with how-to-use analysis guidance drawn
    from Walayat's sheets. Tab order: Monitor | Market Overview | Crypto |
    EGF | Fundamentals | Big Picture | Rules.

  * FIX: Streamlit Cloud AttributeError on store.overview — cache_resource was
    keeping a pre-v3 DataStore instance after deploy. STORE_VERSION cache key +
    self-heal in _get_store() + Refresh clears the resource cache.

Run:  pip install yfinance streamlit  &&  streamlit run ai_stocks_monitor_v3.py
"""

from __future__ import annotations

import html
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf

    YF_OK = True
except Exception:  # pragma: no cover - environment dependent
    yf = None
    YF_OK = False

# =============================================================================
# CONFIG
# =============================================================================
MARKET_TZ = "America/New_York"
NEAR_PCT_DEFAULT = 10.0
SNAPSHOT_TIMES = [(9, 30), (12, 0), (16, 0)]  # the only price updates of the day

# Disk cache. Override with AI_PORTFOLIO_CACHE=/path/to/dir
CACHE_DIR = Path(os.environ.get("AI_PORTFOLIO_CACHE", Path.home() / ".cache" / "ai_portfolio"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# TTLs (seconds)
QUOTE_TTL_S = 60 * 90          # a snapshot is authoritative until the next slot
ATH_TTL_S = 12 * 3600          # running maximum moves slowly
FUND_TTL_S = 24 * 3600         # fundamentals refresh once per day
OVERVIEW_TTL_S = 60 * 60       # benchmarks / earnings / week lookback
CRYPTO_TTL_S = 60 * 30         # spot crypto
ATH_INCREMENTAL_LOOKBACK_DAYS = 30

# Network policy. Yahoo rate-limits aggressively; fewer workers + backoff beats
# a wide fan-out that trips 429s and silently returns empty dicts.
FUND_MAX_WORKERS = 4
NET_MAX_ATTEMPTS = 4
NET_BACKOFF_BASE_S = 0.6
NET_BACKOFF_CAP_S = 8.0
RETRYABLE_HINTS = ("429", "too many requests", "jsondecode", "expecting value",
                   "connection", "timeout", "temporarily", "503", "502", "504")

# =============================================================================
# SNAPSHOT SCHEDULE
# =============================================================================
def _market_holidays() -> set:
    """NYSE holidays when pandas_market_calendars is installed, else empty.

    Install it (`pip install pandas_market_calendars`) to make the snapshot
    schedule holiday-aware; without it the app falls back to weekend-only,
    exactly like the original, and says so in the footer.
    """
    try:
        import pandas_market_calendars as mcal  # type: ignore

        cal = mcal.get_calendar("NYSE")
        yrs = range(pd.Timestamp.now(tz=MARKET_TZ).year - 1,
                    pd.Timestamp.now(tz=MARKET_TZ).year + 2)
        return {d.date() for d in cal.holidays().holidays
                if pd.Timestamp(d).year in yrs}
    except Exception:
        return set()


_HOLIDAYS: set = _market_holidays()


def _is_trading_day(d: pd.Timestamp) -> bool:
    return d.weekday() < 5 and d.date() not in _HOLIDAYS


def _prev_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    """Previous trading day. Bounded loop - a 3-week market closure is not a
    thing, but an unbounded `while` on a bad calendar is."""
    for _ in range(15):
        d = d - pd.Timedelta(days=1)
        if _is_trading_day(d):
            return d
    return d


def _next_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    for _ in range(15):
        d = d + pd.Timedelta(days=1)
        if _is_trading_day(d):
            return d
    return d


def current_snapshot_ts(now=None) -> pd.Timestamp:
    """Timestamp of the most recent scheduled snapshot (the data we display)."""
    if now is None:
        now = pd.Timestamp.now(tz=MARKET_TZ)
    if _is_trading_day(now):
        for h, m in reversed(SNAPSHOT_TIMES):
            slot = now.normalize() + pd.Timedelta(hours=h, minutes=m)
            if now >= slot:
                return slot
    prev = _prev_trading_day(now)
    return prev.normalize() + pd.Timedelta(hours=SNAPSHOT_TIMES[-1][0],
                                           minutes=SNAPSHOT_TIMES[-1][1])


def next_snapshot_ts(now=None) -> pd.Timestamp:
    """Timestamp of the next scheduled snapshot (when the app will wake)."""
    if now is None:
        now = pd.Timestamp.now(tz=MARKET_TZ)
    d = now.normalize()
    if _is_trading_day(now):
        for h, m in SNAPSHOT_TIMES:
            slot = d + pd.Timedelta(hours=h, minutes=m)
            if now < slot:
                return slot
    nxt = _next_trading_day(d)
    return nxt.normalize() + pd.Timedelta(hours=SNAPSHOT_TIMES[0][0],
                                          minutes=SNAPSHOT_TIMES[0][1])


def ms_until_next_snapshot(now: Optional[pd.Timestamp] = None) -> int:
    """Milliseconds to the next snapshot, clamped to (1 s, 24 h].

    The clamp matters: over a weekend the true wait is ~60 h, and feeding that
    to `st_autorefresh` as a millisecond interval is asking a browser timer to
    hold more than it needs to. Waking at most daily and recomputing on arrival
    is both safer and identical in behaviour (a closed market has nothing new).
    """
    now = now or pd.Timestamp.now(tz=MARKET_TZ)
    nxt = next_snapshot_ts(now)
    ms = int((nxt - now).total_seconds() * 1000) + 2000
    return max(1000, min(ms, 24 * 3600 * 1000))


# =============================================================================
# ⭐ 10X BRIGADE — hard-coded top & center (17 Jul 2026 article). Buy ranges
# and 10-year targets exactly as published. No trim levels: long-run
# accumulation plays. BESI is Amsterdam-listed (€) — static, not monitored.
# =============================================================================
# =============================================================================
# ⭐ 10X BRIGADE — hard-coded top & center (17 Jul 2026 article). Buy ranges
# and 10-year targets exactly as published. No trim levels: long-run
# accumulation plays. BESI is Amsterdam-listed (€) — static, not monitored.
# Sorted alphabetically by ticker (v3).
# =============================================================================
BRIGADE = [
    dict(t='ADBE', name='Adobe', buy_lo=190.0, buy_hi=235.0, target='1400', note='Exposure 137%.'),
    dict(
        t='BESI',
        name='BESI',
        buy_lo=145.0,
        buy_hi=194.0,
        target='2000',
        static=True,
        note='Amsterdam-listed (€192.10 on the sheet) — not US, not live-monitored.',
    ),
    dict(t='CLX', name='Clorox', buy_lo=82.0, buy_hi=93.0, target='480'),
    dict(
        t='CRCL',
        name='Circle',
        buy_lo=50.0,
        buy_hi=66.0,
        target='640',
        note='[TW] trimming cryptos; exposure 126%. Sheet trim $124–$138.',
    ),
    dict(
        t='CRM',
        name='Salesforce',
        buy_lo=130.0,
        buy_hi=163.0,
        target='720',
        note="[A comment] '$230 pumping', trim level asked — unanswered; exposure 125%.",
    ),
    dict(t='DUOL', name='Duolingo', buy_lo=65.0, buy_hi=105.0, target='800', note='Exposure 48%.'),
    dict(
        t='FICO',
        name='Fair Isaac',
        buy_lo=830.0,
        buy_hi=1170.0,
        target='6250',
        note='[TW 11 Aug] in buying range; exposure 124%.',
    ),
    dict(
        t='INTU',
        name='Intuit',
        buy_lo=235.0,
        buy_hi=292.0,
        target='1455',
        note='[TW] small sells; exposure 109%.',
    ),
    dict(
        t='NOW',
        name='ServiceNow',
        buy_lo=68.0,
        buy_hi=98.0,
        target='1040',
        note='[TW] small sells; exposure 96%.',
    ),
    dict(
        t='NVO',
        name='Novo Nordisk',
        buy_lo=35.0,
        buy_hi=48.0,
        target='250',
        note='[TW 11 Aug] in buying range; exposure 58%.',
    ),
    dict(t='PATH', name='PATH', buy_lo=8.0, buy_hi=12.0, target='120', note='[TW 18 Aug] small sell.'),
    dict(t='QBTS', name='QBTS', buy_lo=4.0, buy_hi=8.0, target='98'),
    dict(t='SMCI', name='SMCI', buy_lo=18.0, buy_hi=24.0, target='240', note='Exposure 19%.'),
    dict(
        t='VEEV',
        name='Veeva',
        buy_lo=138.0,
        buy_hi=166.0,
        target='1000',
        note='[TW] big sell 8% + trims; exposure 90%.',
    ),
]
BRIGADE_NOTE = ("Special section from the 17 Jul 2026 '10x Stocks to Accumulate' article • "
                "brigade +24.5% since mid-July • no trim levels — long-run accumulation "
                "(GREEN = in buying range, WHITE = within 10% of the top). "
                "Sorted A–Z by ticker.")

# =============================================================================
# MAIN LIST — portfolio sheet + article + briefs. Sorted A–Z by ticker (v3).
# Missing buy/trim filled ONLY from published 25 Aug sheet / Stocks Briefs.
# Brigade tickers are NOT repeated here. `note` = provenance, not rendered.
# =============================================================================
STOCKS = [
    dict(
        t='AAPL',
        name='Apple',
        buy_lo=190.0,
        buy_hi=226.0,
        trim=310.0,
        mech='Within 10% of High',
        section='Secondary',
    ),
    dict(
        t='ABBV',
        name='AbbVie',
        buy_lo=155.0,
        buy_hi=167.0,
        trim=241.0,
        mech='Within 10% of High',
        section='Healthcare',
    ),
    dict(
        t='ADSK',
        name='Autodesk',
        buy_lo=186.0,
        buy_hi=202.0,
        trim=310.0,
        mech='Within 10% of High',
        section='Medium Risk',
    ),
    dict(t="AEHR", name="AEHR", buy_lo=None, buy_hi=73.7, trim=132.66, mech="Within 10% of ATH", note="[ATH-derived 30 Aug 2026] Yahoo all-time high $147.40 (2026-08-14). Buy top = 50% off ATH; trim = within 10% of ATH."),
    dict(
        t='ALB',
        name='Albemarle',
        buy_lo=60.0,
        buy_hi=90.0,
        trim=230.0,
        mech='$206',
        section='Climate',
        note='[Sheet 25 Aug] buy $60–$90, trim $230.',
    ),
    dict(
        t='AMAT',
        name='AMAT',
        buy_lo=202.0,
        buy_hi=276.0,
        trim=703.0,
        mech='Within 5% of High',
        section='Secondary',
    ),
    dict(
        t='AMD',
        name='AMD',
        buy_lo=180.0,
        buy_hi=300.0,
        trim=585.0,
        mech='Only at ATH',
        target='600',
        section='Primary',
        note="[A] won't add much above $300; dream drop $200. Sheet buy $180–$260.",
    ),
    dict(t="AMT", name="American Tower", buy_lo=None, buy_hi=151.86, trim=273.35, mech="Within 10% of ATH", note="[ATH-derived 30 Aug 2026] Yahoo all-time high $303.72 (2021-09-08). Buy top = 50% off ATH; trim = within 10% of ATH."),
    dict(
        t='AMZN',
        name='Amazon',
        buy_lo=152.0,
        buy_hi=201.0,
        trim=258.0,
        mech='Within 10% of High',
        section='Secondary',
    ),
    dict(
        t='ARW',
        name='Arrow Electronics',
        buy_lo=106.0,
        buy_hi=132.0,
        trim=214.0,
        mech='Within 10% of High',
        section='Medium Risk',
    ),
    dict(
        t='ASML',
        name='ASML',
        buy_lo=1000.0,
        buy_hi=1326.0,
        trim=1900.0,
        mech='Within 5% of High',
        target='>2150',
        section='Primary',
        note='[A] 1300 then getting-lucky 1000.',
    ),
    dict(
        t='AVGO',
        name='Broadcom',
        buy_lo=272.0,
        buy_hi=336.0,
        trim=495.0,
        mech='Only at ATH',
        target='470–500+',
        section='Primary',
        note='[A] support $355 — break targets $300/$288. [CSV] trim ATH 495.',
    ),
    dict(t='BABA', name='Alibaba', buy_lo=84.0, buy_hi=106.0, trim=None, section='High Risk'),
    dict(
        t='BHP',
        name='BHP',
        buy_lo=54.0,
        buy_hi=64.0,
        trim=89.0,
        mech='Within 10% of High',
        section='Climate',
        note='[Sheet 25 Aug] buy $54–$64, trim $89.',
    ),
    dict(
        t='BIDU',
        name='Baidu',
        buy_lo=88.0,
        buy_hi=108.0,
        trim=None,
        section='Medium Risk',
        note="[A] 'buy the dumps such as BIDU'; 18 Aug mega buys to $88.",
    ),
    dict(t="BKNG", name="Booking", buy_lo=None, buy_hi=116.79, trim=210.22, mech="Within 10% of ATH", note="[ATH-derived 30 Aug 2026] Yahoo all-time high $233.58 (2025-07-08). Buy top = 50% off ATH; trim = within 10% of ATH."),
    dict(
        t='CCJ',
        name='Cameco',
        buy_lo=70.0,
        buy_hi=82.0,
        trim=122.0,
        mech='Within 10% of High',
        section='Climate',
        note='[Sheet 25 Aug] buy $70–$82, trim $122.',
    ),
    dict(
        t='COIN',
        name='Coinbase',
        buy_lo=112.0,
        buy_hi=148.0,
        trim=211.0,
        mech='$211–232',
        section='Crypto Equity',
    ),
    dict(t='CRSP', name='CRISPR', buy_lo=34.0, buy_hi=41.0, trim=None, section='High Risk'),
    dict(
        t='CRUS',
        name='Cirrus Logic',
        buy_lo=98.0,
        buy_hi=126.0,
        trim=162.0,
        mech='Within 10% of High',
        section='Medium Risk',
    ),
    dict(t='CSGP', name='CoStar', buy_lo=28.0, buy_hi=33.6, trim=None, section='High Risk'),
    dict(
        t='DIOD',
        name='Diodes',
        buy_lo=42.0,
        buy_hi=66.0,
        trim=113.0,
        mech='Within 10% of High',
        section='Medium Risk',
    ),
    dict(t='DOCU', name='Docusign', buy_lo=40.0, buy_hi=45.0, trim=None, section='High Risk'),
    dict(
        t='FCX',
        name='Freeport-McMoRan',
        buy_lo=34.0,
        buy_hi=50.0,
        trim=72.0,
        mech='Within 10% of High',
        section='Climate',
        note='[Sheet 25 Aug] buy $34–$50, trim $72. [B] accum sub $50.',
    ),
    dict(
        t='FLEX',
        name='FLEX',
        buy_lo=42.0,
        buy_hi=70.0,
        trim=150.0,
        mech='Within 10% of High',
        section='Medium Risk',
    ),
    dict(
        t='FOR',
        name='Forestar',
        buy_lo=15.0,
        buy_hi=20.0,
        trim=37.0,
        mech='Within 10% of High',
        section='Housing',
    ),
    dict(
        t='FSLR',
        name='First Solar',
        buy_lo=144.0,
        buy_hi=192.0,
        trim=305.0,
        mech='Within 5% of High',
        section='Climate',
        note='[Sheet 25 Aug] buy $144–$192, trim $305.',
    ),
    dict(
        t='GFS',
        name='GlobalFoundries',
        buy_lo=32.0,
        buy_hi=48.0,
        trim=83.0,
        mech='Within 10% of High',
        section='Medium Risk',
    ),
    dict(
        t='GOOG',
        name='Google',
        buy_lo=240.0,
        buy_hi=292.0,
        trim=404.0,
        mech='Only at ATH',
        target='430',
        section='Primary',
        note='[A] support 332 — break targets 300/272/240. Sheet buy $228–$292.',
    ),
    dict(t='GPN', name='GPN', buy_lo=62.0, buy_hi=68.0, trim=None, section='Medium Risk'),
    dict(t="GSK", name="GSK", buy_lo=None, buy_hi=38.09, trim=68.57, mech="Within 10% of ATH", note="[ATH-derived 30 Aug 2026] Yahoo all-time high $76.19 (1999-01-08). Buy top = 50% off ATH; trim = within 10% of ATH."),
    dict(t='HPQ', name='HP', buy_lo=16.0, buy_hi=18.6, trim=28.0, mech='$28–30', section='Medium Risk'),
    dict(
        t='IBM',
        name='IBM',
        buy_lo=168.0,
        buy_hi=208.0,
        trim=299.0,
        mech='Within 10% of High',
        section='Secondary',
    ),
    dict(t='IIPR', name='IIPR', buy_lo=38.0, buy_hi=44.0, trim=None, section='Housing'),
    dict(t='INMD', name='InMode', buy_lo=12.0, buy_hi=14.0, trim=None, section='Medium Risk'),
    dict(
        t='INTC',
        name='Intel',
        buy_lo=28.0,
        buy_hi=60.0,
        trim=135.0,
        mech='Within 5% of High',
        section='Secondary',
    ),
    dict(
        t='JBL',
        name='Jabil',
        buy_lo=180.0,
        buy_hi=238.0,
        trim=386.0,
        mech='Within 10% of High',
        section='Medium Risk',
    ),
    dict(
        t='JNJ',
        name='JnJ',
        buy_lo=154.0,
        buy_hi=182.0,
        trim=249.0,
        mech='Within 10% of High',
        section='Healthcare',
    ),
    dict(
        t='KLAC',
        name='KLAC',
        buy_lo=90.0,
        buy_hi=132.0,
        trim=292.0,
        mech='Within 5% of High',
        section='Secondary',
    ),
    dict(
        t='LMT',
        name='Lockheed Martin',
        buy_lo=422.0,
        buy_hi=458.0,
        trim=623.0,
        mech='Within 10% of High',
        section='Defence',
    ),
    dict(
        t='LOGI',
        name='Logitech',
        buy_lo=66.0,
        buy_hi=86.0,
        trim=126.0,
        mech='Within 10% of High',
        section='Medium Risk',
    ),
    dict(
        t='LRCX',
        name='Lam Research',
        buy_lo=138.0,
        buy_hi=202.0,
        trim=417.0,
        mech='Within 5% of High',
        section='Secondary',
    ),
    dict(
        t='META',
        name='META',
        buy_lo=360.0,
        buy_hi=548.0,
        trim=717.0,
        mech='Within 10% of High',
        target='~730',
        section='Primary',
        note='[A] 450s → below 400 → as low as 360 (puke case 240). Sheet buy $448–$548.',
    ),
    dict(t='MGNI', name='Magnite', buy_lo=8.6, buy_hi=11.6, trim=20.0, mech='$20–24', section='High Risk'),
    dict(t="MRNA", name="Moderna", buy_lo=None, buy_hi=248.74, trim=447.74, mech="Within 10% of ATH", note="[ATH-derived 30 Aug 2026] Yahoo all-time high $497.49 (2021-08-10). Buy top = 50% off ATH; trim = within 10% of ATH."),
    dict(
        t='MSFT',
        name='Microsoft',
        buy_lo=336.0,
        buy_hi=400.0,
        trim=500.0,
        mech='Within 10% of High',
        target='600',
        section='Primary',
        note='[A] buying opp toward 400, below = getting lucky. Sheet buy $336–$382.',
    ),
    dict(
        t='MSTR',
        name='MicroStrategy',
        buy_lo=56.0,
        buy_hi=90.0,
        trim=182.0,
        mech='$182–222',
        section='Crypto Equity',
        note='[Sheet 25 Aug] buy $56–$90, trim $182/$222. BTC proxy.',
    ),
    dict(
        t='MU',
        name='Micron',
        buy_lo=132.0,
        buy_hi=312.0,
        trim=1192.0,
        mech='Within 5% of High',
        section='Secondary',
    ),
    dict(
        t='NVDA',
        name='NVIDIA',
        buy_lo=174.0,
        buy_hi=190.6,
        trim=219.0,
        mech='Author ladder',
        target='~275',
        section='Primary',
        note='[A 26 Aug] buys 190.6/188/186/181/177/174, sells 219/226/231/236/248. Sheet $148–$183 / $237.',
    ),
    dict(
        t='ON',
        name='ON Semiconductor',
        buy_lo=38.0,
        buy_hi=62.0,
        trim=121.0,
        mech='Within 10% of High',
        section='Medium Risk',
    ),
    dict(
        t='OXY',
        name='Occidental',
        buy_lo=40.0,
        buy_hi=46.0,
        trim=64.0,
        mech='$60',
        section='Climate',
        note='[Sheet 25 Aug] buy $40–$46, trim $64. Briefs wider $40–$74 range-trade.',
    ),
    dict(t='PFE', name='Pfizer', buy_lo=22.0, buy_hi=24.3, trim=None, section='Healthcare'),
    dict(
        t='PINS',
        name='Pinterest',
        buy_lo=None,
        buy_hi=18.0,
        trim=38.0,
        mech='$38+',
        section='Other',
        note='[B] accumulate sub $18, next pump over $38.',
    ),
    dict(
        t='QCOM',
        name='Qualcomm',
        buy_lo=122.0,
        buy_hi=152.0,
        trim=234.0,
        mech='Within 10% of High',
        section='Secondary',
    ),
    dict(t='RBLX', name='Roblox', buy_lo=33.0, buy_hi=41.0, trim=60.0, mech='$60–69', section='High Risk'),
    dict(
        t='RTX',
        name='RTX',
        buy_lo=116.0,
        buy_hi=144.0,
        trim=204.0,
        mech='Within 10% of High',
        section='Defence',
    ),
    dict(
        t='SLB',
        name='SLB',
        buy_lo=37.0,
        buy_hi=44.0,
        trim=60.0,
        mech='$57',
        section='Climate',
        note='[Sheet 25 Aug] buy $37–$44, trim $60. Briefs wider $32–$60.',
    ),
    dict(t="SNAP", name="Snap", buy_lo=None, buy_hi=41.67, trim=75.01, mech="Within 10% of ATH", note="[ATH-derived 30 Aug 2026] Yahoo all-time high $83.34 (2021-09-24). Buy top = 50% off ATH; trim = within 10% of ATH."),
    dict(
        t='SNPS',
        name='Synopsys',
        buy_lo=380.0,
        buy_hi=400.0,
        trim=500.0,
        mech='$500+',
        section='Other',
        note='[B] range $380–$540; accumulate sub $400, trim over $500.',
    ),
    dict(t='SYNA', name='SYNA', buy_lo=46.0, buy_hi=70.0, trim=None, section='High Risk'),
    dict(t='TAK', name='Takeda', buy_lo=12.0, buy_hi=13.0, trim=None, section='Medium Risk'),
    dict(
        t='TCEHY',
        name='Tencent',
        buy_lo=40.0,
        buy_hi=55.0,
        trim=89.0,
        mech='Within 10% of High',
        section='High Risk',
    ),
    dict(
        t='TMO',
        name='Thermo Fisher',
        buy_lo=400.0,
        buy_hi=450.0,
        trim=None,
        target='600+',
        section='Other',
        note='[B 14 Jun] accumulate $400–$450 for eventual $600+.',
    ),
    dict(t="TOELY", name="Tokyo Electron", buy_lo=None, buy_hi=124.68, trim=224.43, mech="Within 10% of ATH", note="[ATH-derived 30 Aug 2026] Yahoo all-time high $249.37 (2026-06-29). Buy top = 50% off ATH; trim = within 10% of ATH."),
    dict(
        t='TSLA',
        name='Tesla',
        buy_lo=172.0,
        buy_hi=286.0,
        trim=449.0,
        mech='Within 10% of High',
        section='Secondary',
    ),
    dict(
        t='TSM',
        name='TSMC',
        buy_lo=315.0,
        buy_hi=390.0,
        trim=479.0,
        mech='Only at ATH',
        target='>500',
        section='Primary',
        note='[A] lightly adding sub 390, sweet spot ~330, support 315. Sheet buy $222–$322.',
    ),
    dict(t='ULH', name='ULH', buy_lo=12.6, buy_hi=14.6, trim=None, section='Medium Risk'),
    dict(
        t='UNH',
        name='UnitedHealth',
        buy_lo=234.0,
        buy_hi=272.0,
        trim=422.0,
        mech='$390',
        section='Healthcare',
        note='[Sheet 25 Aug] buy $234–$272, trim $422. [B] trim rallies.',
    ),
    dict(t="V", name="Visa", buy_lo=None, buy_hi=192.79, trim=347.01, mech="Within 10% of ATH", note="[ATH-derived 30 Aug 2026] Yahoo all-time high $385.57 (2026-08-26). Buy top = 50% off ATH; trim = within 10% of ATH."),
    dict(
        t='WDC',
        name='Western Digital',
        buy_lo=156.0,
        buy_hi=252.0,
        trim=720.0,
        mech='Within 10% of High',
        section='Medium Risk',
    ),
]

# =============================================================================
# RULES TO REMEMBER — Investing Guide + Real Secret distilled (unchanged)
# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# RULES TO REMEMBER — Investing Guide + Real Secret distilled (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
MANTRA = ("The name of the game: ACCUMULATE when CHEAP → DISTRIBUTE at a profit when EXPENSIVE. "
          "And the Real Secret: the Holy Grail is YOU — how you react to price in real time.")

REAL_SECRETS = [
    "1. You do NOT need technical analysis to trade — knowledge is not the goal; trading is a SKILL.",
    "2. Don't invest in trading theory or books — Gann/Elliott only confuse. The only objective: grow the account balance.",
    "3. Trade and CONCENTRATE on ONE market — catching crashes requires being in synch with it (a young person's sport: focus for months at a time).",
    "4. Imprint price action through practice — trade the PRICE, not indicators. Indicators are coin flips; when indicator and price disagree you get confused.",
    "5. The Holy Grail of trading is YOU — not a tool, theory or service. Be skeptical of every tool.",
    "6. MONEY MANAGEMENT is the critical secret — use stops and limits and move them in your FAVOUR. Bank profits, stop losses short, exit on doubt. You'll win ~50% at best, so the average win must far exceed the average loss.",
    "Exercise: hand-draw the daily / weekly / monthly OHLC charts of your one market — you'll learn more than from any book, site or course.",
]

GUIDE_GROUPS = [
    ("🧭 Foundations", [
        "Understand what INVESTING is: you own COMPANIES, not tickers — research and select GOOD companies, then let the numbers gyrate.",
        "Understand that which you are investing in — else you are a weak hand.",
        "Stock prices are NOT rational → you cannot catch the top or the bottom. Markets being irrational works FOR you if you stay rational.",
        "Indices are a RED HERRING — AI stocks lead the indices both ways; never time stock buys off the S&P/Nasdaq.",
    ]),
    ("📝 Plan & Accumulate", [
        "Have a plan and get the job done (e.g., build primaries to ≥10% of target exposure first).",
        "BUY on VALUATIONS, not price predictions — buy when undervalued, increase exposure as it gets cheaper.",
        "Scale into buying ranges: with $1,000 across a $306–$276 range, buy ~$30 per $1 drop — skew orders HEAVIER the deeper it falls.",
        "Your best buys are your most PAINFUL (the PAIN TRADE) — the brain is wired to avoid pain; logic must overcome evolution.",
        "50% DRAWDOWNS ARE NORMAL for good stocks — expect them, always keep powder dry (his cash target ~15%).",
        "Dollar-cost averaging fails in bear markets if you're unprepared — valuations first, mechanics second.",
        "INVEST AND FORGET: riding out both bull and bear beats clever in-and-out (the 10x brigade way).",
    ]),
    ("✂️ Trim & Sell", [
        "TRIM into strength: at 100% invested trim ~5% of target per 10% advance (≈25% sold by +50%); trim HEAVIER when over 100% invested.",
        "Trimming drives the average price paid DOWN and keeps exposure to the mega-trend.",
        "NEVER SELL AT A LOSS — works 8 times out of 10; the biggest portfolio killer is selling out at the bottom, the second is selling for peanuts.",
        "NO STOP LOSSES on investing positions — they do the opposite of accumulating.",
        "NEVER SHORT — losses are unlimited; trim long positions instead.",
    ]),
    ("🧠 Psychology", [
        "Don't buy FOMO, don't sell FEAR — fear vs greed are just opportunities to accumulate or distribute.",
        "THE NEWS IS ALWAYS BAD — that's what gets eyeballs. Ignore MSM/blogosphere; trust the metrics.",
        "Emotional investing = failure; keep a list of target stocks with target levels and let the list do the thinking.",
        "Your private-investor ADVANTAGE: no redemptions, no benchmark — you can hold through drawdowns institutions can't.",
    ]),
    ("📊 Metrics that matter (spreadsheet)", [
        "EGF (earnings growth factor) + direction of travel — negative EGF commands a LOW P/E before accumulating.",
        "EGF-12M — how strongly earnings could grow over the next 12 months.",
        "P/E % of range (current + forward) — FOMO-vs-CHEAP check: near the Oct-22 P/E = cheap; near/beyond the Dec-21 P/E = FOMO.",
        "Buying range + % from the top of the range — the accumulation map (ranges hold ~3+ months, tweaked after earnings).",
        "Share dilution — under 100% = buybacks, over 100% = printing shares (critical when EGF is weak).",
        "Fundamentals 0–10 score — PE, EPS, revenue, cash flow, ROE trend.",
    ]),
]

CRYPTO_REFERENCE = (
    "See the **Crypto** tab for live prices + Walayat's published buy/trim ladder "
    "(spot + MSTR/COIN/CRCL). Numbers are from the Cryptos sheet and the 25 Aug "
    "portfolio CSV — nothing is simulated."
)

# =============================================================================
# STATIC SHEET DATA (from Nadeem's AI Tech Stocks Portfolio, last updated
# 25 Aug 2026). Used by EGF / Big Picture / Crypto / Briefs panels. These are
# PUBLISHED figures, not computed estimates.
# =============================================================================
# Walayat EGF aggregate history (EGFs.csv). Percentages stored as fractions.
EGF_HISTORY = [
    dict(
        date='31 Dec 2024',
        spx=None,
        nasdaq=19280,
        ai_av=0.2,
        ai_12m=0.22,
        pe=0.75,
        sec_av=0.18,
        sec_12m=0.16,
        sec_pe=0.99,
        comments='',
    ),
    dict(
        date='7 Apr 2025',
        spx=None,
        nasdaq=15603,
        ai_av=0.14,
        ai_12m=0.32,
        pe=0.32,
        sec_av=0.14,
        sec_12m=0.2,
        sec_pe=0.59,
        comments='Positive EGFs, PE range cheap; SEC same as primary',
    ),
    dict(
        date='27 Jun 2025',
        spx=None,
        nasdaq=20273,
        ai_av=0.06,
        ai_12m=0.23,
        pe=0.76,
        sec_av=-0.02,
        sec_12m=0.13,
        sec_pe=1.09,
        comments='',
    ),
    dict(
        date='31 Aug 2025',
        spx=None,
        nasdaq=21455,
        ai_av=0.06,
        ai_12m=0.2,
        pe=0.76,
        sec_av=0.0,
        sec_12m=0.09,
        sec_pe=0.93,
        comments='Steady EGFs, moderating PE range; SEC weak EGFs, moderating PE range',
    ),
    dict(
        date='5 Nov 2025',
        spx=None,
        nasdaq=23500,
        ai_av=0.14,
        ai_12m=0.2,
        pe=0.94,
        sec_av=0.1,
        sec_12m=0.13,
        sec_pe=1.45,
        comments='Positive EGF, high PE range; Sec positive EGFs, very high PE range',
    ),
    dict(
        date='9 Dec 2025',
        spx=6840,
        nasdaq=23576,
        ai_av=0.17,
        ai_12m=0.16,
        pe=0.89,
        sec_av=0.1,
        sec_12m=0.13,
        sec_pe=1.47,
        comments='Positive EGF, moderating PE range; Sec positive EGFs, very high PE range',
    ),
]

BIG_PICTURE_GAINS = [{'label': 'To 22 Dec 2023', 'level': 4755, 'pct': None, 'av_yr': None, 'note': 'reference level'}, {'label': 'July 2018', 'level': 2816, 'pct': 0.63, 'av_yr': 0.126, 'note': ''}, {'label': 'July 2013', 'level': 1686, 'pct': 1.72, 'av_yr': 0.172, 'note': ''}, {'label': '2003', 'level': 990, 'pct': 3.63, 'av_yr': 0.182, 'note': ''}, {'label': '2000 Top', 'level': 1550, 'pct': 1.96, 'av_yr': 0.085, 'note': 'Even the worst time to buy stocks in modern history still yields 8.5% per annum!'}, {'label': '1993', 'level': 448, 'pct': 9.24, 'av_yr': 0.308, 'note': ''}, {'label': '1987 Top', 'level': 334, 'pct': 12.74, 'av_yr': 0.354, 'note': 'All that fuss about nothing!'}, {'label': '1983', 'level': 162, 'pct': 27.32, 'av_yr': 0.683, 'note': ''}, {'label': '1973', 'level': 108, 'pct': 41.48, 'av_yr': 0.83, 'note': ''}, {'label': '1963', 'level': 69.1, 'pct': 65.4, 'av_yr': 1.09, 'note': ''}, {'label': '1953', 'level': 24.75, 'pct': 184.37, 'av_yr': 2.634, 'note': ''}, {'label': '1943', 'level': 11.85, 'pct': 386.17, 'av_yr': 4.827, 'note': ''}, {'label': '1933', 'level': 9.95, 'pct': 460.11, 'av_yr': 5.112, 'note': ''}, {'label': '1929 Top', 'level': 31.7, 'pct': 143.73, 'av_yr': 1.529, 'note': 'All your great grandpa had to do was NOT SELL and pass it down.'}]

BIG_PICTURE_MANTRAS = ['Most investors act like headless chickens, buying high, selling low — because they fail to see the big picture.', "What's the message? INVEST AND FORGET!", 'Instead most investors try to weave in and out and completely miss the BIG PICTURE!', 'What matters most is not TIMING but TIME IN THE MARKET!', "You are NOT going to catch THE top or THE bottom — and for UK investors, FX can make a 'clever' exit 15% worse.", 'STOP thinking you can SELL MOST to buy back LATER — you will miss the big picture.', 'FOCUS ON THE BIG PICTURE!']

# Spot + crypto-equity book from Cryptos.csv + portfolio CRYPTOs section.
CRYPTO_ASSETS = [
    dict(
        t='BTC-USD',
        name='Bitcoin',
        kind='spot',
        buy_lo=48000,
        buy_hi=60000,
        trim_start=75000,
        trim_75=118000,
        trim_90=130000,
        dip_hi_ref=108000,
        dip_lo_pct=0.18,
        dip_hi_pct=0.23,
        primary_tgt=134000,
        second_tgt=144000,
        moon=196000,
        initial_tgt=106000,
        next_bear=44000,
        next_bear_lo=38000,
        next_bear_hi=48000,
        note='[B] Homing in on $48k target, accumulate sub $60k. [Crypto sheet] primary $134k.',
    ),
    dict(
        t='ETH-USD',
        name='Ethereum',
        kind='spot',
        buy_lo=2255,
        buy_hi=2870,
        trim_start=3400,
        trim_75=5900,
        trim_90=6900,
        dip_hi_ref=4100,
        dip_lo_pct=0.3,
        dip_hi_pct=0.45,
        note='[Crypto sheet] exit 3400/5900/6900; correction rebuy -30%/-45% off high.',
    ),
    dict(
        t='SOL-USD',
        name='Solana',
        kind='spot',
        buy_lo=163,
        buy_hi=195,
        trim_start=190,
        trim_75=400,
        trim_90=490,
        dip_hi_ref=263,
        dip_lo_pct=0.26,
        dip_hi_pct=0.38,
        note='[Crypto sheet] exit 190/400/490.',
    ),
    dict(
        t='DOGE-USD',
        name='Dogecoin',
        kind='spot',
        buy_lo=None,
        buy_hi=None,
        trim_start=0.2,
        trim_75=0.4,
        trim_90=0.5,
        note='[Crypto sheet] exit 0.20/0.40/0.50.',
    ),
    dict(
        t='LINK-USD',
        name='Chainlink',
        kind='spot',
        buy_lo=None,
        buy_hi=None,
        trim_start=20,
        trim_75=40,
        trim_90=50,
        note='[Crypto sheet] exit 20/40/50.',
    ),
    dict(
        t='ADA-USD',
        name='Cardano',
        kind='spot',
        buy_lo=None,
        buy_hi=None,
        trim_start=0.5,
        trim_75=1.5,
        trim_90=2.3,
        note='[Crypto sheet] exit 0.5/1.5/2.3.',
    ),
    dict(
        t='DOT-USD',
        name='Polkadot',
        kind='spot',
        buy_lo=None,
        buy_hi=None,
        trim_start=7.5,
        trim_75=23,
        trim_90=53,
        note='[Crypto sheet] exit 7.5/23/53.',
    ),
    dict(
        t='AVAX-USD',
        name='Avalanche',
        kind='spot',
        buy_lo=30,
        buy_hi=37.26,
        trim_start=33,
        trim_75=75,
        trim_90=125,
        dip_hi_ref=54,
        dip_lo_pct=0.31,
        dip_hi_pct=0.45,
        note='[Crypto sheet] exit 33/75/125.',
    ),
    dict(
        t='MSTR',
        name='MicroStrategy',
        kind='equity',
        buy_lo=56.0,
        buy_hi=90.0,
        trim=182.0,
        trim_start=180,
        trim_75=300,
        trim_90=400,
        fair_value=98,
        cheap=89,
        extreme=206,
        primary_tgt=250,
        second_tgt=300,
        note='[Sheet] BTC proxy. Fair ~$98, cheap ≤$89, extreme ≥$206. Exit 180/300/400.',
    ),
    dict(
        t='COIN',
        name='Coinbase',
        kind='equity',
        buy_lo=112.0,
        buy_hi=148.0,
        trim=211.0,
        trim_start=252,
        trim_75=350,
        trim_90=400,
        note='[Sheet] buy $112–$148, trim $211–$232. [Crypto] exit 252/350/400.',
    ),
    dict(
        t='CRCL',
        name='Circle',
        kind='equity',
        buy_lo=50.0,
        buy_hi=66.0,
        trim=124.0,
        note='[Sheet] buy $50–$66, trim $124–$138. Stablecoin gamble.',
    ),
    dict(
        t='BMNR',
        name='Bitmine (ETH proxy)',
        kind='equity',
        buy_lo=12.0,
        buy_hi=15.0,
        trim=28.0,
        note='[Sheet] Bitmine-Eth buy $12–$15, trim $28–$38. Verify Yahoo ticker.',
    ),
]

STOCK_BRIEFS = {'NVDA': ('Very Strong', 'Dip Buy', 'Very Strong Earnings growth is making the stock cheaper over TIME — Strong earnings failing to pump Nvidia suggests dip is probable, aim to accumulate sub $190.', '3-Jul'), 'GOOG': ('Strong', 'Light Trims', "Very good trend and not expensive, don't trim too much too early! But it is near an extreme so bubble valuation area so trend pause until good earnings play catch up.", '26-May'), 'AMD': ('Narrow', 'Buy Deep Dip', 'Very Strong earnings growth but right now very expensive! Seek sub $240 for risk vs reward.', '26-May'), 'MSFT': ('Strong', 'Accumulate', 'Very cheap, traded below 2022 PE low on the recent dip to $350, should accumulate and not trim sub $500.', '3-Jul'), 'META': ('Strong', 'Buy Deep Dip', "It had strong earnings growth but now weak, don't buy too high i.e. destined to new bear market lows below 520.", '3-Jul'), 'TSM': ('Strong', 'Dip Buy', "Mistake to sell all should have held a core position no matter what! It is expensive right now, it's going to take a panic event to get TSMC cheap!", '3-Jul'), 'QCOM': ('Weak', 'SELL', 'Weak earnings, volatile FOMO trend to $250+ to sell into, scaling out every 10 bucks it pumps, with a view to buying back sub $150 down to $120.', '29-May'), 'ASML': ('Very Strong', 'Dip Buy', 'Not too expensive, wait for a dip to buy starting at 1272, whilst trimming into new highs.', '3-Jul'), 'AVGO': ('Strong', 'Dip Buy', 'Consistently very strong earnings growth, do not sell out! Wait for a dip to buy and buy BIG at around $320.', '3-Jul'), 'LRCX': ('Strong', 'Buy Deep Dip', 'Very expensive even with good earnings growth a 30% to 50% drop is doable to under $200.', '26-May'), 'IBM': ('Weak', 'Accumulate', 'Stock has fallen with improving earnings which makes it very cheap i.e. $226 is like buying at $165 a year ago! It could range down to $160 once more.', '14-May'), 'KLAC': ('Strong', 'Dip Buy', 'Good growth but expensive right now! 30% drop is doable.', '14-May'), 'AMAT': ('Strong', 'Buy Deep Dip', 'Growing earnings but very expensive i.e. PE has doubled, selling sub $250 to accumulate.', '14-May'), 'AMZN': ('Strong', 'Light Trims', "Strong earnings growth, supportive of new high bull run — don't over trim; seek $200 to accumulate.", '14-May'), 'TSLA': ('Narrow', 'Buy Deep Dip', "Stock is falling but PE is going up — Don't buy too much too high! Stay short! It's a gamble!", '14-May'), 'MU': ('Narrow', 'Leave for later', 'In an earnings bubble that will eventually crack when the orders disappear.', '14-May'), 'AAPL': ('Strong', 'Dip Buy', 'Steady earnings growth due to buybacks, so buy if cheap for long-run return.', '14-May'), 'INTC': ('Weak', 'SELL', "Erratic earnings, scale out with sell limits $129, $139, $149 to sell what's left.", '13-May'), 'LMT': ('Strong', 'Range Trade', 'Earnings does not grow much so stock tends to run away from itself on war fomo. Buy on a deep dip to trim into fomo, its a dividend stock.', '7-May'), 'RTX': ('Strong', 'Dip Buy', 'Strong earnings growth, RTX is the defence stock to accumulate for the long-run, tends to run away from the buying ranges so add when it dips.', '16-Jun'), 'FCX': ('', 'Dip Buy', 'Strong earnings growth, consistent earnings beats, fair value in terms of PE range, accum sub $50.', '14-Jun'), 'OXY': ('', 'Range Trade', 'Earnings are taking off on the back of high oil price, EGFs imply its going to pump higher, the range is $74 to $40 so accumulate towards bottom and distribute towards top.', '16-Jun'), 'SLB': ('', 'Range Trade', 'In a $60 to $32 range as it awaits earnings growth.', '16-Jun'), 'FSLR': ('', 'Accumulate', 'Cheap, growing earnings accumulate, a dip is getting lucky to add more sub $200, will make new all time highs to trim into.', '9-May'), 'CCJ': ('', 'Buy Deep Dip', 'Earnings lifting but high PE, it may be coming to end of its bull run.', '7-May'), 'TMO': ('', 'Accumulate', 'Stable earnings, accum between $450 and $400 for eventual $600+.', '14-Jun'), 'SNPS': ('', 'Dip Buy', 'Stable earnings, crashed software, in a $540 to $380 range, accumulate sub $400, trim over $500.', '28-May'), 'PINS': ('', 'Dip Buy', 'High risk gamble stock, crashed from $90 high, accumulate sub $18 for next pump over $38.', '19-May'), 'BKNG': ('', 'Dip Buy', 'Okay earnings, taking a hit from war travel disruption, accumulate for the long-run as the price drops, and trim the pumps.', '19-May'), 'UNH': ('', 'Trim Rallies', 'Weak earnings, trim the rallies until earnings start to improve, $380 is like $480.', '9-May'), 'COIN': ('', 'Dip Buy', "Lacks earnings growth, it's a play on bitcoin trend, so swing trade it, accum sub $150.", '1-Jul'), 'MSTR': ('', 'Buy Deep Dip', 'Leveraged to bitcoin, accumulate for next cycle, sub $90 but volatile could go as low as $50, and there is the risk of an epic collapse to zero.', '1-Jul'), 'CRCL': ('', 'Dip Buy', 'Loss maker stablecoin gamble, accumulate sub $100 for eventual $300+.', '1-Jul'), 'BABA': ('', 'Buy Deep Dip', 'Was good but now going bad for some reason.', '7-May'), 'TCEHY': ('', 'Accumulate', '$60 equates to $50 a year ago, buy the dip.', '9-May')}

# Benchmark tickers for Market Overview (real Yahoo symbols only).
MARKET_BENCH = ["^GSPC", "^IXIC", "^DJI", "^VIX", "GC=F", "CL=F", "DX-Y.NYB"]
MARKET_BENCH_NAMES = {
    "^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow", "^VIX": "VIX",
    "GC=F": "Gold", "CL=F": "WTI Crude", "DX-Y.NYB": "US Dollar Index",
}

# Crypto Yahoo symbols we actually fetch (equities already in ALL_TICKERS).
CRYPTO_SPOT_TICKERS = [c["t"] for c in CRYPTO_ASSETS if c.get("kind") == "spot"]
CRYPTO_EQUITY_TICKERS = [c["t"] for c in CRYPTO_ASSETS if c.get("kind") == "equity"]

# =============================================================================
# DERIVED TICKER TABLES (built once at import)
# =============================================================================
# Removed from the monitored book. Both are delisted: Yahoo returns no price
# history for either (verified against the live API on 2026-08-29 — the
# download reports "No data found, symbol may be delisted"). They could only
# ever render as "NO DATA" while costing a request each.
#
# They are excluded by ticker here rather than deleted from STOCKS above, so
# the removal is visible and reversible, and so the level tables stay a
# verbatim copy of the published source (tools/build_optimized.py splices them
# in unchanged).
DELISTED: dict[str, str] = {
    "MPW": "delisted — no Yahoo data",
    "RDFN": "delisted — no Yahoo data",
}

PORTFOLIO: list[dict] = [s for s in STOCKS if s["t"] not in DELISTED]
MONITORED: list[dict] = [s for s in BRIGADE if not s.get("static")] + PORTFOLIO
ALL_TICKERS: list[str] = list(dict.fromkeys(s["t"] for s in MONITORED))  # de-dup, ordered

# Earnings / overview fetch set = equity book only (no crypto-USD pairs here).
EQUITY_TICKERS: list[str] = list(ALL_TICKERS)

# Levels as floats in one place, so the hot path never calls .get()/float().
LEVELS: dict[str, tuple[float, float, float]] = {
    s["t"]: (
        float(s["buy_lo"]) if s.get("buy_lo") is not None else float("nan"),
        float(s["buy_hi"]) if s.get("buy_hi") is not None else float("nan"),
        float(s["trim"]) if s.get("trim") is not None else float("nan"),
    )
    for s in MONITORED
}
BY_TICKER: dict[str, dict] = {s["t"]: s for s in MONITORED}


def validate_levels(levels: dict[str, tuple[float, float, float]] = LEVELS) -> list[str]:
    """Data-quality checks on the hand-maintained level tables.

    The original shipped these tables with no validation, so a typo (trim below
    the buy top, buy_lo above buy_hi) silently produced a zone no one could
    ever see. Cheap to check, expensive to debug by eye.
    """
    import math

    problems: list[str] = []
    for t, (lo, hi, trim) in levels.items():
        if not math.isnan(hi) and not math.isnan(lo) and lo > hi:
            problems.append(f"{t}: buy_lo ({lo:g}) > buy_hi ({hi:g}) — buy range inverted")
        if not math.isnan(trim) and not math.isnan(hi) and trim <= hi:
            problems.append(
                f"{t}: trim ({trim:g}) <= buy_hi ({hi:g}) — TRIM always shadows BUY")
        # v3: names without buy_hi/trim are intentional watch-only — user fills
        # them as found. Do NOT flag them (no top-of-page warning).
    return problems


# =============================================================================
# DISK CACHE + NETWORK RETRY
# =============================================================================
class DiskCache:
    """Tiny JSON key-value store with atomic writes.

    Atomic (tmp file + os.replace) so a crash or a concurrent session can never
    leave a half-written file that later un-pickles into a confusing crash.
    """

    def __init__(self, root: Path = CACHE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in key)
        return self.root / f"{safe}.json"

    def get(self, key: str, max_age_s: Optional[float] = None) -> Optional[Any]:
        p = self._path(key)
        try:
            if not p.exists():
                return None
            if max_age_s is not None and (time.time() - p.stat().st_mtime) > max_age_s:
                return None
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None  # corrupt or unreadable cache is just a miss

    def put(self, key: str, value: Any) -> None:
        p = self._path(key)
        tmp = p.with_suffix(f".tmp.{os.getpid()}")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(value, fh, separators=(",", ":"))
            os.replace(tmp, p)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def age_s(self, key: str) -> Optional[float]:
        p = self._path(key)
        return (time.time() - p.stat().st_mtime) if p.exists() else None


def _is_retryable(exc: BaseException) -> bool:
    s = f"{type(exc).__name__}:{exc}".lower()
    return any(h in s for h in RETRYABLE_HINTS)


def with_retry(fn: Callable[[], Any], *, attempts: int = NET_MAX_ATTEMPTS,
               base: float = NET_BACKOFF_BASE_S, cap: float = NET_BACKOFF_CAP_S,
               label: str = "") -> Any:
    """Exponential backoff with full jitter.

    Jitter is the part people leave out: without it, every worker that got a
    429 retries in lockstep and re-triggers the limit.
    """
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - network layer, re-raised below
            last = exc
            if i == attempts - 1 or not _is_retryable(exc):
                break
            sleep_s = min(cap, base * (2 ** i)) * (0.5 + random.random() / 2.0)
            time.sleep(sleep_s)
    raise last if last else RuntimeError(f"{label}: failed with no exception")


# =============================================================================
# PARSE HELPERS — robust to single-ticker (flat columns) and multi-ticker frames
# =============================================================================
def _tickers_of(data: pd.DataFrame) -> list[str]:
    """Level-0 tickers actually present in the frame.

    The original used `data.columns.levels[0]`, which (a) returns [] for the
    flat single-ticker case and (b) includes categories pandas no longer has
    data for, so every missing symbol cost a KeyError inside `except: continue`.
    """
    if data is None or data.empty:
        return []
    if isinstance(data.columns, pd.MultiIndex):
        return [str(t) for t in data.columns.get_level_values(0).unique()]
    return [str(data.columns.name)] if data.columns.name else []


def _series(data: pd.DataFrame, ticker: str, field: str) -> Optional[pd.Series]:
    try:
        if isinstance(data.columns, pd.MultiIndex):
            s = data[ticker][field]
        else:
            s = data[field]
        return s.dropna()
    except Exception:
        return None


def _parse_intraday(data) -> dict:
    """{ticker: (price, prev_close, ts)}; prev_close = last bar of the previous
    session."""
    out: dict[str, tuple] = {}
    for t in _tickers_of(data):
        closes = _series(data, t, "Close")
        if closes is None or closes.empty:
            continue
        idx = closes.index
        price = float(closes.iloc[-1])
        ts = idx[-1]
        ts = ts.tz_localize(MARKET_TZ) if ts.tzinfo is None else ts.tz_convert(MARKET_TZ)
        days = idx.normalize()
        earlier = closes[days < days[-1]]
        prev = float(earlier.iloc[-1]) if len(earlier) else None
        out[str(t)] = (price, prev, ts.isoformat())
    return out


def _parse_window_high(data) -> dict:
    """{ticker: highest intraday High in the quote window}.

    Folded into the all-time high so a name setting a new high *now* reads ~0%
    from ATH rather than a stale negative from yesterday's daily bar.
    """
    out: dict[str, float] = {}
    for t in _tickers_of(data):
        highs = _series(data, t, "High")
        if highs is not None and not highs.empty:
            out[str(t)] = float(highs.max())
    return out


def _parse_high(data) -> dict:
    out: dict[str, float] = {}
    for t in _tickers_of(data):
        highs = _series(data, t, "High")
        if highs is not None and not highs.empty:
            out[str(t)] = float(highs.max())
    return out


# =============================================================================
# FETCHERS (network)
# =============================================================================
def _download_quotes() -> tuple[dict, dict]:
    """Return ({ticker: (price, prev_close, ts)}, {ticker: window_high})."""
    data = with_retry(lambda: yf.download(
        tickers=ALL_TICKERS, period="5d", interval="30m",
        group_by="ticker", auto_adjust=False, progress=False, threads=True),
        label="quotes")
    return _parse_intraday(data), _parse_window_high(data)


def _download_aths_full(tickers: Sequence[str]) -> dict:
    return _parse_high(
        with_retry(lambda: yf.download(
            tickers=list(tickers), period="max", interval="1d",
            group_by="ticker", auto_adjust=False, progress=False, threads=True),
            label="ath:full")
    )


def _download_aths_incremental(start: str, tickers: Sequence[str]) -> dict:
    """Only the sessions since `start` — a running maximum cannot be moved by
    older data, so re-reading full history every day is pure waste."""
    return _parse_high(
        with_retry(lambda: yf.download(
            tickers=list(tickers), start=start, interval="1d",
            group_by="ticker", auto_adjust=False, progress=False, threads=True),
            label="ath:incremental")
    )


def _download_fundamentals_one(t: str) -> tuple[str, dict]:
    try:
        return t, with_retry(lambda: (yf.Ticker(t).info or {}), attempts=NET_MAX_ATTEMPTS,
                             label=f"info:{t}")
    except Exception:
        return t, {}


def _download_fundamentals(tickers: Optional[Sequence[str]] = None) -> tuple[dict, dict]:
    """Return ({ticker: info}, {ticker: error_string}).

    The second dict is the point: the original collapsed every failure to `{}`,
    so a Yahoo rate limit rendered as 83 rows of em-dashes with no hint that
    anything had gone wrong.
    """
    tickers = list(tickers or ALL_TICKERS)
    out: dict[str, dict] = {}
    errs: dict[str, str] = {}
    if not tickers or not YF_OK:
        return out, errs
    with ThreadPoolExecutor(max_workers=min(FUND_MAX_WORKERS, len(tickers))) as pool:
        for t, info in pool.map(_download_fundamentals_one, tickers):
            if info:
                out[t] = info
            else:
                errs[t] = "empty response (rate limited or delisted)"
    return out, errs


def _download_benchmarks() -> dict:
    """Index / macro snapshots for Market Overview. Real Yahoo symbols only."""
    data = with_retry(lambda: yf.download(
        tickers=MARKET_BENCH, period="10d", interval="1d",
        group_by="ticker", auto_adjust=False, progress=False, threads=True),
        label="benchmarks")
    out: dict[str, dict] = {}
    for t in MARKET_BENCH:
        closes = _series(data, t, "Close")
        if closes is None or closes.empty:
            continue
        price = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) > 1 else None
        week_ago = float(closes.iloc[-6]) if len(closes) >= 6 else (
            float(closes.iloc[0]) if len(closes) else None)
        out[t] = {
            "price": price,
            "prev": prev,
            "week_ago": week_ago,
            "day_pct": ((price / prev - 1.0) if prev else None),
            "week_pct": ((price / week_ago - 1.0) if week_ago else None),
        }
    return out


def _download_crypto_quotes() -> dict:
    """Spot crypto + any crypto-equity tickers not already in the main book
    (e.g. BMNR). MSTR/COIN/CRCL come from the main quote feed."""
    extra_eq = [t for t in CRYPTO_EQUITY_TICKERS if t not in set(ALL_TICKERS)]
    tickers = list(dict.fromkeys(CRYPTO_SPOT_TICKERS + extra_eq))
    if not tickers:
        return {}
    data = with_retry(lambda: yf.download(
        tickers=tickers, period="10d", interval="1d",
        group_by="ticker", auto_adjust=False, progress=False, threads=True),
        label="crypto")
    out: dict[str, dict] = {}
    for t in tickers:
        closes = _series(data, t, "Close")
        highs = _series(data, t, "High")
        if closes is None or closes.empty:
            continue
        price = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) > 1 else None
        week_ago = float(closes.iloc[-6]) if len(closes) >= 6 else (
            float(closes.iloc[0]) if len(closes) else None)
        ath_window = float(highs.max()) if highs is not None and not highs.empty else None
        out[t] = {
            "price": price,
            "prev": prev,
            "week_ago": week_ago,
            "day_pct": ((price / prev - 1.0) if prev else None),
            "week_pct": ((price / week_ago - 1.0) if week_ago else None),
            "window_high": ath_window,
        }
    return out


def _download_earnings_one(t: str) -> tuple[str, dict]:
    """Next earnings date + last reported surprise from Yahoo. No invented dates."""
    try:
        def load():
            tk = yf.Ticker(t)
            cal = {}
            try:
                cal = tk.calendar or {}
            except Exception:
                cal = {}
            next_dt = None
            # calendar may be dict with 'Earnings Date' list, or DataFrame
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date") or cal.get("earningsDate")
                if isinstance(ed, (list, tuple)) and ed:
                    next_dt = ed[0]
                elif ed is not None and not isinstance(ed, (list, tuple)):
                    next_dt = ed
            elif cal is not None and hasattr(cal, "empty") and not cal.empty:
                # DataFrame form
                if "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].iloc[0]
                    next_dt = val
            # last reported from earnings_dates
            last_reported = None
            last_surprise = None
            last_eps = None
            est_eps = None
            try:
                edf = tk.get_earnings_dates(limit=8)
            except Exception:
                edf = None
            if edf is not None and not edf.empty:
                # rows with Reported EPS
                for idx, row in edf.iterrows():
                    rep = row.get("Reported EPS") if hasattr(row, "get") else row["Reported EPS"] if "Reported EPS" in edf.columns else None
                    try:
                        import math
                        rep_f = float(rep) if rep is not None and rep == rep else None
                    except Exception:
                        rep_f = None
                    if rep_f is not None:
                        last_reported = idx
                        last_eps = rep_f
                        sur = row["Surprise(%)"] if "Surprise(%)" in edf.columns else None
                        try:
                            last_surprise = float(sur) if sur is not None and sur == sur else None
                        except Exception:
                            last_surprise = None
                        break
                # upcoming estimate = first row with no reported
                for idx, row in edf.iterrows():
                    rep = row["Reported EPS"] if "Reported EPS" in edf.columns else None
                    try:
                        rep_f = float(rep) if rep is not None and rep == rep else None
                    except Exception:
                        rep_f = None
                    if rep_f is None:
                        next_dt = next_dt or idx
                        est = row["EPS Estimate"] if "EPS Estimate" in edf.columns else None
                        try:
                            est_eps = float(est) if est is not None and est == est else None
                        except Exception:
                            est_eps = None
                        break
            def _iso(x):
                if x is None:
                    return None
                try:
                    ts = pd.Timestamp(x)
                    return ts.isoformat()
                except Exception:
                    return str(x)
            return {
                "next": _iso(next_dt),
                "last": _iso(last_reported),
                "last_eps": last_eps,
                "last_surprise_pct": last_surprise,  # already in percent units from Yahoo
                "est_eps": est_eps,
            }
        return t, with_retry(load, attempts=2, label=f"earn:{t}")
    except Exception as exc:
        return t, {"error": f"{type(exc).__name__}: {exc}"}


def _download_earnings(tickers: Sequence[str]) -> dict:
    out: dict[str, dict] = {}
    if not tickers or not YF_OK:
        return out
    with ThreadPoolExecutor(max_workers=min(FUND_MAX_WORKERS, len(tickers))) as pool:
        for t, info in pool.map(_download_earnings_one, tickers):
            out[t] = info or {}
    return out


def _download_week_returns(tickers: Sequence[str]) -> dict:
    """Trailing ~5-session return per equity ticker from daily bars."""
    if not tickers:
        return {}
    data = with_retry(lambda: yf.download(
        tickers=list(tickers), period="10d", interval="1d",
        group_by="ticker", auto_adjust=False, progress=False, threads=True),
        label="week_returns")
    out: dict[str, dict] = {}
    for t in tickers:
        closes = _series(data, t, "Close")
        if closes is None or len(closes) < 2:
            continue
        price = float(closes.iloc[-1])
        # ~5 trading sessions back (or earliest available)
        ref_idx = -6 if len(closes) >= 6 else 0
        ref = float(closes.iloc[ref_idx])
        if ref:
            out[t] = {"price": price, "ref": ref, "week_pct": price / ref - 1.0}
    return out


# =============================================================================
# DERIVED METRICS — computed once per ticker per day, not per row per rerun
# =============================================================================
def _num(v) -> Optional[float]:
    """Coerce a yfinance value to float, or None. Kills the scattered
    `info.get(x) or 0` pattern that silently treats 0.0 and None alike."""
    try:
        if v is None:
            return None
        f = float(v)
        return None if f != f else f  # NaN -> None
    except Exception:
        return None


def egf_proxy(info: dict) -> Optional[float]:
    """EGF proxy as a FRACTION (0.31 = +31%).

    UNIT CHANGE vs the original, which returned 31.0. The original then divided
    by 100 in one call site and not in the other, so the Monitor tab and the
    Fundamentals tab disagreed by a factor of 100 on the same quantity. One
    unit, decided here, formatted at the edge.
    """
    ttm_eps, fwd_eps = _num(info.get("trailingEps")), _num(info.get("forwardEps"))
    if ttm_eps and fwd_eps and ttm_eps > 0 and fwd_eps > 0:
        return fwd_eps / ttm_eps - 1.0
    ttm_pe, fwd_pe = _num(info.get("trailingPE")), _num(info.get("forwardPE"))
    if ttm_pe and fwd_pe and ttm_pe > 0 and fwd_pe > 0:
        return ttm_pe / fwd_pe - 1.0
    return None


def _first_shares(info: dict) -> Optional[float]:
    """Share count with a fallback chain over REAL Yahoo fields only.

    yfinance omits `sharesOutstanding` on a minority of tickers (ADRs and
    recent IPOs especially); the original then showed "—" for DCF on names that
    were perfectly valueable. No value is ever invented — if none of the three
    fields is present, this returns None and the cell renders as an em-dash.
    """
    for k in ("sharesOutstanding", "impliedSharesOutstanding", "floatShares"):
        v = _num(info.get(k))
        if v and v > 0:
            return v
    return None


def dcf_fair_value(info: dict, wacc: float = 0.095, terminal_g: float = 0.025,
                   stage1_years: int = 5, fade_years: int = 5) -> Optional[float]:
    """Per-share fair value from a TWO-STAGE FCF DCF. This is the model the app
    uses.

        years 1..S            grow FCF at the reported growth rate (capped 25%)
        next F years          fade linearly from that rate to terminal_g
        thereafter            perpetuity growing at terminal_g
        equity                PV + totalCash - totalDebt
        per share             equity / shares outstanding

    Why this replaces the single-stage version
    ------------------------------------------
    The original computed `EV = FCF x (1 + g) / (WACC - terminal_g)` with
    `g` = this year's earnings growth capped at 6%. That mixes two different
    growth rates: it inflates cash flow by `g` but discounts at a *different*
    rate, terminal_g. If `g` is the perpetual growth rate the denominator must
    be `WACC - g`. As written the model said "grow 6% once, then 2.5% forever",
    which is neither the stated model nor a standard one — and it capitalises a
    single year of current growth into perpetuity.

    Validated by the Gordon identity: when the reported growth equals the
    terminal rate, the explicit forecast reproduces the closed-form perpetuity
    exactly (asserted in test_parity.py). Monotonic decreasing in WACC, and it
    refuses WACC <= terminal_g instead of returning a negative denominator.

    No invented inputs. If Yahoo reports no growth rate, stage-1 growth is the
    terminal rate — the model then claims no growth premium at all, rather than
    substituting a default. Returns None (renders as an em-dash) when FCF <= 0,
    the share count is unavailable, or the discount margin is not positive.

    A rough anchor, NOT a price target.
    """
    fcf, shares = _num(info.get("freeCashflow")), _first_shares(info)
    wacc, terminal_g = float(wacc), float(terminal_g)
    if not fcf or fcf <= 0 or not shares or shares <= 0:
        return None
    if wacc - terminal_g <= 0.005:
        return None

    g0 = _num(info.get("earningsGrowth"))
    if g0 is None:
        g0 = _num(info.get("revenueGrowth"))
    if g0 is None:
        g0 = terminal_g  # no growth data -> claim no growth premium
    g0 = min(max(g0, 0.0), 0.25)

    rates = [g0] * stage1_years
    if fade_years > 0:
        rates += [g0 + (terminal_g - g0) * (i + 1) / fade_years
                  for i in range(fade_years)]

    pv = 0.0
    cash_flow = fcf
    for i, g in enumerate(rates, start=1):
        cash_flow *= (1.0 + g)
        pv += cash_flow / (1.0 + wacc) ** i
    terminal = cash_flow * (1.0 + terminal_g) / (wacc - terminal_g)
    pv += terminal / (1.0 + wacc) ** len(rates)

    equity = pv + (_num(info.get("totalCash")) or 0.0) - (_num(info.get("totalDebt")) or 0.0)
    return equity / shares


def dcf_fair_value_legacy(info: dict, wacc: float = 0.095,
                          terminal_g: float = 0.025) -> Optional[float]:
    """The ORIGINAL single-stage formula, bit for bit.

    Kept only as the parity reference: test_parity.py asserts the shipped app's
    numbers are unchanged wherever both models return a value, and quantifies
    where the corrected model differs. Not used by the UI.

    EV = FCF x (1 + g) / (WACC - terminal_g),  equity = EV + cash - debt.
    """
    fcf, shares = _num(info.get("freeCashflow")), _num(info.get("sharesOutstanding"))
    growth = _num(info.get("earningsGrowth"))
    if growth is None:
        growth = _num(info.get("revenueGrowth"))
    # `0.03` is the original's hard-coded stand-in when Yahoo reports no growth.
    # Reproduced exactly on purpose: this function exists to prove the shipped
    # model is a deliberate change, not an accident. Nothing in the app calls it.
    g = min(max(growth, 0.0), 0.06) if growth is not None else 0.03
    denom = float(wacc) - float(terminal_g)
    if not fcf or fcf <= 0 or not shares or shares <= 0 or denom <= 0.005:
        return None
    ev = fcf * (1.0 + g) / denom
    equity = ev + (_num(info.get("totalCash")) or 0.0) - (_num(info.get("totalDebt")) or 0.0)
    return equity / shares


def fundamentals_score(info: dict, egf: Optional[float] = None) -> Optional[int]:
    """Transparent 0-10 fundamentals score. `egf` may be supplied precomputed
    (the original recomputed it inside, so every row paid for it twice)."""
    if not info:
        return None
    peg = _num(info.get("trailingPegRatio")) or _num(info.get("pegRatio"))
    fwd_pe, ttm_pe = _num(info.get("forwardPE")), _num(info.get("trailingPE"))
    if egf is None:
        egf = egf_proxy(info)
    checks = [
        (egf or 0.0) > 0,
        (_num(info.get("earningsGrowth")) or 0.0) > 0,
        (_num(info.get("revenueGrowth")) or 0.0) > 0,
        (_num(info.get("profitMargins")) or 0.0) > 0.10,
        (_num(info.get("returnOnEquity")) or 0.0) > 0.15,
        bool(fwd_pe) and bool(ttm_pe) and fwd_pe < ttm_pe,
        (_num(info.get("totalCash")) or 0.0) > (_num(info.get("totalDebt")) or 0.0),
        (_num(info.get("freeCashflow")) or 0.0) > 0,
        (_num(info.get("grossMargins")) or 0.0) > 0.30,
        bool(peg) and 0.0 < peg <= 2.0,
    ]
    return sum(1 for c in checks if c)


def debt_ratio(info: dict) -> tuple[Optional[float], str]:
    """Return (ratio, basis). The original labelled debt/cash as 'D/E', which
    is not what D/E means; this returns the basis so the header is honest."""
    debt = _num(info.get("totalDebt"))
    equity = _num(info.get("totalStockholdersEquity"))
    cash = _num(info.get("totalCash"))
    if debt and equity and equity > 0:
        return debt / equity, "D/E"
    if debt and cash and cash > 0:
        return debt / cash, "D/C"
    return None, "D/E"


def compute_metrics(info: dict) -> dict:
    """Everything derivable from one `.info` payload that does NOT depend on
    live price or on the sidebar sliders. Computed once per ticker per day."""
    if not info:
        return {}
    egf = egf_proxy(info)
    return {
        "fwd_pe": _num(info.get("forwardPE")),
        "ttm_pe": _num(info.get("trailingPE")),
        "peg": _num(info.get("trailingPegRatio")) or _num(info.get("pegRatio")),
        "egf": egf,
        "score": fundamentals_score(info, egf=egf),
        "eps_yoy": _num(info.get("earningsGrowth")),
        "rev_g": _num(info.get("revenueGrowth")),
        "gross_m": _num(info.get("grossMargins")),
        "oper_m": _num(info.get("operatingMargins")),
        "profit_m": _num(info.get("profitMargins")),
        "roe": _num(info.get("returnOnEquity")),
        "pb": _num(info.get("priceToBook")),
        "ps": _num(info.get("priceToSalesTrailing12Months")),
        "cash": _num(info.get("totalCash")),
        "debt": _num(info.get("totalDebt")),
        "fcf": _num(info.get("freeCashflow")),
        "mcap": _num(info.get("marketCap")),
        "target": _num(info.get("targetMeanPrice")),
        "rec": fmt_recommendation(info.get("recommendationKey"))[0],
        "rec_color": fmt_recommendation(info.get("recommendationKey"))[1],
    }


def fmt_recommendation(key) -> tuple[str, str]:
    """Yahoo recommendationKey -> (pretty label, color)."""
    if not key:
        return "—", "#546E7A"
    k = str(key)
    pretty = {"strong_buy": "Strong Buy", "buy": "Buy", "hold": "Hold",
              "underperform": "Underperform", "sell": "Sell",
              "none": "No rating"}.get(k, k.replace("_", " ").title())
    color = ("#00E676" if k in ("strong_buy", "buy")
             else ("#FF5252" if k in ("sell", "underperform") else "#FFD54F"))
    return pretty, color


# =============================================================================
# DATA STORE — disk-first, stale-while-revalidate, failures surfaced
# =============================================================================
@dataclass
class SourceState:
    name: str
    n: int = 0
    stale: bool = False
    fetched: bool = False
    seconds: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)
    note: str = ""


class DataStore:
    """Owns all Yahoo access, with single-flight request coalescing.

    Two properties matter:

    *Single-flight.* Every source is keyed, and at most one fetch per key runs
    at a time. `prefetch()` starts the work; the following `quotes()` /
    `aths()` / `fundamentals()` calls join the SAME futures rather than
    starting their own. The first draft of this class had prefetch and the
    getters fire independent fetches, which doubled the cold-start cost
    (30.1s against the original's 17.6s) and doubled Yahoo traffic with it.

    *Parallel by default.* Because the three sources are independent and all
    three are started before any is awaited, a cold start costs roughly the
    SLOWEST source rather than the sum of them. The original fetched
    sequentially: 6.2s + 11.1s + 1.4s.

    Cached data is served first; a refresh happens behind it, so a slider drag
    never waits on the network.
    """

    def __init__(self, cache: Optional[DiskCache] = None, offline: bool = False):
        self.cache = cache or DiskCache()
        self.offline = offline or not YF_OK
        self.states: dict[str, SourceState] = {}
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yf")
        self._inflight: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.window_highs: dict[str, float] = {}  # intraday highs, for the ATH

    # -- single-flight plumbing ------------------------------------------
    def _start(self, key: str, loader: Callable[[], Any]) -> Any:
        with self._lock:
            fut = self._inflight.get(key)
            if fut is None:
                fut = self._pool.submit(loader)
                self._inflight[key] = fut
            return fut

    def _run(self, key: str, loader: Callable[[], Any], wait: bool = True) -> Any:
        """Start (or join) the loader for `key`. `wait=False` is fire-and-forget."""
        fut = self._start(key, loader)
        if not wait:
            return None
        try:
            return fut.result()
        except Exception:
            return None  # the loader records the error on its SourceState
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def _state(self, name: str) -> SourceState:
        return self.states.setdefault(name, SourceState(name=name))

    # -- loaders (run in the pool, write to disk, record errors) ----------
    def _fetch_quotes(self, slot_key: str) -> Optional[dict]:
        key, stt = f"quotes-{slot_key}", self._state("quotes")
        try:
            t0 = time.perf_counter()
            quotes, highs = _download_quotes()
            raw = {"quotes": quotes, "highs": highs}
            self.cache.put(key, raw)
            stt.seconds = round(time.perf_counter() - t0, 2)
            stt.fetched, stt.stale, stt.n = True, False, len(raw)
            return raw
        except Exception as exc:
            stt.errors["_fetch"] = f"{type(exc).__name__}: {exc}"
            return None

    def _fetch_aths(self) -> Optional[dict]:
        stt = self._state("ath")
        try:
            t0 = time.perf_counter()
            today = pd.Timestamp.now(tz=MARKET_TZ).date().isoformat()
            state = self.cache.get("ath-state", max_age_s=None) or {}

            missing = [t for t in ALL_TICKERS if t not in state]
            just_seeded: set[str] = set()
            if missing:  # first run, or a newly added ticker
                for t, h in _download_aths_full(missing).items():
                    state[t] = {"max": h, "asof": today}
                    just_seeded.add(t)

            # Names just seeded from full history already hold every high, so
            # including them here would repeat work for nothing.
            refresh = [t for t in ALL_TICKERS if t not in just_seeded]
            if refresh:
                asofs = [v.get("asof") for t, v in state.items()
                         if t in refresh and v.get("asof")]
                start = min(asofs) if asofs else today
                start = max(start, (pd.Timestamp.now(tz=MARKET_TZ) - pd.Timedelta(
                    days=ATH_INCREMENTAL_LOOKBACK_DAYS)).date().isoformat())
                for t, h in _download_aths_incremental(start, refresh).items():
                    prev = state.get(t, {}).get("max")
                    state[t] = {"max": max(h, prev) if prev is not None else h,
                                "asof": today}

            self.cache.put("ath-state", state)
            stt.seconds = round(time.perf_counter() - t0, 2)
            stt.fetched, stt.stale, stt.n = True, False, len(state)
            return state
        except Exception as exc:
            stt.errors["_fetch"] = f"{type(exc).__name__}: {exc}"
            return None

    def _fetch_funds(self, day_key: str) -> Optional[dict]:
        stt = self._state("fundamentals")
        try:
            t0 = time.perf_counter()
            info, errs = _download_fundamentals(ALL_TICKERS)
            metrics = {t: compute_metrics(i) for t, i in info.items()}
            bundle = {"info": info, "metrics": metrics, "errors": errs}
            self.cache.put(f"funds-{day_key}", bundle)
            stt.seconds = round(time.perf_counter() - t0, 2)
            stt.fetched, stt.stale, stt.n = True, False, len(info)
            stt.errors.update(errs)
            return bundle
        except Exception as exc:
            stt.errors["_fetch"] = f"{type(exc).__name__}: {exc}"
            return None

    # -- public getters (never raise; serve cache first) -----------------
    def quotes(self, slot_key: str, wait: bool = True) -> dict:
        stt, key = self._state("quotes"), f"quotes-{slot_key}"
        raw = self.cache.get(key, max_age_s=QUOTE_TTL_S)
        if raw is None and not self.offline:
            raw = self._run(key, lambda: self._fetch_quotes(slot_key), wait=wait)
        if raw is None:  # fresh fetch unavailable -> serve whatever is on disk
            raw = self.cache.get(key, max_age_s=None)
            if raw is not None:
                stt.stale, stt.note = True, "serving last cached snapshot"
        if raw is None:
            return {t: None for t in ALL_TICKERS}
        # New shape bundles the window highs with the quotes; tolerate an older
        # cache that stored the quotes dict on its own.
        if isinstance(raw, dict) and isinstance(raw.get("quotes"), dict):
            self.window_highs = raw.get("highs") or {}
            raw = raw["quotes"]
        out: dict[str, Any] = {t: None for t in ALL_TICKERS}
        for t, v in raw.items():
            if t in out and v:
                price, prev, ts = v
                out[t] = (price, prev, pd.Timestamp(ts))
        stt.stale = False
        stt.n = sum(1 for v in out.values() if v)
        stt.errors.update({t: "no quote" for t, v in out.items() if v is None})
        return out

    def aths(self, wait: bool = True) -> dict:
        stt = self._state("ath")
        age = self.cache.age_s("ath-state")
        fresh = age is not None and age <= ATH_TTL_S
        state = self.cache.get("ath-state", max_age_s=None)
        if (state is None or not fresh) and not self.offline:
            got = self._run("ath-state", self._fetch_aths, wait=wait)
            if got is not None:
                state, fresh = got, True
        if state is None:
            stt.stale = True
            return {}
        stt.stale = not fresh
        stt.n = len(state)
        if not fresh:
            stt.note = "refresh in flight — showing cached highs"
        stt.errors.update({t: "no history" for t in ALL_TICKERS if t not in state})
        out = {t: float(v["max"]) for t, v in state.items()}
        # Fold in the intraday high from the quote snapshot. A running maximum
        # can only rise, so this cannot make the number worse; it stops today's
        # new high from reading as a stale negative until the daily bar lands.
        for t, h in (self.window_highs or {}).items():
            if t in out:
                out[t] = max(out[t], h)
            else:
                out[t] = h
        return out

    def fundamentals(self, day_key: str, wait: bool = True) -> tuple[dict, dict]:
        stt, key = self._state("fundamentals"), f"funds-{day_key}"
        bundle = self.cache.get(key, max_age_s=FUND_TTL_S)
        if bundle is None and not self.offline:
            bundle = self._run(key, lambda: self._fetch_funds(day_key), wait=wait)
        if bundle is None:  # stale-while-error: yesterday's fundamentals beat none
            cands = sorted(self.cache.root.glob("funds-*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            for cand in cands:
                bundle = self.cache.get(cand.stem, max_age_s=None)
                if bundle:
                    stt.stale, stt.note = True, "serving last cached fundamentals"
                    break
        if not bundle:
            return {}, {}
        stt.stale = False
        stt.n = len(bundle.get("info", {}))
        stt.errors.update(bundle.get("errors", {}))
        return bundle.get("info", {}), bundle.get("metrics", {})

    # -- market overview / crypto / earnings --------------------------------
    def _fetch_overview(self, day_key: str) -> Optional[dict]:
        stt = self._state("overview")
        try:
            t0 = time.perf_counter()
            benches = _download_benchmarks()
            # earnings only for the equity book (skip if huge — still OK at ~80)
            earns = _download_earnings(EQUITY_TICKERS)
            week = _download_week_returns(EQUITY_TICKERS)
            bundle = {"benchmarks": benches, "earnings": earns, "week": week}
            self.cache.put(f"overview-{day_key}", bundle)
            stt.seconds = round(time.perf_counter() - t0, 2)
            stt.fetched, stt.stale, stt.n = True, False, len(earns)
            return bundle
        except Exception as exc:
            stt.errors["_fetch"] = f"{type(exc).__name__}: {exc}"
            return None

    def _fetch_crypto(self) -> Optional[dict]:
        stt = self._state("crypto")
        try:
            t0 = time.perf_counter()
            spot = _download_crypto_quotes()
            self.cache.put("crypto-spot", spot)
            stt.seconds = round(time.perf_counter() - t0, 2)
            stt.fetched, stt.stale, stt.n = True, False, len(spot)
            return spot
        except Exception as exc:
            stt.errors["_fetch"] = f"{type(exc).__name__}: {exc}"
            return None

    def overview(self, day_key: str, wait: bool = True) -> dict:
        stt, key = self._state("overview"), f"overview-{day_key}"
        bundle = self.cache.get(key, max_age_s=OVERVIEW_TTL_S)
        if bundle is None and not self.offline:
            bundle = self._run(key, lambda: self._fetch_overview(day_key), wait=wait)
        if bundle is None:
            bundle = self.cache.get(key, max_age_s=None)
            if bundle is not None:
                stt.stale, stt.note = True, "serving last cached overview"
        return bundle or {}

    def crypto_spot(self, wait: bool = True) -> dict:
        stt = self._state("crypto")
        spot = self.cache.get("crypto-spot", max_age_s=CRYPTO_TTL_S)
        if spot is None and not self.offline:
            spot = self._run("crypto-spot", self._fetch_crypto, wait=wait)
        if spot is None:
            spot = self.cache.get("crypto-spot", max_age_s=None)
            if spot is not None:
                stt.stale, stt.note = True, "serving last cached crypto"
        return spot or {}

    # -- background warming ------------------------------------------------
    def prefetch(self, slot_key: str, day_key: str) -> None:
        """Start a refresh for anything stale. Never blocks, never duplicates:
        if a fetch for a key is already in flight the same future is reused."""
        if self.offline:
            return
        if self.cache.get(f"quotes-{slot_key}", QUOTE_TTL_S) is None:
            self._start(f"quotes-{slot_key}", lambda: self._fetch_quotes(slot_key))
        if self.cache.get(f"funds-{day_key}", FUND_TTL_S) is None:
            self._start(f"funds-{day_key}", lambda: self._fetch_funds(day_key))
        if (self.cache.age_s("ath-state") or float("inf")) > ATH_TTL_S:
            self._start("ath-state", self._fetch_aths)
        if self.cache.get(f"overview-{day_key}", OVERVIEW_TTL_S) is None:
            self._start(f"overview-{day_key}", lambda: self._fetch_overview(day_key))
        if self.cache.get("crypto-spot", CRYPTO_TTL_S) is None:
            self._start("crypto-spot", self._fetch_crypto)

    def pending(self) -> list[str]:
        names = {"ath-state": "ath"}
        out = []
        for k, f in self._inflight.items():
            if not f.done():
                name = names.get(k, k.split("-")[0])
                out.append("ath" if name == "ath" else name)
        return sorted(set(out))

    def status_line(self) -> str:
        bits = []
        inflight = set(self.pending())
        for name in ("quotes", "ath", "fundamentals", "overview", "crypto"):
            s = self.states.get(name)
            if not s:
                continue
            if name in inflight:
                tag = "fetching…"
            elif s.errors and any(k.startswith("_") for k in s.errors):
                tag = "fetch failed (cached)"
            elif s.fetched:
                tag = f"fetched {s.seconds}s"
            elif s.stale:
                tag = "stale"
            else:
                tag = "cached"
            n_missing = len([k for k in s.errors if not k.startswith("_")])
            if n_missing:
                tag += f", {n_missing} missing"
            bits.append(f"{name}: {tag}")
        return "  •  ".join(bits)


# =============================================================================
# ZONE LOGIC (pure — unit-testable)
# =============================================================================
ZONE_BUY, ZONE_NEAR, ZONE_TRIM, ZONE_WAIT, ZONE_NODATA = "BUY", "NEAR", "TRIM", "WAIT", "NODATA"
ZONE_COLORS = {ZONE_BUY: "#00E676", ZONE_NEAR: "#FFFFFF", ZONE_TRIM: "#FF5252",
               ZONE_WAIT: "#9E9E9E", ZONE_NODATA: "#616161"}
ZONE_CLASSES = {ZONE_BUY: "z-buy", ZONE_NEAR: "z-near", ZONE_TRIM: "z-trim",
                ZONE_WAIT: "z-wait", ZONE_NODATA: "z-nodata"}


def zone_of(price: Optional[float], stock: dict, near_pct: float = NEAR_PCT_DEFAULT) -> str:
    """BUY: in or below the buying range • NEAR: within near_pct above the top •
    TRIM: at/above trim • WAIT: otherwise • NODATA: no quote."""
    if price is None:
        return ZONE_NODATA
    trim = stock.get("trim")
    buy_hi = stock.get("buy_hi")
    if trim is not None and price >= float(trim):
        return ZONE_TRIM
    if buy_hi is not None and price <= float(buy_hi):
        return ZONE_BUY
    if buy_hi is not None:
        if price <= float(buy_hi) + float(buy_hi) * float(near_pct) / 100.0:
            return ZONE_NEAR
    return ZONE_WAIT


# ---------------------------------------------------------------------------
# Vectorised zone classification.
#
# The first version of this built its level matrix INSIDE the function with a
# nested comprehension (3n dict lookups + n list allocations per call). Measured
# against the scalar loop it was 3.8x-6.7x SLOWER at every size from 25 to
# 50,000 names — the Python data-preparation cost more than the arithmetic it
# was supposed to save. Classic anti-pattern: the "vectorised" part was 5%
# of the work.
#
# The fix is to hoist the matrix to module scope so the NumPy path does no
# per-call Python iteration at all. See tools/crossover.py for the sweep.
# ---------------------------------------------------------------------------
_TICKER_ORDER: list[str] = list(LEVELS)
_LEVEL_MATRIX: "np.ndarray" = np.array([LEVELS[t] for t in _TICKER_ORDER],
                                       dtype=np.float64)
_LEVEL_ROW: dict[str, int] = {t: i for i, t in enumerate(_TICKER_ORDER)}
_NAN = float("nan")

# Book size above which the NumPy path beats the scalar loop.
# tools/crossover.py measures the crossover at ~150 names (at 83 the vector
# path is still 1.27x SLOWER; at 300 it is 1.47x faster; at 50,000 it is 2x
# faster). Set at 300 to leave margin — switching on a 1.1x edge is not worth
# the branch. The shipped book is 83 names, so the scalar loop runs in
# production today; the NumPy path is there for when the list grows.
VECTOR_CROSSOVER = 300


def compute_zones(prices: dict[str, Optional[float]], near_pct: float = NEAR_PCT_DEFAULT,
                  levels: Optional[dict[str, tuple[float, float, float]]] = None) -> dict[str, str]:
    """Zone classification for the whole book, dispatching on size.

    Small book -> scalar loop. Large book -> NumPy, over a level matrix built
    once at import. Both paths are asserted equal by test_parity.py and by
    tools/crossover.py at every size.
    """
    n = len(prices)
    if n < VECTOR_CROSSOVER or levels is not None:
        by = BY_TICKER
        return {t: zone_of(prices[t], by[t], near_pct) for t in prices}
    return _compute_zones_vector(prices, near_pct)


def _compute_zones_vector(prices: dict[str, Optional[float]], near_pct: float) -> dict[str, str]:
    """NumPy path: no per-call Python iteration except the price gather."""
    order = _TICKER_ORDER
    if list(prices.keys()) == order:
        # Fast path: the store always builds the quote dict in ticker order, so
        # the gather is a C-level fromiter over dict.values().
        p = np.fromiter((v if v is not None else _NAN for v in prices.values()),
                        dtype=np.float64, count=len(order))
    else:
        p = np.array([prices.get(t) if prices.get(t) is not None else _NAN
                      for t in order], dtype=np.float64)

    buy_hi = _LEVEL_MATRIX[:, 1]
    trim = _LEVEL_MATRIX[:, 2]

    is_nodata = np.isnan(p)
    is_trim = (~is_nodata) & (p >= trim)
    is_buy = (~is_trim) & (~is_nodata) & (p <= buy_hi)
    near_top = buy_hi * (1.0 + float(near_pct) / 100.0)
    is_near = (~is_trim) & (~is_buy) & (~is_nodata) & (p <= near_top)

    codes = np.full(len(order), ZONE_WAIT, dtype=object)
    codes[is_near] = ZONE_NEAR
    codes[is_buy] = ZONE_BUY
    codes[is_trim] = ZONE_TRIM
    codes[is_nodata] = ZONE_NODATA
    return dict(zip(order, codes.tolist()))


def display_name(stock: dict, zone: str) -> str:
    return stock["name"].upper() if zone in (ZONE_BUY, ZONE_TRIM) else stock["name"]


# =============================================================================
# FORMATTERS
# =============================================================================
DASH = "—"


def fmt_money(v) -> str:
    # Fast path: the render loop calls this ~4x per row with values that are
    # already floats. Bypassing _num() saves a function call + try/except each.
    if type(v) is float or type(v) is int:
        # `v != v` is the cheap NaN test — a NaN is missing data, and the
        # original rendered it as the string "$nan".
        return DASH if v != v else (f"${v:,.2f}" if v < 1000 else f"${v:,.0f}")
    v = _num(v)
    if v is None:
        return DASH
    return f"${v:,.2f}" if v < 1000 else f"${v:,.0f}"


def fmt_range(stock) -> str:
    lo, hi = _num(stock.get("buy_lo")), _num(stock.get("buy_hi"))
    if hi is None:
        return DASH
    if lo is None:
        return f"≤ {fmt_money(hi)}"
    return f"{fmt_money(lo)} – {fmt_money(hi)}"


def fmt_ath(price, ath) -> str:
    price, ath = _num(price), _num(ath)
    if price is None or not ath:
        return DASH
    return f"{(price / ath - 1.0) * 100.0:+.1f}%"


def fmt_g(v) -> str:
    """Decimal fraction (0.31) -> '+31.0%'. The one place growth is formatted."""
    v = _num(v)
    return DASH if v is None else f"{v * 100.0:+.1f}%"


def fmt_n(v, digits: int = 1) -> str:
    if type(v) is float or type(v) is int:
        return DASH if v != v else f"{v:.{digits}f}"
    v = _num(v)
    return DASH if v is None else f"{v:.{digits}f}"


def fmt_usd_b(v) -> str:
    v = _num(v)
    return DASH if v is None else f"${v / 1e9:,.1f}B"


def fmt_pct_day(price, prev) -> str:
    price, prev = _num(price), _num(prev)
    # `not prev` in the original also caught prev == 0.0 and printed an em-dash
    return DASH if (price is None or prev is None or prev == 0) else f"{(price / prev - 1) * 100:+.2f}%"


def _e(v) -> str:
    """HTML-escape anything interpolated into markup.

    The original interpolated company names, targets and mech strings straight
    into HTML (one of them into a `style='...'` attribute, where a stray quote
    breaks the tag). Nothing here is attacker-controlled today, but the tables
    are one copy-paste away from carrying a yfinance company name.
    """
    return html.escape(str(v), quote=True)


# =============================================================================
# UI
# =============================================================================
TABLE_CSS = """
<style>
table.pm { width:100%; border-collapse:collapse; font-size:15.5px; }
table.pm th { padding:4px 8px; color:#78909C; text-align:left;
              border-bottom:1px solid #37474F; }
table.pm td { padding:5px 8px; color:#B0BEC5; }
table.pm td.tk, table.pm th.tk { font-family:monospace; }
table.pm td.nm, table.pm td.st { font-weight:600; }
table.pm tr { border-bottom:1px solid #263238; }
table.pm.tight { font-size:14px; }
table.pm.tight td { padding:4px 6px; }
.z-buy{color:#00E676}.z-near{color:#FFFFFF}.z-trim{color:#FF5252}
.z-wait{color:#9E9E9E}.z-nodata{color:#616161}
.c-ok{color:#00E676}.c-warn{color:#FFD54F}.c-bad{color:#FF5252}
.c-mut{color:#546E7A}.c-dim{color:#8D6E63}
</style>
"""


def _cell(text, cls: str = "") -> str:
    return f"<td class='{cls}'>{text}</td>" if cls else f"<td>{text}</td>"


def status_text(stock: dict, zone: str, price: Optional[float], near_pct: float) -> str:
    lo = _num(stock.get("buy_lo"))
    if zone == ZONE_BUY:
        if lo is not None and price is not None and price < lo:
            return "🟢 IN BUY RANGE (below — getting lucky)"
        return "🟢 IN BUY RANGE"
    if zone == ZONE_NEAR:
        return f"⚪ WITHIN {near_pct:.0f}% OF BUY TOP"
    if zone == ZONE_TRIM:
        return "🔴 TRIM / SELL ZONE"
    if zone == ZONE_NODATA:
        return "NO DATA"
    return "wait"


# --- precomputed static cells -------------------------------------------------
# Five of the thirteen cells (ticker, company, buying range, trim, target) and
# both status variants depend only on (ticker, zone, near_pct) — they never
# change between reruns. Building them once removes ~400 `html.escape` calls and
# ~500 function calls per render. Measured: escaping every cell inline cost more
# than the string formatting it protected.
_STATIC_CELLS: dict[tuple, tuple] = {}


def static_cells(stock: dict, zone: str, near_pct: float) -> tuple:
    """(ticker, name, status_plain, status_lucky, range, trim, target) as HTML.

    Memoised on (ticker, zone, near_pct). `status_lucky` is the
    "below — getting lucky" variant, chosen at render time from the price.
    """
    key = (stock["t"], zone, near_pct)
    got = _STATIC_CELLS.get(key)
    if got is not None:
        return got
    cls = ZONE_CLASSES.get(zone, "z-wait")

    mech, trim = stock.get("mech"), _num(stock.get("trim"))
    trim_txt = DASH if trim is None else fmt_money(trim) + (
        f" <span class='c-mut' style='font-size:13px;'>({_e(mech)})</span>" if mech else "")

    got = (
        _cell(_e(stock["t"]), "tk"),
        _cell(_e(display_name(stock, zone)), f"nm {cls}"),
        _cell(_e(status_text(stock, zone, None, near_pct)), f"st {cls}"),
        _cell(_e(status_text(stock, zone, -1e18, near_pct)), f"st {cls}"),
        _cell(_e(fmt_range(stock))),
        _cell(trim_txt),
        _cell(_e(stock.get("target") or DASH)),
    )
    _STATIC_CELLS[key] = got
    return got


def build_row(stock, price, prev, zone, near_pct, ath, metrics: Optional[dict] = None) -> str:
    """One monitor row. `metrics` is the precomputed derived dict; passing None
    renders the '—' path.

    Signature change: the last argument used to be the raw `.info` dict, which
    forced every row to recompute EGF and FScore. It is now the precomputed
    metrics dict, produced once per ticker per day.
    """
    m = metrics or {}
    cls = ZONE_CLASSES.get(zone, "z-wait")
    tk, name, st_plain, st_lucky, rng, trim, target = static_cells(stock, zone, near_pct)

    below_lo = False
    if zone == ZONE_BUY and price is not None:
        lo = _num(stock.get("buy_lo"))
        below_lo = lo is not None and price < lo

    egf = m.get("egf")
    if egf is None:
        egf_cell = _cell(DASH, "c-mut")
    else:
        egf_cell = _cell(fmt_g(egf), "c-ok" if egf > 0 else "c-bad")

    score = m.get("score")
    if score is None:
        score_cell = _cell(DASH, "c-mut")
    else:
        score_cell = _cell(f"{score}/10", f"st {'c-ok' if score >= 7 else ('c-warn' if score >= 4 else 'c-bad')}")

    return (
        "<tr>" + tk + name
        + f"<td class='{cls}'>{fmt_money(price)}</td>"
        + f"<td>{fmt_pct_day(price, prev)}</td>"
        + (st_lucky if below_lo else st_plain)
        + rng + trim + target
        + f"<td>{fmt_n(m.get('fwd_pe'))}</td>"
        + f"<td>{fmt_n(m.get('peg'))}</td>"
        + egf_cell + score_cell
        + f"<td>{fmt_ath(price, ath)}</td></tr>"
    )


def build_static_row(stock) -> str:
    """Non-US / unmonitored row (e.g. BESI) — static info, gray."""
    # Column order must match table_html(): ticker, name, price, day%, status,
    # range, trim, target, then the five metric columns. (An earlier draft had
    # 3 blanks before the status and 4 after the target — the parity suite
    # caught the resulting one-column misalignment.)
    cells = (
        [_cell(_e(stock["t"]), "tk c-dim"), _cell(_e(stock["name"]), "nm c-dim")]
        + [_cell(DASH, "c-dim")] * 2                      # price, day %
        + [_cell("NOT MONITORED (non-US)", "z-nodata")]   # status
        + [_cell(_e(fmt_range(stock)), "c-dim")]          # buying range
        + [_cell(DASH, "c-dim")]                          # trim
        + [_cell(_e(stock.get("target") or DASH), "c-dim")]
        + [_cell(DASH, "c-dim")] * 5                      # fwd pe .. % from ATH
    )
    return "<tr>" + "".join(cells) + "</tr>"


def table_html(rows, target_label: str = "12m Target") -> str:
    heads = ["Ticker", "Company", "Price", "Day %", "Status", "Buying Range", "Trim ≥",
             target_label, "Fwd P/E", "PEG", "EGF proxy", "FScore", "% from ATH"]
    th = "".join(f"<th{' class=tk' if i == 0 else ''}>{_e(h)}</th>"
                 for i, h in enumerate(heads))
    return (TABLE_CSS + "<table class='pm'><tr>" + th + "</tr>"
            + "".join(rows) + "</table>")


def render_zone_strip(zone_map, quotes, near_pct) -> None:
    """Compact header: the SYMBOLS in each zone (small font)."""
    groups = [
        ("🟢 IN BUY RANGE", ZONE_BUY, ZONE_CLASSES[ZONE_BUY]),
        (f"⚪ WITHIN {near_pct:.0f}% OF BUY TOP", ZONE_NEAR, ZONE_CLASSES[ZONE_NEAR]),
        ("🔴 TRIM / SELL ZONE", ZONE_TRIM, ZONE_CLASSES[ZONE_TRIM]),
        ("NO DATA", ZONE_NODATA, ZONE_CLASSES[ZONE_NODATA]),
    ]
    lines = []
    for label, zone, cls in groups:
        tickers = sorted(t for t, z in zone_map.items() if z == zone)
        if not tickers:
            continue
        lines.append(
            f"<div style='font-size:14px; line-height:1.55; margin:1px 0;'>"
            f"<span style='font-weight:600;'>{_e(label)} "
            f"<span class='c-mut'>({len(tickers)})</span>:</span> "
            f"<span class='{cls}' style='font-family:monospace;'>"
            f"{_e(' '.join(tickers))}</span></div>")
    latest = None
    for q in (quotes or {}).values():
        if q and q[2] is not None and (latest is None or q[2] > latest):
            latest = q[2]
    if latest is not None:
        lines.append(f"<div class='c-mut' style='font-size:13px; margin-top:2px;'>"
                     f"quotes as of {_e(latest.strftime('%d %b %H:%M'))} ET</div>")
    st.markdown(TABLE_CSS +
                "<div style='border:1px solid #263238; border-radius:8px; padding:8px 12px; "
                "margin-bottom:12px;'>" + "".join(lines) + "</div>",
                unsafe_allow_html=True)


def render_fundamentals_tab(stocks_list, quotes, zones, funds, metrics_by_ticker,
                            wacc: float = 0.095, terminal_g: float = 0.025):
    """Full metric set + DCF + analyst view."""
    headers = ["Ticker", "Company", "Price", "Fwd P/E", "Trail P/E", "PEG",
               "EGF proxy", "EPS YoY", "Rev growth", "Gross M", "Oper M",
               "Profit M", "ROE", "P/B", "P/S", "Debt ratio", "Cash", "Debt",
               "FCF", "Mkt Cap", "FScore", "DCF/sh", "DCF upside", "Analyst Rec",
               "Analyst Tgt", "Tgt upside"]
    head = "".join(f"<th{' class=tk' if i == 0 else ''}>{_e(h)}</th>"
                   for i, h in enumerate(headers))

    rows = []
    for s in stocks_list:
        t = s["t"]
        q = (quotes or {}).get(t)
        price = q[0] if q else None
        zone = zones.get(t, ZONE_WAIT)
        m = metrics_by_ticker.get(t) or {}
        cls = ZONE_CLASSES.get(zone, "z-wait")

        score = m.get("score")
        score_html = DASH if score is None else (
            f"<span class='{'c-ok' if score >= 7 else ('c-warn' if score >= 4 else 'c-bad')}'>"
            f"{score}/10</span>")
        egf = m.get("egf")
        egf_html = DASH if egf is None else (
            f"<span class='{'c-ok' if egf > 0 else 'c-bad'}'>{egf * 100:+.1f}%</span>")

        ratio, basis = debt_ratio(funds.get(t) or {})
        de = DASH if ratio is None else f"{ratio:.1f}"

        dcf = dcf_fair_value(funds.get(t) or {}, wacc, terminal_g)
        if dcf is None or price is None:
            dcf_html, dcf_up_html = DASH, DASH
        else:
            up = dcf / price - 1.0
            col = "c-ok" if up > 0.05 else ("c-bad" if up < -0.05 else "c-warn")
            dcf_html, dcf_up_html = fmt_money(dcf), f"<span class='{col}'>{up * 100:+.1f}%</span>"

        rec, rec_color = fmt_recommendation((funds.get(t) or {}).get("recommendationKey"))
        rec_html = f"<span class='{'c-mut' if rec == DASH else ''}' style='color:{rec_color};'>{_e(rec)}</span>"
        tgt = m.get("target")
        tgt_html = fmt_money(tgt) if (tgt and price) else DASH
        tgt_up = f"{(tgt / price - 1.0) * 100:+.1f}%" if (tgt and price) else DASH

        cells = [
            _e(t), _e(display_name(s, zone)), fmt_money(price),
            fmt_n(m.get("fwd_pe")), fmt_n(m.get("ttm_pe")), fmt_n(m.get("peg")),
            egf_html, fmt_g(m.get("eps_yoy")), fmt_g(m.get("rev_g")),
            fmt_g(m.get("gross_m")), fmt_g(m.get("oper_m")), fmt_g(m.get("profit_m")),
            fmt_g(m.get("roe")), fmt_n(m.get("pb")), fmt_n(m.get("ps")),
            de, fmt_usd_b(m.get("cash")), fmt_usd_b(m.get("debt")),
            fmt_usd_b(m.get("fcf")), fmt_usd_b(m.get("mcap")),
            score_html, dcf_html, dcf_up_html, rec_html, tgt_html, tgt_up,
        ]
        tds = "".join(
            _cell(c, f"{cls} nm" if i == 1 else ("tk" if i == 0 else ""))
            for i, c in enumerate(cells))
        rows.append(f"<tr>{tds}</tr>")

    st.markdown(TABLE_CSS + "<table class='pm tight'><tr>" + head + "</tr>"
                + "".join(rows) + "</table>", unsafe_allow_html=True)

    # ── column explanations + FScore methodology ────────────────────────────
    st.markdown("---")
    st.subheader("📚 What the columns mean")
    expl = [
        ("Fwd P/E", "price ÷ next-12-months EPS estimate — the multiple you pay for FORWARD earnings."),
        ("Trail P/E", "price ÷ last 4 quarters' EPS — the multiple for CURRENT earnings."),
        ("PEG", "P/E ÷ growth rate — under ~1 cheap for its growth, over ~2 expensive (rule of thumb)."),
        ("EGF proxy", "forward vs trailing EPS growth (fwdEps ÷ trailEps − 1). Proxy for Walayat's EGF; "
                      "his exact figure uses next-quarter estimates yfinance doesn't expose. Green = growing."),
        ("EPS YoY", "last quarter's earnings vs the same quarter a year ago."),
        ("Rev growth", "last quarter's revenue vs a year ago."),
        ("Gross / Oper / Profit M", "profit left at each stage of the income statement, as % of revenue."),
        ("ROE", "return on equity — how efficiently shareholder capital is put to work."),
        ("P/B", "price ÷ book value. P/S: price ÷ trailing 12m revenue."),
        ("Debt ratio", f"total debt ÷ total equity where Yahoo reports equity, else ÷ total cash. "
                       f"The header shows which basis each figure uses (D/E vs D/C). Cash / Debt / FCF in $B."),
        ("Mkt Cap", "market value of the whole company."),
        ("DCF/sh", "COMPUTED fair-value estimate, TWO-STAGE: 5 years of explicit growth at the "
                    "reported rate (capped 25%), a 5-year linear fade to the terminal rate, then a "
                    "perpetuity; + cash − debt, per share. n/a when FCF ≤ 0 or the discount margin "
                    "is not positive. A rough anchor, NOT a price target."),
        ("Analyst Rec / Tgt", "Yahoo's aggregated analyst rating and mean 12-month price target "
                              "with implied upside vs the current price."),
    ]
    st.markdown(TABLE_CSS +
                "<table class='pm tight'>" + "".join(
                    f"<tr><td style='width:170px;'><span style='color:#90CAF9;font-weight:600;'>"
                    f"{_e(k)}</span></td><td>{_e(v)}</td></tr>" for k, v in expl)
                + "</table>", unsafe_allow_html=True)

    st.subheader("⭐ FScore — the 0-10 fundamentals score, explained")
    st.markdown(
        "Walayat maintains a time-consuming hand-built 0-10 Fundamentals column. FScore is a "
        "transparent, automatable stand-in built from the same ideas (PE, EPS, revenue, cash "
        "flow, ROE). **+1 point for each check passed; missing data counts as a FAIL, so the "
        "score skews conservative:**")
    checks = [
        "1. EGF proxy > 0 — forward EPS above trailing EPS.",
        "2. EPS YoY > 0 — last quarter's earnings beat the year-ago quarter.",
        "3. Revenue growth > 0.",
        "4. Profit margin > 10%.",
        "5. ROE > 15%.",
        "6. Forward P/E < trailing P/E — earnings rising into the multiple.",
        "7. Cash > debt — balance-sheet cushion.",
        "8. Free cash flow > 0.",
        "9. Gross margin > 30% — pricing power / quality of the business.",
        "10. PEG between 0 and 2 — valuation not detached from growth.",
    ]
    for c in checks:
        st.markdown(f"- {c}")
    st.caption(
        "Reading it: 8-10 green = strong fundamentals (his 'epic' territory); 4-7 amber = mixed, "
        "check the individual columns; 0-3 red = weak — be extra demanding on the buying range. "
        "Fundamentals refresh once per day.")


def _how_to_box(title: str, bullets: list[str]) -> None:
    """Standard 'how to use this tab' callout on every analysis tab."""
    items = "".join(f"<li style='margin:4px 0;'>{_e(b)}</li>" for b in bullets)
    st.markdown(
        f"<div style='border:1px solid #37474F; border-radius:10px; padding:12px 16px; "
        f"background:rgba(144,202,249,0.06); margin-bottom:14px;'>"
        f"<div style='color:#90CAF9;font-weight:700;margin-bottom:6px;'>{_e(title)}</div>"
        f"<ul style='margin:0; padding-left:18px; color:#B0BEC5; font-size:14px;'>{items}</ul>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_market_overview_tab(overview: dict, quotes: dict, zones: dict,
                               metrics_by_ticker: dict, near_pct: float) -> None:
    """Upcoming earnings, week lookback, macro sentiment, actionable cross-read."""
    st.subheader("🌐 Market Overview")
    _how_to_box("How to use this tab", [
        "Check Upcoming Earnings BEFORE the open of an earnings week — do not add size into a print you cannot price.",
        "Use Week Lookback to see which names already moved: a stock that dumped into its buy range is a candidate; one that ripped into trim is a candidate to scale out.",
        "Read macro gauges (VIX, DXY, crude, gold) as CONTEXT, not triggers — Walayat times individual stocks off valuations, not the S&P.",
        "Cross-read zone strip + briefs: BUY + strong EGF + quiet macro > BUY into a VIX spike only if you still have powder dry.",
        "Every figure here is from Yahoo Finance or Walayat's published sheet. Missing = em-dash, never filled in.",
    ])

    benches = (overview or {}).get("benchmarks") or {}
    earns = (overview or {}).get("earnings") or {}
    week = (overview or {}).get("week") or {}

    # ── Macro strip ────────────────────────────────────────────────────────
    st.markdown("##### Macro snapshot (Yahoo)")
    cols = st.columns(len(MARKET_BENCH))
    for i, t in enumerate(MARKET_BENCH):
        b = benches.get(t) or {}
        name = MARKET_BENCH_NAMES.get(t, t)
        px = b.get("price")
        d = b.get("day_pct")
        w = b.get("week_pct")
        with cols[i]:
            st.markdown(
                f"<div style='border:1px solid #263238;border-radius:8px;padding:8px 10px;'>"
                f"<div class='c-mut' style='font-size:12px;'>{_e(name)}</div>"
                f"<div style='font-size:18px;font-weight:700;color:#E8EDF4;'>"
                f"{_e(fmt_money(px) if t not in ('^VIX',) else (fmt_n(px, 2) if px is not None else DASH))}"
                f"</div>"
                f"<div style='font-size:12px;'>"
                f"<span class='{'c-ok' if (d or 0) > 0 else ('c-bad' if (d or 0) < 0 else 'c-mut')}'>"
                f"D {_e(fmt_g(d) if d is not None else DASH)}</span>"
                f"&nbsp;·&nbsp;"
                f"<span class='{'c-ok' if (w or 0) > 0 else ('c-bad' if (w or 0) < 0 else 'c-mut')}'>"
                f"W {_e(fmt_g(w) if w is not None else DASH)}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # Sentiment one-liner from VIX level (descriptive, not predictive)
    vix = (benches.get("^VIX") or {}).get("price")
    if vix is not None:
        if vix >= 30:
            mood = f"VIX at {vix:.1f} — fear regime. Historically when Walayat has powder dry this is ACCUMULATE territory, not a reason to freeze."
        elif vix >= 20:
            mood = f"VIX at {vix:.1f} — elevated. Favour limit orders inside published buy ranges; avoid chase."
        elif vix >= 14:
            mood = f"VIX at {vix:.1f} — normal. Stick to the plan: valuations first, mechanics second."
        else:
            mood = f"VIX at {vix:.1f} — complacent. Trim-into-strength discipline matters more than usual."
        st.info(mood)

    st.markdown("---")

    # ── Upcoming earnings (next 30 days) ───────────────────────────────────
    st.markdown("##### Upcoming earnings & recent prints (portfolio names)")
    st.caption("Source: Yahoo Finance earnings calendar / earnings dates. Dates the feed does not carry render as —.")
    now = pd.Timestamp.now(tz=MARKET_TZ)
    upcoming = []
    recent = []
    for t, info in earns.items():
        if not info or info.get("error"):
            continue
        nxt = info.get("next")
        last = info.get("last")
        name = (BY_TICKER.get(t) or {}).get("name", t)
        if nxt:
            try:
                ts = pd.Timestamp(nxt)
                if ts.tzinfo is None:
                    ts = ts.tz_localize(MARKET_TZ)
                else:
                    ts = ts.tz_convert(MARKET_TZ)
                days = (ts.normalize() - now.normalize()).days
                if -1 <= days <= 45:
                    upcoming.append((ts, days, t, name, info))
            except Exception:
                pass
        if last:
            try:
                ts = pd.Timestamp(last)
                if ts.tzinfo is None:
                    ts = ts.tz_localize(MARKET_TZ)
                else:
                    ts = ts.tz_convert(MARKET_TZ)
                days = (now.normalize() - ts.normalize()).days
                if 0 <= days <= 14:
                    recent.append((ts, days, t, name, info))
            except Exception:
                pass
    upcoming.sort()
    recent.sort(reverse=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Next 45 days**")
        if not upcoming:
            st.caption("No upcoming earnings dates returned by Yahoo for the book right now.")
        else:
            rows = []
            for ts, days, t, name, info in upcoming:
                zone = zones.get(t, ZONE_WAIT)
                cls = ZONE_CLASSES.get(zone, "z-wait")
                est = info.get("est_eps")
                rows.append(
                    "<tr>"
                    + _cell(_e(t), "tk")
                    + _cell(_e(name), f"nm {cls}")
                    + _cell(_e(ts.strftime('%a %d %b %Y')))
                    + _cell(_e(f"in {days}d" if days >= 0 else "today"))
                    + _cell(_e(fmt_n(est, 2) if est is not None else DASH))
                    + _cell(_e(zone), cls)
                    + "</tr>"
                )
            heads = ["Ticker", "Company", "Earnings", "When", "EPS est.", "Zone"]
            th = "".join(f"<th>{_e(h)}</th>" for h in heads)
            st.markdown(TABLE_CSS + "<table class='pm tight'><tr>" + th + "</tr>"
                        + "".join(rows) + "</table>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Reported in the last 14 days**")
        if not recent:
            st.caption("No recent prints in the Yahoo feed for the book.")
        else:
            rows = []
            for ts, days, t, name, info in recent:
                sur = info.get("last_surprise_pct")  # Yahoo already in %
                eps = info.get("last_eps")
                sur_html = DASH
                if sur is not None:
                    col = "c-ok" if sur > 0 else ("c-bad" if sur < 0 else "c-mut")
                    sur_html = f"<span class='{col}'>{sur:+.1f}%</span>"
                rows.append(
                    "<tr>"
                    + _cell(_e(t), "tk")
                    + _cell(_e(name), "nm")
                    + _cell(_e(ts.strftime('%d %b')))
                    + _cell(_e(fmt_n(eps, 2) if eps is not None else DASH))
                    + _cell(sur_html)
                    + "</tr>"
                )
            heads = ["Ticker", "Company", "Reported", "EPS", "Surprise"]
            th = "".join(f"<th>{_e(h)}</th>" for h in heads)
            st.markdown(TABLE_CSS + "<table class='pm tight'><tr>" + th + "</tr>"
                        + "".join(rows) + "</table>", unsafe_allow_html=True)

    st.markdown("---")
    # ── Week lookback ──────────────────────────────────────────────────────
    st.markdown("##### Previous week — portfolio movers")
    st.caption("~5 trading sessions, daily close-to-close from Yahoo. Sorted by move.")
    movers = []
    for t, w in week.items():
        pct = w.get("week_pct")
        if pct is None:
            continue
        s = BY_TICKER.get(t) or {"t": t, "name": t}
        movers.append((pct, t, s.get("name", t), w.get("price"), zones.get(t, ZONE_WAIT)))
    movers.sort()  # losers first
    if not movers:
        st.caption("Week-return data not yet available (cache warming).")
    else:
        losers = movers[:12]
        winners = list(reversed(movers[-12:]))
        lc, rc = st.columns(2)
        def _mv_table(title, items, col):
            with col:
                st.markdown(f"**{title}**")
                rows = []
                for pct, t, name, px, zone in items:
                    cls = ZONE_CLASSES.get(zone, "z-wait")
                    colc = "c-ok" if pct > 0 else "c-bad"
                    brief = (STOCK_BRIEFS.get(t) or (None, None, "", None))[2]
                    brief_s = (brief[:90] + "…") if brief and len(brief) > 90 else (brief or "")
                    rows.append(
                        "<tr>"
                        + _cell(_e(t), "tk")
                        + _cell(_e(name), f"nm {cls}")
                        + _cell(fmt_money(px))
                        + _cell(f"<span class='{colc}'>{pct*100:+.1f}%</span>")
                        + _cell(_e(zone), cls)
                        + _cell(_e(brief_s), "c-mut")
                        + "</tr>"
                    )
                heads = ["Ticker", "Company", "Price", "Week", "Zone", "Brief (sheet)"]
                th = "".join(f"<th>{_e(h)}</th>" for h in heads)
                st.markdown(TABLE_CSS + "<table class='pm tight'><tr>" + th + "</tr>"
                            + "".join(rows) + "</table>", unsafe_allow_html=True)
        _mv_table("Largest declines", losers, lc)
        _mv_table("Largest advances", winners, rc)

    st.markdown("---")
    # ── Zone × brief cross-read ────────────────────────────────────────────
    st.markdown("##### Actionable cross-read (zone × Walayat brief)")
    st.caption("Only names that are currently BUY / NEAR / TRIM and have a published brief.")
    rows = []
    for t, zone in sorted(zones.items(), key=lambda kv: (0 if kv[1]==ZONE_BUY else 1 if kv[1]==ZONE_NEAR else 2 if kv[1]==ZONE_TRIM else 9, kv[0])):
        if zone not in (ZONE_BUY, ZONE_NEAR, ZONE_TRIM):
            continue
        brief = STOCK_BRIEFS.get(t)
        if not brief:
            continue
        moat, action, text, upd = brief
        s = BY_TICKER.get(t) or {"t": t, "name": t}
        q = (quotes or {}).get(t)
        price = q[0] if q else None
        cls = ZONE_CLASSES.get(zone, "z-wait")
        rows.append(
            "<tr>"
            + _cell(_e(t), "tk")
            + _cell(_e(s.get("name", t)), f"nm {cls}")
            + _cell(fmt_money(price))
            + _cell(_e(zone), f"st {cls}")
            + _cell(_e(action or DASH))
            + _cell(_e(moat or DASH))
            + _cell(_e(text))
            + _cell(_e(upd or ""))
            + "</tr>"
        )
    if not rows:
        st.caption("No actionable names with a published brief right now.")
    else:
        heads = ["Ticker", "Company", "Price", "Zone", "Action", "Moat", "Brief", "As of"]
        th = "".join(f"<th>{_e(h)}</th>" for h in heads)
        st.markdown(TABLE_CSS + "<table class='pm tight'><tr>" + th + "</tr>"
                    + "".join(rows) + "</table>", unsafe_allow_html=True)


def render_crypto_tab(crypto_spot: dict, quotes: dict, zones: dict) -> None:
    st.subheader("🪙 Crypto")
    _how_to_box("How to use this tab", [
        "Treat crypto as a SEPARATE accumulate/distribute book — Walayat's rule: the higher you buy, the more likely you hold the bag when it turns.",
        "Spot ladder: start trimming at the published 'Start' level, be 75% out by mid, 90% out by the top. Rebuy only on the trailing correction bands (e.g. BTC −18% to −23% off the last high).",
        "MSTR is a leveraged BTC proxy — use the fair/cheap/extreme bands; do not average into 'Deep Shit Expensive'.",
        "COIN / CRCL swing with BTC; buy inside the sheet ranges, do not chase.",
        "Next-bear BTC target on the sheet is $44k (range $38–48k). That is a PLANNING number, not a live forecast.",
        "All targets and bands below are copied from the Cryptos sheet / 25 Aug portfolio CSV. Live prices are Yahoo. Nothing is simulated.",
    ])

    # Live spot table
    st.markdown("##### Live spot + published ladder")
    rows = []
    for c in sorted(CRYPTO_ASSETS, key=lambda x: x["name"].lower()):
        t = c["t"]
        if c.get("kind") == "spot":
            q = (crypto_spot or {}).get(t) or {}
            price = q.get("price"); day = q.get("day_pct"); week = q.get("week_pct")
        else:
            qq = (quotes or {}).get(t)
            if qq:
                price = qq[0]
                day = (qq[0] / qq[1] - 1.0) if qq[1] else None
                week = None
            else:
                q = (crypto_spot or {}).get(t) or {}
                price = q.get("price"); day = q.get("day_pct"); week = q.get("week_pct")
        buy = DASH
        if c.get("buy_hi") is not None:
            lo, hi = c.get("buy_lo"), c.get("buy_hi")
            buy = f"{fmt_money(lo)} – {fmt_money(hi)}" if lo is not None else f"≤ {fmt_money(hi)}"
        trim = DASH
        if c.get("trim_start") is not None:
            trim = (f"{fmt_money(c['trim_start'])} → {fmt_money(c.get('trim_75'))} → "
                    f"{fmt_money(c.get('trim_90'))}")
        elif c.get("trim") is not None:
            trim = fmt_money(c["trim"])
        # zone-like signal vs buy/trim if we have price
        signal = DASH
        if price is not None and c.get("buy_hi") is not None:
            if price <= float(c["buy_hi"]):
                signal = "🟢 BUY BAND"
            elif c.get("trim_start") is not None and price >= float(c["trim_start"]):
                signal = "🔴 TRIM BAND"
            elif c.get("trim") is not None and price >= float(c["trim"]):
                signal = "🔴 TRIM BAND"
            else:
                signal = "wait"
        rows.append(
            "<tr>"
            + _cell(_e(t), "tk")
            + _cell(_e(c["name"]), "nm")
            + _cell(_e(c.get("kind", "")))
            + _cell(fmt_money(price) if price is not None else DASH)
            + _cell(fmt_g(day) if day is not None else DASH)
            + _cell(fmt_g(week) if week is not None else DASH)
            + _cell(_e(buy))
            + _cell(_e(str(trim)))
            + _cell(_e(signal))
            + "</tr>"
        )
    heads = ["Ticker", "Name", "Kind", "Price", "Day", "Week", "Buy band", "Trim ladder", "Signal"]
    th = "".join(f"<th>{_e(h)}</th>" for h in heads)
    st.markdown(TABLE_CSS + "<table class='pm tight'><tr>" + th + "</tr>"
                + "".join(rows) + "</table>", unsafe_allow_html=True)

    # BTC gift / targets
    btc = next((c for c in CRYPTO_ASSETS if c["t"] == "BTC-USD"), None)
    mstr = next((c for c in CRYPTO_ASSETS if c["t"] == "MSTR"), None)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Bitcoin — published targets (sheet)")
        if btc:
            items = [
                ("Initial target", btc.get("initial_tgt")),
                ("Primary target", btc.get("primary_tgt")),
                ("Second target", btc.get("second_tgt")),
                ("Moon shot", btc.get("moon")),
                ("Next bear target", btc.get("next_bear")),
                ("Next bear range", f"{fmt_money(btc.get('next_bear_lo'))} – {fmt_money(btc.get('next_bear_hi'))}"
                 if btc.get("next_bear_lo") else None),
                ("Dip-buy off last high", f"−{btc.get('dip_lo_pct', 0)*100:.0f}% to −{btc.get('dip_hi_pct', 0)*100:.0f}%"
                 f" off {fmt_money(btc.get('dip_hi_ref'))}" if btc.get("dip_hi_ref") else None),
            ]
            for k, v in items:
                if v is None:
                    continue
                vv = fmt_money(v) if isinstance(v, (int, float)) else v
                st.markdown(f"- **{k}:** {vv}")
            st.caption(btc.get("note") or "")
    with c2:
        st.markdown("##### MSTR — valuation bands (sheet)")
        if mstr:
            items = [
                ("Fair value", mstr.get("fair_value")),
                ("Cheap ≤", mstr.get("cheap")),
                ("Max extreme ≥", mstr.get("extreme")),
                ("Primary target high", mstr.get("primary_tgt")),
                ("Secondary target high", mstr.get("second_tgt")),
                ("Buy band", f"{fmt_money(mstr.get('buy_lo'))} – {fmt_money(mstr.get('buy_hi'))}"),
                ("Trim ladder", f"{fmt_money(mstr.get('trim_start'))} → "
                               f"{fmt_money(mstr.get('trim_75'))} → {fmt_money(mstr.get('trim_90'))}"),
            ]
            for k, v in items:
                if v is None:
                    continue
                vv = fmt_money(v) if isinstance(v, (int, float)) else v
                st.markdown(f"- **{k}:** {vv}")
            st.caption(mstr.get("note") or "")

    st.markdown("---")
    st.markdown("##### Exit strategy — scaling out (published approx prices)")
    st.caption("From the Cryptos sheet: start trimming / 75% exited / 90% exited.")
    exit_rows = []
    for c in CRYPTO_ASSETS:
        if c.get("trim_start") is None:
            continue
        exit_rows.append(
            "<tr>"
            + _cell(_e(c["name"]))
            + _cell(fmt_money(c["trim_start"]))
            + _cell(fmt_money(c.get("trim_75")))
            + _cell(fmt_money(c.get("trim_90")))
            + "</tr>"
        )
    th = "".join(f"<th>{_e(h)}</th>" for h in ["Asset", "Start trimming", "Exited 75%", "Exited 90%"])
    st.markdown(TABLE_CSS + "<table class='pm tight'><tr>" + th + "</tr>"
                + "".join(exit_rows) + "</table>", unsafe_allow_html=True)
    st.info("Rule from the sheet: *The higher you buy the more likely you will be left holding the bag when cryptos turn lower — so have a mechanism to STOP adding and start selling.*")


def render_egf_tab(metrics_by_ticker: dict, quotes: dict, zones: dict) -> None:
    st.subheader("📈 EGF — Earnings Growth Factor")
    _how_to_box("How to use this tab", [
        "EGF is Walayat's core metric: how fast earnings are growing, and which way the growth is travelling.",
        "Positive EGF + cheap PE-range = accumulate candidate. Negative EGF demands a LOW P/E before you add.",
        "EGF-12M is the forward look — strong 12M with a weak live print often means 'wait for the dip, then load'.",
        "The history table is copied from his EGFs sheet (aggregates for the AI book). Live per-ticker EGF proxy is from Yahoo forward vs trailing EPS — a stand-in for his hand-built figure, labelled as a proxy.",
        "Never invent an EGF. If Yahoo has no forward/trailing EPS, the cell is an em-dash.",
    ])

    # History from sheet
    st.markdown("##### AI book EGF history (from EGFs sheet)")
    st.caption("Published aggregates — Ex-Micron / Ex-TSLA columns as on the sheet. Stored as fractions, shown as %.")
    rows = []
    for r in EGF_HISTORY:
        def pct(x):
            return DASH if x is None else f"{x*100:.0f}%"
        rows.append(
            "<tr>"
            + _cell(_e(r["date"]))
            + _cell(_e(f"{r['spx']:,}" if r.get("spx") else DASH))
            + _cell(_e(f"{r['nasdaq']:,}" if r.get("nasdaq") else DASH))
            + _cell(_e(pct(r.get("ai_av"))))
            + _cell(_e(pct(r.get("ai_12m"))))
            + _cell(_e(pct(r.get("pe"))))
            + _cell(_e(pct(r.get("sec_av"))))
            + _cell(_e(pct(r.get("sec_12m"))))
            + _cell(_e(pct(r.get("sec_pe"))))
            + _cell(_e(r.get("comments") or ""))
            + "</tr>"
        )
    heads = ["Date", "S&P", "Nasdaq", "AI EGF Av", "AI EGF 12M", "PE range Av",
             "Sec EGF", "Sec EGF 12M", "Sec PE range", "Comments"]
    th = "".join(f"<th>{_e(h)}</th>" for h in heads)
    st.markdown(TABLE_CSS + "<table class='pm tight'><tr>" + th + "</tr>"
                + "".join(rows) + "</table>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Live EGF proxy by ticker (Yahoo forward÷trailing EPS − 1)")
    st.caption("Sorted by EGF descending. Green = growing forward EPS. This is a PROXY for Walayat's EGF (he uses next-quarter estimates yfinance does not always expose).")
    items = []
    for t, m in (metrics_by_ticker or {}).items():
        egf = m.get("egf")
        if egf is None:
            continue
        s = BY_TICKER.get(t) or {"t": t, "name": t}
        q = (quotes or {}).get(t)
        price = q[0] if q else None
        items.append((egf, t, s.get("name", t), price, m, zones.get(t, ZONE_WAIT)))
    items.sort(reverse=True)
    if not items:
        st.caption("No EGF proxy available yet (fundamentals cache warming, or Yahoo omitted EPS).")
    else:
        rows = []
        for egf, t, name, price, m, zone in items:
            cls = ZONE_CLASSES.get(zone, "z-wait")
            egf_html = f"<span class='{'c-ok' if egf > 0 else 'c-bad'}'>{egf*100:+.1f}%</span>"
            rows.append(
                "<tr>"
                + _cell(_e(t), "tk")
                + _cell(_e(name), f"nm {cls}")
                + _cell(fmt_money(price))
                + _cell(egf_html)
                + _cell(fmt_g(m.get("eps_yoy")))
                + _cell(fmt_g(m.get("rev_g")))
                + _cell(fmt_n(m.get("fwd_pe")))
                + _cell(fmt_n(m.get("peg")))
                + _cell(_e(f"{m.get('score')}/10" if m.get("score") is not None else DASH))
                + _cell(_e(zone), cls)
                + "</tr>"
            )
        heads = ["Ticker", "Company", "Price", "EGF proxy", "EPS YoY", "Rev g",
                 "Fwd P/E", "PEG", "FScore", "Zone"]
        th = "".join(f"<th>{_e(h)}</th>" for h in heads)
        st.markdown(TABLE_CSS + "<table class='pm tight'><tr>" + th + "</tr>"
                    + "".join(rows) + "</table>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Reading EGF with the rest of the sheet")
    st.markdown("""
- **EGF rising + PE % of range low** → classic accumulate setup.
- **EGF falling + PE % of range high** → distribute / stay in cash on that name.
- **EGF negative** → only buy at the bottom of the published range (or not at all).
- **EGF-12M >> live EGF** → growth is expected to re-accelerate; prefer waiting for a dip inside the buy range rather than chasing.
- Pair this tab with **Fundamentals** (DCF / FScore) and **Monitor** (zone) before sizing.
""")


def render_big_picture_tab() -> None:
    st.subheader("🖼️ Big Picture")
    _how_to_box("How to use this tab", [
        "Read this when you feel the urge to sell EVERYTHING or buy EVERYTHING — it is the antidote to headless-chicken mode.",
        "The long-run S&P table shows that even buying the 2000 top still compounded ~8.5%/yr to Dec 2023. Time in > timing.",
        "Use it to size psychology, not orders: keep powder dry, never sell the core at a loss, let the 10X brigade compound.",
        "FX matters for non-US investors — a 'clever' GBP exit can cost 15% on the round trip. Factor it before large trims.",
        "All figures below are from Walayat's Big Picture sheet (reference level 4,755 on 22 Dec 2023). Not live-updated.",
    ])

    st.markdown(
        f"<div style='border:2px solid #FFB300; border-radius:10px; padding:14px 18px; "
        f"background:rgba(255,179,0,0.07); margin-bottom:16px;'>"
        + "".join(f"<div style='color:#FFD54F; font-size:15px; margin:6px 0;'>▸ {_e(m)}</div>"
                  for m in BIG_PICTURE_MANTRAS)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("##### S&P long-run — annualised gain to 22 Dec 2023 (level 4,755)")
    st.caption("Source: Big Picture sheet. 'Even the worst time to buy stocks in modern history still yields 8.5% per annum.'")
    rows = []
    for r in BIG_PICTURE_GAINS:
        pct = DASH if r.get("pct") is None else f"{r['pct']*100:.0f}%"
        av = DASH if r.get("av_yr") is None else f"{r['av_yr']*100:.1f}%"
        rows.append(
            "<tr>"
            + _cell(_e(r["label"]))
            + _cell(_e(f"{r['level']:,.2f}" if isinstance(r["level"], float) else f"{r['level']:,}"))
            + _cell(_e(pct))
            + _cell(_e(av))
            + _cell(_e(r.get("note") or ""))
            + "</tr>"
        )
    heads = ["From", "S&P level", "% gain to Dec 2023", "Av / yr", "Note"]
    th = "".join(f"<th>{_e(h)}</th>" for h in heads)
    st.markdown(TABLE_CSS + "<table class='pm tight'><tr>" + th + "</tr>"
                + "".join(rows) + "</table>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Operating rules that fall out of the big picture")
    for title, bullets in [
        ("Stay invested in the mega-trend", [
            "AI stocks lead the indices both ways — never time stock buys off the S&P/Nasdaq alone.",
            "50% drawdowns are normal for good stocks; powder dry (~15% cash target) is how you buy them.",
            "The 10X brigade is the pure expression of invest-and-forget.",
        ]),
        ("Accumulate / distribute, don't flip", [
            "Buy valuations, not price predictions. Scale heavier the deeper it falls inside the range.",
            "Trim into strength; never sell the investing book at a loss (works ~8/10).",
            "No stop-losses on investing positions — they do the opposite of accumulating.",
        ]),
        ("Psychology", [
            "The news is always bad — that is what gets eyeballs. Trust the metrics.",
            "Your private-investor advantage: no redemptions, no benchmark. Use it.",
            "When you feel FOMO or fear, open this tab, then open Monitor and execute the plan on the page.",
        ]),
    ]:
        st.markdown(f"**{title}**")
        for b in bullets:
            st.markdown(f"- {b}")

    st.caption("Distilled from the Big Picture tab of the AI Tech Stocks Portfolio spreadsheet "
               "(Nadeem Walayat, MarketOracle.co.uk).")



def inject_theme() -> None:
    """[THEME] main window = deep blue, sidebar = deep plum (user preference)."""
    st.markdown(
        """
        <style>
        .stApp { background-color: #0A1B30; color: #E8EDF4; }
        .stApp h1, .stApp h2, .stApp h3 { color: #DDE7F5; }
        section[data-testid="stSidebar"] { background-color: #241A2E; }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stRadio, section[data-testid="stSidebar"] p
        { color: #EBDFF2; }
        .stMarkdown, .stText, .stCaption { color: #E8EDF4; }
        </style>
        """, unsafe_allow_html=True)


def render_rules_tab():
    st.subheader("📖 Rules to Remember — Walayat's Investing Guide & Real Secret, distilled")
    _how_to_box("How to use this tab", [
        "Read the mantra first: ACCUMULATE when CHEAP → DISTRIBUTE when EXPENSIVE.",
        "The 6 Real Secrets are about YOU (skill, focus, money management) — not indicators.",
        "Use the guide groups as a pre-trade checklist: plan, valuations, trim rules, psychology, metrics.",
        "When emotions spike, come here, then execute the levels on Monitor — do not improvise.",
    ])
    st.markdown(
        f"<div style='border:2px solid #FFB300; border-radius:10px; padding:12px 16px; "
        f"background:rgba(255,179,0,0.07); font-size:16px; color:#FFD54F; font-weight:600;'>"
        f"🔑 {_e(MANTRA)}</div><br>", unsafe_allow_html=True)
    st.markdown("**The 6 Real Secrets for Successful Trading**")
    for s in REAL_SECRETS:
        st.markdown(f"- {s}")
    st.divider()
    col1, col2 = st.columns(2)
    for i, (title, bullets) in enumerate(GUIDE_GROUPS):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"**{title}**")
            for b in bullets:
                st.markdown(f"- {b}")
            st.markdown("")
    st.caption("Distilled from the 'Real Secret' (Jan 2019) and 'Investing Guide' tabs of the "
               "AI Tech Stocks Portfolio spreadsheet — originals there for the full text.")


# =============================================================================
# APP
# =============================================================================
# Bump this whenever DataStore's public API changes. Streamlit's
# @st.cache_resource keeps the live instance across reruns AND across code
# pushes on Cloud until the process restarts — an old instance missing new
# methods (e.g. overview / crypto_spot added in v3) raises AttributeError.
STORE_VERSION = "v3.1-overview-crypto"


@st.cache_resource
def _store(_version: str = STORE_VERSION) -> DataStore:
    """One store per server process. `cache_resource` (not `cache_data`) so the
    background thread pool and its in-flight futures survive reruns.

    `_version` is part of the cache key: bump STORE_VERSION to force a fresh
    DataStore after a deploy that adds methods, without waiting for a reboot.
    """
    return DataStore()


def _get_store() -> DataStore:
    """Return a DataStore that is guaranteed to expose the current API.

    Self-heals the Streamlit Cloud case where an older cached instance is still
    alive after a push that added overview/crypto_spot.
    """
    store = _store(STORE_VERSION)
    if not hasattr(store, "overview") or not hasattr(store, "crypto_spot"):
        try:
            _store.clear()
        except Exception:
            pass
        store = _store(STORE_VERSION)
    return store


def main():
    st.set_page_config(page_title="AI Portfolio", layout="wide")
    inject_theme()

    if not YF_OK:
        st.error("❌ yfinance is not installed. Run:  pip install yfinance   — "
                 "this app runs on REAL market data from Yahoo Finance only.")
        st.stop()

    now = pd.Timestamp.now(tz=MARKET_TZ)
    slot = current_snapshot_ts(now)
    nxt = next_snapshot_ts(now)

    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=ms_until_next_snapshot(now),
                       key=f"snapshot_{slot.strftime('%Y%m%d_%H%M')}")
    except ImportError:
        st.sidebar.caption("⚠️ streamlit-autorefresh not installed — data refreshes when the app re-opens.")

    st.title("📊 AI Portfolio")
    st.caption(
        "Accumulate the dumps, distribute the pumps  •  GREEN/ALL CAPS = in buying range (or below)  •  "
        "WHITE = within 10% of buy top  •  RED/ALL CAPS = trim zone  •  Levels: 26 Aug article › author "
        "comments › Trade Wind › portfolio sheet 25 Aug › briefs")

    store = _get_store()

    # Sidebar first: refresh is a user action, and it must be able to clear the
    # disk cache before any load happens.
    with st.sidebar:
        st.header("⚙️ Monitor")
        near_pct = st.slider("Near-zone % above buy top", 1.0, 25.0, NEAR_PCT_DEFAULT, 0.5)
        only_actionable = st.checkbox("Only show actionable (buy / near / trim)", value=False)
        st.divider()
        st.subheader("🧮 DCF assumptions (Fundamentals tab)")
        wacc_pct = st.slider("Discount rate / WACC (%)", 7.0, 14.0, 9.5, 0.25,
                             help="Higher = more conservative fair value.")
        tg_pct = st.slider("Terminal growth (%)", 1.0, 4.0, 2.5, 0.05,
                           help="Long-run perpetual growth; must stay well below the WACC.")
        st.divider()
        if st.button("🔄 Refresh data now"):
            for p in store.cache.root.glob("*.json"):
                if p.name != "ath-state.json":
                    try:
                        p.unlink()
                    except Exception:
                        pass
            store.states.clear()
            try:
                _store.clear()  # drop any stale cached DataStore instance
            except Exception:
                pass
            st.rerun()
        st.caption(
            "Data: Yahoo Finance (yfinance). Levels are Nadeem Walayat's published numbers. "
            "Nothing is simulated or defaulted: a missing figure is an em-dash and is "
            "counted above, never filled in. Monitor only — no orders. Not investment advice.")

    slot_key = slot.strftime("%Y-%m-%d_%H-%M")
    day_key = now.strftime("%Y-%m-%d")

    # Kick off refreshes in the background, then render from whatever is cached.
    try:
        store.prefetch(slot_key, day_key)
    except Exception:
        pass  # never block the UI on a background warm

    quotes = store.quotes(slot_key)
    aths = store.aths()
    funds, metrics_map = store.fundamentals(day_key)

    market_open = _is_trading_day(now) and 9 * 60 + 30 <= now.hour * 60 + now.minute < 16 * 60
    if market_open:
        st.success(f"🔴 Snapshot {slot.strftime('%H:%M')} ET — prices update 09:30 / 12:00 / 16:00 ET; "
                   f"next update {nxt.strftime('%a %H:%M')} ET.")
    else:
        st.info(f"🌙 MARKET CLOSED — latest snapshot {slot.strftime('%a %d %b %H:%M')} ET. Prices "
                f"update only at 09:30 / 12:00 / 16:00 ET on trading days; re-open the app anytime "
                f"to refresh.")

    prices = {t: (q[0] if q else None) for t, q in quotes.items()}
    zones = compute_zones(prices, near_pct)

    # Failures are data, not exceptions — show them instead of silent em-dashes.
    problems = validate_levels()
    missing = sorted(t for t, q in quotes.items() if not q)
    status_parts = [store.status_line()]
    if missing:
        status_parts.append(f"{len(missing)} ticker(s) with no quote: {', '.join(missing)}")
    st.caption("  •  ".join(p for p in status_parts if p))
    if problems:
        with st.expander(f"⚠️ {len(problems)} level-table data issue(s) (click to review)"):
            for p in problems:
                st.markdown(f"- {_e(p)}")

    # Load overview + crypto (disk-first; prefetch already kicked them off).
    # getattr fallbacks keep a half-upgraded cached store from crashing the app.
    overview = store.overview(day_key) if hasattr(store, "overview") else {}
    crypto_spot = store.crypto_spot() if hasattr(store, "crypto_spot") else {}

    (tab_monitor, tab_overview, tab_crypto, tab_egf,
     tab_fund, tab_big, tab_rules) = st.tabs([
        "📈 Monitor",
        "🌐 Market Overview",
        "🪙 Crypto",
        "📈 EGF",
        "🔬 Fundamentals",
        "🖼️ Big Picture",
        "📖 Rules to Remember",
    ])

    with tab_monitor:
        _how_to_box("How to use Monitor", [
            "GREEN / ALL CAPS = price is inside (or below) the published buying range — accumulate.",
            "WHITE = within the Near-% of the buy top (sidebar slider, default 10%) — get ready, do not chase.",
            "RED / ALL CAPS = at or above the trim level — scale out into strength.",
            "Tables are A–Z by ticker. 10X Brigade at top has 10-year targets and no trim (invest-and-forget).",
            "Cross-check Market Overview (earnings this week) and EGF before sizing a BUY.",
        ])
        render_zone_strip(zones, quotes, near_pct)

        # ── ⭐ 10X BRIGADE (A–Z) ────────────────────────────────────────────
        brigade_sorted = sorted(BRIGADE, key=lambda s: s["t"])
        rows = []
        for s in brigade_sorted:
            if s.get("static"):
                rows.append(build_static_row(s))
                continue
            q = quotes.get(s["t"])
            price, prev = (q[0], q[1]) if q else (None, None)
            rows.append(build_row(s, price, prev, zones.get(s["t"], ZONE_WAIT), near_pct,
                                  aths.get(s["t"]), metrics_map.get(s["t"])))
        st.markdown(
            "<div style='border:2px solid #FFB300; border-radius:10px; "
            "padding:10px 14px 12px; background:rgba(255,179,0,0.06); margin-bottom:16px;'>"
            "<h3 style='color:#FFB300; margin:4px 0 2px;'>⭐ 10X BRIGADE</h3>"
            f"<div style='color:#9E9E9E; font-size:12px; margin-bottom:6px;'>{_e(BRIGADE_NOTE)}</div>"
            + table_html(rows, target_label="10Yr Target") + "</div>",
            unsafe_allow_html=True)

        # ── main list (A–Z) ────────────────────────────────────────────────
        stocks = sorted(PORTFOLIO, key=lambda s: s["t"])
        if only_actionable:
            stocks = [s for s in stocks
                      if zones.get(s["t"]) in (ZONE_BUY, ZONE_NEAR, ZONE_TRIM)]
        st.subheader("Portfolio")
        st.caption("Sorted A–Z by ticker. Levels from 26 Aug article › sheet 25 Aug › briefs.")
        rows = [build_row(s, (quotes.get(s["t"]) or (None, None, None))[0],
                          (quotes.get(s["t"]) or (None, None, None))[1],
                          zones.get(s["t"], ZONE_WAIT), near_pct,
                          aths.get(s["t"]), metrics_map.get(s["t"]))
                for s in stocks]
        st.markdown(table_html(rows), unsafe_allow_html=True)

        with st.expander("ℹ️ Sources, exclusions & crypto reference"):
            st.markdown(
                "**Article (26 Aug 2026)** — primaries (levels win where newer than the sheet) "
                "+ author comments (NVDA stacked orders, BIDU / MRNA / CRM).  \n"
                "**10X Brigade (17 Jul 2026)** — 14 ten-year candidates, boxed at the top, A–Z.  \n"
                "**Portfolio CSV (25 Aug)** — buying ranges + trim mechanisms; filled previously "
                "blank levels (TMO, BHP, CCJ, ALB, UNH, MSTR, …).  \n"
                "**Stocks Briefs** — action/moat commentary + TMO $400–$450.  \n"
                "**Excluded** — non-US listings (BESI static; SMSN.L, SMT.L, WTAI.L/INTL.L, "
                "RBTX.L, UKW.L, BDEV.L, PRX.NV) and delisted (MPW, RDFN).")
            st.markdown(CRYPTO_REFERENCE)
            if missing:
                st.caption("No data (check ticker on Yahoo Finance): " + ", ".join(missing))

    with tab_overview:
        render_market_overview_tab(overview, quotes, zones, metrics_map, near_pct)

    with tab_crypto:
        render_crypto_tab(crypto_spot, quotes, zones)

    with tab_egf:
        render_egf_tab(metrics_map, quotes, zones)

    with tab_fund:
        st.subheader("🔬 Fundamentals — full yfinance metric set")
        _how_to_box("How to use this tab", [
            "Sort mentally by FScore and EGF first, then check DCF upside as a second opinion — not a target.",
            "A name in BUY on Monitor with FScore ≥ 7 and positive EGF is the cleanest accumulate setup.",
            "DCF is two-stage (5y growth → 5y fade → perpetuity). Tune WACC / terminal g in the sidebar; higher WACC = more conservative.",
            "Debt ratio shows D/E when equity is reported, else D/C — read the basis, not just the number.",
            "Table is A–Z by ticker. Missing Yahoo fields stay as em-dashes.",
        ])
        fund_list = sorted(MONITORED, key=lambda s: s["t"])
        render_fundamentals_tab(fund_list, quotes, zones, funds, metrics_map,
                                wacc=wacc_pct / 100.0, terminal_g=tg_pct / 100.0)

    with tab_big:
        render_big_picture_tab()

    with tab_rules:
        render_rules_tab()

    st.caption(
        f"Feed: yfinance  •  prices: snapshot {slot.strftime('%a %d %b %H:%M')} ET  •  next update "
        f"{nxt.strftime('%a %H:%M')} ET  •  {len(MONITORED)} live tickers + "
        f"{sum(1 for s in BRIGADE if s.get('static'))} static + "
        f"{len(DELISTED)} excluded ({', '.join(sorted(DELISTED))} — delisted)  •  "
        f"ATH = running max, refreshed incrementally + folded with today's intraday high  •  "
        f"DCF = two-stage (5y growth → 5y fade → perpetuity)  •  fundamentals daily  •  "
        f"{'holidays ignored (install pandas_market_calendars)' if not _HOLIDAYS else 'NYSE calendar'}  •  "
        f"Monitor only — not investment advice.")


if __name__ == "__main__":
    main()
