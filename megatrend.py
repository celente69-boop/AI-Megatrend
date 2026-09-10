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
VERSION 4 CHANGES (8 Sept 2026 update)
--------------------------------------
* PORTFOLIO SYNCED to the 8 Sept 2026 master sheet + the 8 Sept article
  (Bonds Market Crisis...): NVDA 148-183/trim 237, AMD 180-260, MU
  476-625 (was 132-312), LRCX 156-226 (article raise), AMAT 220-324
  (article raise), QCOM trim ~220, TSLA trim 400 (article), MSFT 336-382,
  META 448-548, GOOG 228-292, TSM 222-322, HPQ/JNJ/COIN/RBLX/MGNI trim
  ladders updated. NFLX added (user: buy under $70). LULU added from the
  article (trim target $170).
* NEW TAB 'Latest Article': the full 8 Sept article as-is, with the stocks
  it mentions highlighted at the top of the tab (live price + zone colour),
  plus an LM Studio chat box (OpenAI-compatible API, default
  http://localhost:1234/v1) to ask questions about the article.
* Crypto notes refreshed from the article (BTC base-case <$58k / 18-mo $48k,
  invalidation >$84k; SOL rebuy $70s, staggered trims from $105).
* v4.2 NEW TAB 'Premarket' (04:00-09:30 ET weekdays): premarket prints per ticker
  with gap vs the previous close, zone AT the premarket price, a 'gapping into
  buy range' callout and a big-movers list. No tab was replaced.
* v4.1 FIX: cache keys embed a fingerprint of the monitored ticker book, so a
  mid-session upgrade (adding LULU/NFLX) can never serve the pre-upgrade
  snapshot for the current slot (which showed the new names as NO DATA).

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
Run:  pip install yfinance streamlit  &&  streamlit run ai_stocks_monitor.py
"""
from __future__ import annotations
import hashlib
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
PREMARKET_TTL_S = 60 * 5       # premarket prints refresh ~5 min while open
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
        note='[TW] trimming cryptos; exposure 126%. Sheet trim $124–$138. [A 8 Sep] trimmed the '
             '$58→$106 pump; sell limits stacked above $100, rebuy limits below.',
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
        note='[TW 11 Aug] in buying range; exposure 124%. [A 4 Sep] bought the 10% dump at $910.',
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
        buy_lo=220.0,
        buy_hi=324.0,
        trim=703.0,
        mech='Within 5% of High',
        section='Secondary',
        note='[A 8 Sep] range nudged higher to $324–$220; small buys at $380 (support); '
             '12m $600–$650. Sheet still shows $202–$276.',
    ),
    dict(
        t='AMD',
        name='AMD',
        buy_lo=180.0,
        buy_hi=260.0,
        trim=585.0,
        mech='Only at ATH',
        target='600',
        section='Primary',
        note='[Sheet 8 Sep] buy $180–$260. [A] dream drop $200; 2x off sub-$300 buys.',
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
        note='[A 8 Sep] light adds at $229, main buys from $201; 12m >$300 (new ATH).',
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
        trim=232.0,
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
        buy_lo=228.0,
        buy_hi=292.0,
        trim=404.0,
        mech='Only at ATH',
        target='430',
        section='Primary',
        note='[Sheet 8 Sep] buy $228–$292. [A] support 332 — break targets 300/272/240.',
    ),
    dict(t='GPN', name='GPN', buy_lo=62.0, buy_hi=68.0, trim=None, section='Medium Risk'),
    dict(t="GSK", name="GSK", buy_lo=None, buy_hi=38.09, trim=68.57, mech="Within 10% of ATH", note="[ATH-derived 30 Aug 2026] Yahoo all-time high $76.19 (1999-01-08). Buy top = 50% off ATH; trim = within 10% of ATH."),
    dict(t='HPQ', name='HP', buy_lo=16.0, buy_hi=18.6, trim=30.0, mech='$28–30', section='Medium Risk',
         note='[Sheet 8 Sep] trim ladder $28–$30.'),
    dict(
        t='IBM',
        name='IBM',
        buy_lo=168.0,
        buy_hi=208.0,
        trim=299.0,
        mech='Within 10% of High',
        section='Secondary',
        note='[A 8 Sep] rebought in the $208–$168 range after selling the $330 fomo pump; '
             '12m ~$280; $235 is not cheap.',
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
        trim=253.0,
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
        note='[A 8 Sep] range UNCHANGED (still expensive at $132); added some at $168; '
             '12m $235–$245.',
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
        t='LULU',
        name='Lululemon',
        buy_lo=None,
        buy_hi=None,
        trim=170.0,
        mech=None,
        section='Article',
        note='[A 8 Sep] added at ~$102 (down 81% off the high); avg cost $168; next trim '
             'target $170; earlier buy zone was $160–$200.',
    ),
    dict(
        t='LRCX',
        name='Lam Research',
        buy_lo=156.0,
        buy_hi=226.0,
        trim=417.0,
        mech='Within 5% of High',
        section='Secondary',
        note='[A 8 Sep] range RAISED to $226–$156 (sheet bottom $150); light adds at $250; '
             '12m $406–$450.',
    ),
    dict(
        t='META',
        name='META',
        buy_lo=448.0,
        buy_hi=548.0,
        trim=717.0,
        mech='Within 10% of High',
        target='~730',
        section='Primary',
        note='[Sheet 8 Sep] buy $448–$548. [A] puke cases below 400 to 360; trim into pumps.',
    ),
    dict(t='MGNI', name='Magnite', buy_lo=8.6, buy_hi=11.6, trim=24.0, mech='$20–24', section='High Risk'),
    dict(t="MRNA", name="Moderna", buy_lo=None, buy_hi=248.74, trim=447.74, mech="Within 10% of ATH", note="[ATH-derived 30 Aug 2026] Yahoo all-time high $497.49 (2021-08-10). Buy top = 50% off ATH; trim = within 10% of ATH."),
    dict(
        t='MSFT',
        name='Microsoft',
        buy_lo=336.0,
        buy_hi=382.0,
        trim=500.0,
        mech='Within 10% of High',
        target='600',
        section='Primary',
        note='[Sheet 8 Sep] buy $336–$382. [A 8 Sep] recent trims; buying opp toward the '
             'range, below = getting lucky.',
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
        buy_lo=476.0,
        buy_hi=625.0,
        trim=1192.0,
        mech='Within 5% of High',
        section='Secondary',
        note='[Sheet 8 Sep] buy $476–$625. [A 8 Sep] holds none; light adds ~$800, support '
             '$740, target zone $625 (50% off high), lower limits to $450; 12m 2x talk.',
    ),
    dict(
        t='NFLX',
        name='Netflix',
        buy_lo=None,
        buy_hi=70.0,
        trim=None,
        mech=None,
        section='User',
        note='[User 9 Sep] buy price under $70.',
    ),
    dict(
        t='NVDA',
        name='NVIDIA',
        buy_lo=148.0,
        buy_hi=183.0,
        trim=237.0,
        mech='Only at ATH',
        target='~275',
        section='Primary',
        note='[Sheet 8 Sep] buy $148–$183, trim ATH $237. [A 8 Sep] smart money sold into '
             'earnings; primed for buying opps into the October window.',
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
        trim=220.0,
        mech='Within 10% of High',
        section='Secondary',
        note='[A 8 Sep] re-accumulate $152–$120, trim ~$220 (sheet trim $234); earnings '
             'contracting — trim the pumps. 12m ~$230 max.',
    ),
    dict(t='RBLX', name='Roblox', buy_lo=33.0, buy_hi=41.0, trim=69.0, mech='$60–69', section='High Risk'),
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
        trim=400.0,
        mech='Within 10% of High',
        section='Secondary',
        note='[A 8 Sep] trim once it breaks above $400 (sheet trim $449); 12m target $450 '
             'after the bear market bottoms.',
    ),
    dict(
        t='TSM',
        name='TSMC',
        buy_lo=222.0,
        buy_hi=322.0,
        trim=479.0,
        mech='Only at ATH',
        target='>500',
        section='Primary',
        note='[Sheet 8 Sep] buy $222–$322; sweet spot ~$330. [A] lightly adding sub 390.',
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
    "(spot + MSTR/COIN/CRCL). Numbers are from the Cryptos sheet, the 8 Sept portfolio "
    "CSV and the 8 Sept article — nothing is simulated."
)

# =============================================================================
# LATEST ARTICLE (8 Sept 2026) — full text stored verbatim for the Article tab
# and as context for LM Studio questions. Source: latest article.csv export.
# =============================================================================
ARTICLE_TITLE = 'Bonds Market Crisis to Trigger Stocks, Housing and Crypto Panic Events, Trend Forecasts'
ARTICLE_DATE = "8 Sept 2026"
ARTICLE_TEXT = r"""Bonds Market Crisis to Trigger Stocks, Housing and Crypto Panic Events, Trend Forecasts

20 hours ago

Executive Summary: Macro Crisis & Accumulation Strategy

·  Macroeconomic Catalyst: A global bond market crisis, fueled by persistent inflation, massive government debt refinancing, and geopolitical conflicts, will force central banks including the US Federal Reserve to hike interest rates, shattering the mainstream consensus of impending rate cuts.

·  Equities Outlook: The S&P 500's current strength is a "nothing burger" propped up by an unsustainable AI capex bubble . A severe market correction is anticipated, with the S&P 500 forecast to drop to target by the end of 2026.

·  The "Holy Grail" Strategy: Investors must ignore the broader index and maintain high cash reserves (around 70%) to mechanically buy deep price deviations in AI stocks (like KLAC, AMAT, and LRCX) and crypto assets (like Bitcoin and Solana), and trimming on the pumps .

·  Global Real Estate Fracture: Rising yields will impact global housing markets differently: causing price stagnation and low inventory in the US, forced-selling and cyclical declines in the UK, bubble-popping corrections in Canada and Australia, and a continuous structural crash in China .

·  Geopolitical Undercurrents: The current economic deterioration, masked by manipulated employment and inflation statistics, mirrors the historical decay of the Roman Empire, signaling a long-term decline of US imperial and financial dominance .

El Presidente Trump threatens to stop the US from trading with EU and China if the Fed does not cut rates. If Trump actually did this then it would be probably one of the most catastrophic things that Trump could do to kill the US Economy, bond prices would spike far higher than folk can imagine, yes we would get our stock market CRASH buying opportunity but unless President TACO backtracked quickly then this trigger a financial crisis along the lines of 2008 that we got a taste of during the Tariffs tantrum, where the bounce back from could take a decade!

So despite noise from the White House demanding rate cuts, instead rate hikes are incoming as has been my base case all year, folk forget that the consensus up until quite recently was that Trumps inside man Warsh would obey his master and cut rates, nope, not going to happen, be prepared for a bond market panic event with the expected rate hikes to deliver further deep deviations in target stocks which looks set to soon manifest as we go into the October crash window all courtesy of the clown in the white house and his Iran War sending inflation expectations soaring across the world, which were already riding high courtesy of tariffs.

The Trump regimes luck has run out and we will see the S&P nothing burger succumb to gravity and meander its way towards my year end target of 6350 as we are now into a global bond market panic, so even if the Fed wanted to keep rates on hold it can't not if everyone else is hiking to attract buyers for their bonds who demand higher yields in face of higher inflation expectations and debt deluge by all. The only reason the Fed has been dithering and not already raised rates is because of the mid-terms so as not to upset the cry baby in the white house, however as my analysis in this article will show a September rate hike has been probable for some time, to get the long end yields down the Fed needs to increase short end yields hence rate hikes.

"Donald Trump has called for interest rates to be cut later this month, claiming higher rates put the US at a "very unfair disadvantage".

The chickens are coming home to roost, all of the damage done by Trump faces a day of reckoning, $40 trillion in debt with a $2 trillion deficit and the clown wants to CUT RATES as if that will encourage investors to BUY the US debt deluge. So whilst we are all eager to buy the dips and have been doing so in target stocks, know this that we will all buy too high!

It's not going to be long before there will be comments asking why x,y,z stock is dropping, what else do folk expect to happen during a BEAR MARKET? Even with the S&P burger riding high at 7700 folk can see the extent of the BEAR MARKET in the spreadsheet column Q, where most stocks are down more than 20% from their highs and many such as Qualcom, LRCX, IBM, KLAC, Intel, and TSLA are down over 30% and there's some down over 40% i.e. AMAT, stock prices drop during a BEAR MARKET! I imagine most target stocks by 6350 will bet at least a further 25% lower which will nudge most down over 50%, definitely most of the secondaries will be 50% off their highs, the likes of TESLA eyeing sub $200 vs it's near $500 high.

This article is a continuation of my last posted a couple of weeks ago, this covers secondary stocks as well as the unfolding bond market crisis as debt bombs are primed to explode not just because of AI out competing corporate and sovereign debt but also not forgetting the fool in the White House who continues to do everything to make America Poor Again with a fresh round of disruptive tariffs, and then we have the War on Iran that has cut off 20% of the worlds oil life blood triggering waves of inflation as fuel prices surge. In fact America got lucky in that the Trump Derangement Presidency hit at the same time as AI capex boom that has acted to mask the damage done to the US economy, it is sheer luck that the US is not already in recession thanks to the AI Mania.

https://www.patreon.com/Nadeem_Walayat/posts/nvidia-earnings-167757592

CONTENTS

Stock Market Index Chaos

·  US FAKE Unemployment Statistics

·  US Bond Market Series of Panic Events in 2026 and 2027

·  US CPI LIE Panic Event

·  Money Printer Goes Brrrr

·  US Rate Hike Incoming

·  Bond Yields Spiking so Why No QE to Lower Yields?

·  UK and Japanese Bond Yields

·  Japanese Yen Crisis illustrated!

·  Yen Dollar Pressure Cooker

·  STOCK MARKET ARMAGEDDON SCENERIO

·  2026 Smells like 1987

·  Stock Market Mid-term Years Seasonal Dumps

·  S&P Sept to December 2026 Trend Forecast

·  State of the BEAR MARKET

·  Rising Bond Yield Impact on Housing Markets

·  US House Prices 12 Month Forecast

·  UK House Prices Forecast

·  China House Prices CRASH

·  Canada Housing Bear Market

·  Australia House Prices Mania Bubble Top

·  The Holy Grail of Investing

·  AI Secondary Stocks

·  1. Micron $1016 - EGFS 150%, 157%, Dir 0%, 0%, PE Ranges 63%, 10%

·  2. LRCX $307 - EGFS 36%, 77%, Dir 18%, 29%, PE Ranges 251%, 118%

·  3. AMAT $455 - EGFS 38%, 42%, Dir 12%, 5%, PE Ranges 185%, 115%

·  4. KLAC $185 - EGFS 18%, 59%, Dir 11%, 16%, PE Ranges 254%, 129%

·  5. AMZN $259 - EGFS 7%, 61%, Dir 12%, 5%, PE Ranges 94%, -7%

·  6. Qualcom $169 - EGFS -23%, -9%, Dir -5%, -23%, PE Ranges 39%, 49%

·  7. TSLA $354 - EGFS -8%, 20%, Dir -4%, -11%, PE Ranges 421%, 338%

·  8. IBM $235 - EGFS -3%, 6%, Dir 14%, -13%, PE Ranges 69%, 59%

·  LULU $103 - We Will All Buy Too High

·  Bitcoin's Most Profitable Bear Market

·  The SOLANA Accumulation and Distribution Game.

·  THE OPERATING SYSTEM

·  The Fall of the Roman US Empire

Stock Market Index Chaos

The stock market indices are chaos personified, appear to be defying the reality of soaring bond yields manifesting in the nothing burgers trading along their highs, basically going nowhere whilst under the hood many target stocks are in severe bear markets or experiencing huge price pumps where one can capitalise on both, which is why I and many patrons are beating the S&P despite being 70% in cash without the risk of drawdown through the mechanisms of buying the deep deviations from the highs when stocks become cheap and then trimming the rips as evidenced by the 10x brigade, up 33% since Mid July after have collapsed earlier in the year on the narrative that AI had killed software stocks, as I often say at the lows all of the news is BAD, whilst at the highs all of the news is euphoric which is why I pay NO attention to that which most focus upon, instead what I look at are the macros, metrics, stock charts and narratives, I pay virtually no attention to the NEWS MEDIA. I don't even watch the news! I can't tell you when I last watched a news channel such as the BBC.

The S&P this year more than most years truly has been a complete nothing burger in terms of reflecting individual stock trends, folk who take their cue from the S&P have missed the 50% dumps and 2x to 3x pumps.

The S&P defies the macros of a destructive Presidency where so far the damage done has been masked by the Hyperscalers capex boom that totals $1 trillion to date, with another $1 trillion expected to be spent over the next 12 months, that's why the S&P has defied the Trump tariffs catastrophe, subversion of democracy, crony capitalism, corruption, disruption of economic alliances, debt deluge, topped off with an unwinnable war on the commands of the parasite that controls America to an extent, a war that was supposed to last 5 days is into it's 7 month that has brought fuel reserves to critical levels and will result in demand destruction. So under the hood the macros are screaming RECESSION INCOMING, but we have the hyperscalers capex boom vaporising cash flows, taking on debt in a mad rush to hoard compute just as they did with software engineers not so many years ago, excess capacity compute power that only has a shelf life of 3 to 4 years! There is a lot of creative accounting going on under the hood which is why the hyper scalier stock prices are subdued with most trading well off their highs, its why Nvidia's excellent earnings report FAILED to ignite much of a rally because of Financialization of the chips sector is inflating a house of cards revenue and earnings bubble.

On paper Nvidia looks dirt cheap, but the stock price is failing to break higher leaving many investors confused, they don't understand that the reason why the Nvidia earnings pump fizzled out is because the smart money SOLD INTO IT! The smart money is distributing whilst the noob money is buying. I expect Nvidia to drop because real earnings are not as good as the numbers being reported, hence we are primed for buying opps as I covered in my last article.

The data centre build out reminds me of post pandemic hiring surge when they took on too many workers, it's lemming-esque behaviour if others are hiring then we got to hire even more, hoarding of workers, and similar is happening with AI compute power, if others are buying Nvidia GPU's then we also have to got buy them, we'll think about how to make use of them later, that's what's happening with the data-centre build out, hoarding of compute power at inflated prices, it's a BUBBLE that will POP because there isn't the demand for all this compute, and the more that gets built and hoarded the lower will be the actual use and thus the bigger the bubble gets vs reality of demand for towards which there is financialisation going on so as to allow customers to expand buying capabilities far beyond which they can make use of. It is an AI MANIA!

So the hyperscalers continue to vaporise their free cash flows transforming cash rich mega-corps into debt addicts, inflating the corporate debt bubbles sending yields soaring as hyperscalers out compete sovereign debt markets, putting long yields under pressure and risk sparking a bond market panic event that I have written to expect many times over the past year, we are well into a slow motion bond market panic event.

And if all of that was not bad enough we have the three horsemen of the IPO Apocalypse where SPAC-X set the ball rolling and following in mid October will be Anthropic's $1 trillion IPO that will add to the SPAC-X liquidity drain, with the OpenAI IPO pushed into Mid 2027, all 3 act to suck money out of the markets as the insiders cash out in waves of unlocks. It does not take rocket science to work out that these IPO's are total poop, alls one needs to do is look at the PE ratios, a peak to trough 90% drop is in store for all three.

SPAC-X pumped, then halved, and now is in a counter trend rally suckering in noobs who think it's now a good buy before it next slumps to well below $100.

It's only a matter of time before the indices play catch up to what's taking place under the hood, in the meantime one cycles between buying and selling opportunities as they come along.

US FAKE Unemployment Statistics

The books are cooked! Latest jobs report, US unemployment rate goes DOWN WHILST 80% of jobs are LOST!

US official BLS unemployment rate of 4.1% is based on -

The participation rate in July 2008 going into the financial crisis was 66%

At 66% participation rate US unemployment rate would be 10.75%

Actual US unemployment rate is near TRIPLE the official fake statistics because folk have been systematically excluded from the stats which makes them worthless to compare the present unemployment rate against the past i.e. at the height of the financial crisis the US unemployment rate was 10%! Today's rate on the same basis is HIGHER than at the peak of the financial crisis!

The Non farm jobs report has become worthless!

The empire is dying whilst the books are being cooked to give the impression of Imperial Might, this is what it was like as the Roman Empire died, in total denial, diluting the gold in the coins, news of fake imperial conquests when in reality the Empire was in retreat. How many times has the Trump regime claimed victory in it's war against Iran?

The US Empire peaked with the dot com bubble, since whence has been in decline, in denial, ever more eager to use that one advantage it has left, the US military. Americans point to the peak of China without realising that they too have peaked and are now an empire in decline lashing out at friend and foe in desperate attempts to halt decline, doing exactly the same as China, cooking the books and publishing fake statistics.

US Bond Market Series of Panic Events in 2026 and 2027

Eight trillion of US government debt needs refinancing during 2026, let that sink in, $8 trillion! This means that rolling over US debt is hoovering up market liquidity at a rate of $800 billion per month hence sending yields across the curve higher sending the Treasury panicking by announcing a pittance of $4 billion of bond buy backs, a literal drop in the ocean aimed towards lowering long rates that is not going to work because of the war inflation fundamentals in force, alls they are doing is printing dollars means higher inflation pushing rates HIGHER not lower, to lower long yields the Fed needs to increase short end rates to encourage short end buying to take pressure off the long end and so a rate hike is incoming.

Trump responded to the bond market crisis in the only way he knows how which was to threaten military action against the bond market! If the bond market took him seriously then you would see yields spike higher.

And there's still folk who think that this clown is a genius playing 4d chess. A hollow Presidency that perpetuates chaos and confusion, all that garbage about replacing Powell with Warsh so rates would be cut.

The problem is the refinancing cliff does not stop with $8 trillion in 2026 because a lot of the refinancing is short-term debt which means everything refinanced for less than 1 year duration gets refinanced during 2027, guess what the refinancing tally will be next year? About $11 trillion! The US has run out of the refinancing runway and thus yields will be higher in 2027! Not forgetting the $2 trillion budget deficit which will add $2 trillion of NEW debt to this year and likely more to next year. Refinancing means there needs to be someone willing to buy all this debt, there isn't as the Norwegian Sovereign wealth fund illustrates that announced a $80 billion drop in their exposure to US debt, similar with the wrecked Gulf states, same with the BoJ dumping treasuries to cap dollar strength.

Rising yields benefit financials and insurance companies, whilst rising inflation commodity and energy stocks, and cash rich stocks with large balances to earn interest on such as Apple. Whilst those with a lot of debt obviously pay the price and thus will have lower future earnings. Cash is King!

US CPI LIE Panic Event

This is why CPI is a LIE - a real inflation spike is underway that the fake stats are not reporting on due to manipulating the date, but is being experienced in the economy as we are seeing it's impact in the bond markets as market rates rise increasing the cost of borrowing all whilst most focus on the Fed funds rate.

The next inflation report is on Friday 11th September, the market expects 3.4% vs 3.4% previous, given what's going on under the hood i.e. the spike in fuel prices then I suspect we will see inflation higher than expected. which will set the scene for apocalypse Wednesday as we have the oil price trending higher that looks set to spike above $100 once more towards $120, then what for inflation expectations? Yields will spike even higher creating snowball run on sovereign debt as bond investors panic sell for the safe haven if cash US Dollars, in fact they don't need to sell, alls they need to do is NOT refinance maturing US debt because in this climate cash is KING!

Money Printer Goes Brrrr

The money printer usually goes brrr because the governments are addicted to spending money they don't have...

US M2 money supply

·  2016 +6.5%

·  2017 +5.3%

·  2018 +3.6% Rate hike tightening

·  2019 +6.6%

·  2020 +24.9%

·  2021 +12.4% Transitory inflation BS

·  2022 -.01% Recession / Bear market

·  2023 -2.3% Recession / Birth of the Bull

·  2024 +3.8% baseline

·  2025 +4% baseline

·  2026 +5.6% accelerating.

However there is a 1 to 2 year lag between M2 and inflation

There is NO tightening, money supply is accelerating which means inflation is not going to come down any time soon, it's going to remain stubbornly high, made worse by the war in inflation fuel price spike, 2022 was this decades first inflation wave which I had been warning to expect in a decade of inflation waves since 2020. We are in the early stages of the next inflation wave hence why the markets have been delusional all year to expect rate cuts when the opposite has always been most probable as I have been iterating since the start of the year.

US Rate Hike Incoming

The consensus has been that the Fed won't hike before the mid-terms elections instead choose to delay, kick the can down the road until after the elections, however in both previous mid-term years the Fed DID hike rates, Sept 2022 by 0.75%, and in the Sept 2018 mid-terms 2018 by 0.25%.

The next rate decision is on 16th September at 7pm UK time, the CME tool has jumped from 33% rate hike probability a week ago to a 64% chance in the wake of 'inflationary' signals.

Though the reality is that the fed funds rate FOLLOWS the market rate as the 2 year bond yield illustrates with current pressure building for an imminent Fed rate hike, the yield is 4.36% vs Fed 3.63%, the 2 year is trending higher which means there will need to be a series of rate hikes.

As above we have a probably higher CPLIE on Friday 11th Sept that would set the stage for the Fed hike rates starting on the following Wednesday which has been my base case all year, many will soon say a rate hike was obvious, usually the same who were saying not so long ago that rate cuts were certain given that Warsh was Trumps puppet at the Fed.

The bottom line is that for the anticipated crash / bear dump to manifest itself then we need the likes of what was largely an unexpected Fed rate hike until very recently, I say largely because whilst it will be a shock to MSM talking heads, the CME tool and the bond yields have been pointing to rate hikes for some time, hence the Fed should raise rates this month.

The fool in the white house calling for rate cuts does not appear to understand that the US can not cut rates if everyone else is RAISING rates, as we will see next week when the Bank of England raises rates on Thursday followed by Japan on Friday and thus throwing the rate hike ball back at the Fed.

The extent of market reaction depends on the statements that accompany the rate hike, i.e. how hawkish is the Fed, are they implying that they are going to pause for a while after the hike or is this one of a series of hikes, anyway regardless it will at least deliver technical sell signals in the indices so as to set the ball rolling, and we will just have to see the degree to which price declines cascade as fomo turns to fear as investors focus on the impact of higher rates on future corporate earnings.

I will just continue doing what I have been doing all year, buying the deep dips as per the buying ranges and selling what pumps such as recent trims of Microsoft, folk can see what I am doing in the Trade Winds article where I update each days buys and sells, so folk should bookmark this link - https://www.patreon.com/Nadeem_Walayat/posts/trade-wind-166506707

Bond Yields Spiking so Why No QE to Lower Yields?

The question that the mainstream media fails to ask: If bond yields are spiraling out of control, why don't the central banks just flip the switch back on the printing press? Why aren't they rushing to "rescue" the market with more Quantitative Easing?

The answer, if you peel back the curtain is that the central bank clowns have painted themselves into a corner.

They want you to believe this is about a "mandate" for price stability. That’s just the cover story. The reality is that we are witnessing the inevitable endgame of a decade of the Great Monetary Ponzi. Here is why the "QE magic" has run out of road:

1. The Printing Press Trap (The Inflation Feedback Loop)

They spent years pumping trillions into the system, flooding the economy with cheap, printed money. That was the easy part. Now, the bill has come due in the form of persistent inflation. If they were to restart QE now to suppress bond yields they would be admitting that the currency itself is being sacrificed.

You cannot print your way out of a debt crisis when the very act of printing destroys the value of the money you are printing. If the CBs restart QE the market will instantly see it for what it is: monetization of the debt. Fiat would crater, inflation expectations would skyrocket, and the bond market would likely crash harder because investors would demand a massive "inflation premium" just to hold onto the debased paper. They are terrified that if they blink they lose control of the narrative completely.

2. The Quantitative Tightening (QT) Charade

Right now, they are trying to play the "responsible adult" with Quantitative Tightening. They are selling bonds trying to prove they aren't just printing money for the Treasury. But the math says its a desperate attempt to normalize a balance sheet that is fundamentally broken. They are withdrawing liquidity, which is precisely why bond yields are soaring. They are stuck in a catch-22:

Stop QT/Restart QE: You save the bond market but kill the currency and hyper-inflate the economy.

Continue QT: You kill the bond market (and mortgage holders) but try to maintain the illusion of being "in control."

3. Fiscal Dominance: The Real Truth

The market knows the truth, even if the Fed won't say it: Fiscal Dominance. The governments are addicted to borrowing, the CBs are effectively trapped as the lenders of last resort. The only reason they aren't doing QE today is because they are praying that something else breaks first, or that inflation magically retreats, giving them an excuse to pivot.

But make no mistake the "system" is failing. We are past the point of subtle policy tweaks. They are hoping to keep the plates spinning long enough to avoid an outright collapse but every move they make is just a different flavour of delay.

The Bottom Line

They aren't doing QE because they know it’s the "nuclear option." If they press that button now they confirm to the entire world that the economy can no longer function without perpetual money printing. They are holding out, hoping the market doesn't realize the emperor has no clothes.

But as observers of these cycles you have to ask yourself: how long can they pretend that the debt is sustainable before the market decides that 5% or 6% yields aren't high enough to compensate for the risk of collapsing fiat?

As you look at these markets you should be asking yourselves: if they are forced to pivot back to QE when the next crisis hits what does that do to the valuation of your holdings? Because when the printing press starts again it won't be to "save the economy, it will be to save the system from itself.

UK and Japanese Bond Yields

It's not just US yields spiking, the white house buffoon has sparked a global inflation panic sending yields soaring across the world, in this war there are not innocent bystanders, we all pay the price!

UK 10 year has nudged over 5%.

What am I doing? Accumulating 3GIL. It's proved a profitable trade before.

Japanese bond yields are also spiking near doubled vs a year ago and doing diddly squat to halt the Yen blood bath which ensures Japanese rate hikes are incoming.

Japanese Yen Crisis illustrated!

UK and Japan have near the same GDP! Near twice the GDP per capita.

This should NOT be possible because Japan has twice the population! And well Japan is technologically well beyond the UK in virtually every metric!

The crisis is that Japan is DIRT CHEAP! The YEN IS DIRT CHEAP! Whilst the British Pound is Expensive! Up near 80% in 6 year.

There is a hard reset coming, it has to happen because The UK and Japan are NOT COMPARABLE in terms of GDP. The pendulum has swing to an EXTREME as a function of Japanese investments abroad, selling yen to buy assets such as tech stocks and properties in London. Which means the GBP is primed for a significant bear market. In terms of immediate direction of travel, GBP should see a downswing into late October the depth of which depends on whether first GBP 1.32 and then 1.30 breaks to target sub 1.25 which will be followed by an upswing into the new year.

Yen Dollar Pressure Cooker

The dollar yen rate determines risk on vs risk off... the higher the dollar goes the better for risk on assets as it feeds the carry trade, however it acts like a pressure cooker which eventually vents stream that we see as risk off events, sending risk on assets sharply lower.

Right now we are coming off an extreme with the pressure cooker primed to pop it's lid as both US and Japanese central banks are attempting to prevent the the dollar from going higher that is contributing to the Japanese bond market crisis that should see the dollar between 14% to 22% lower against the Yen, that delivers the carry trade unwind.

STOCK MARKET ARMAGEDDON SCENERIO

When Japanese Yields EXCEED US Yields. Ahead of which there will be a wake up moment for markets of what's in the pipeline, everyone will rush for the exit at the same time!

The carry trade won't just unwind, it will COLLAPSE, as would the dollar.

The trends are there.... Japanese yields are rising faster then US yields.

The Yen is EXTREMELY WEAK.

You ve got two mechanisms for an EPIC VIOLENT Capital flows back into Japan after decades of investing abroad.

On the brightside it gives one an opp to capitalise on the weak yen with another holiday! Accumulating Yen via the likes of GBJP.

2026 Smells like 1987

Everyone cheering to bring on the CRASH will be puking their guts out when it happens as their YTD gains evaporate, 2026 was never meant to be an UP YEAR!

So folk need to prepare themselves mentally to see ALL of their gains for 2026 VANISH else they will be crying into their pillows at night, wishing they had never looked at a stock ticker.

On the plus side as prices drop your percent cash will naturally go up 😃

Stock Market Mid-term Years Seasonal Dumps

September is the weakest month of the year, with the 2nd half of the month where most of the action is, fits with the fed rate hike on 16th Sept! Mid-terms ADD to weakness due to greater political uncertainty, S&P in September is down - 1.4% all years, and -2% mid-terms. October early trend tends to be volatile but tends to end strong, +1.1% all years, +3% mid-terms.

2026 Trend to date.

Up into end Jan, 10% correction into end March, 20% rally into end of May, stagnating along the highs since which implies distribution / rotation under way, smart money selling.

2025 Trend

Mid Feb top, 20% drop into early April followed by a relentless rally all the way into the end of the year for a gain of +16.4%.

Both of the last 2 mid-term years 2022 and 2018 had rate hikes in September!

2022 Trend

Volatile 9 month bear market, -27% to the low. From Mid August to Mid October S&P fell by 20%.

2018 Trend

Late Jan top, 11% correction into Mid Feb, +15% rally trend into early October, up about 9% on the year, with the third Fed rate hike on 26th Sept. S&P fell 20% from early October into the end of the year.

So both mid-term years saw a 20% drop where the 2018 drop in particular caught most by surprise.

S&P Sept to October 2026 Trend Forecast

The seasonal pattern suggests September be a strong down month that continues into October where the market tends to make a bottom. The S&P peaked on 17th August at 7830, so a 2 month dump could see the S&P lower into Mid October by 10% down to 7000, put in a bottom before rallying into late November and then comes the big surprise which takes the S&P down to my target of 6350 by the end of 2026.

So we could see an initial 2 month correction morph into a 4 month bear market going into January 2027 delivering a 20% drop which would be like 2018 that would catch virtually everyone by surprise achieving my year end forecast of 6350 which I first mentioned as a possibility several months ago.

Which implies that August should have been THE TOP, hence my statements that 2026 rhymes like 1987.

State of the BEAR MARKET

Spreadsheet - Column Q - Stocks over 20% off their highs.

Link - https://docs.google.com/spreadsheets/d/1PuPxdy4sGJwC1AHfbSfHdPRSaQ43OeqEIO8L4I7-JAU/edit?usp=sharing

Primaries - META leads the way -27%, AVGO -25%, AMD -20%

Secondaries - KLAC -43%, AMAT -38%, Qualcom -37%, Intel -37%, LRCX -31%, TS:LA -30%, folk will find reasons NOT to accumulate even KLAC! I added some Friday. IBM -29%, Micron -26%

Medium Risk - FSLR -36%, WDC -43%, ON -46%, CRUS -40%, GFS -52%, LOG -30%.

High Risk - Most stocks as per spreadsheet.

This is why most days I am buying something, regardless of how high the S&P trades.

Rising Bond Yield Impact on Housing Markets

Cash is King especially when rates are rising. Folk can appear rich on paper, own lots of properties but properties are illiquid, if you can't earn or borrow to service ones debts and costs then what you have is a House of Cards that could collapse when faced with liquidity issues, this is how home builders tend to go bust, have lots of properties under development but unable to finance completion, it's how China's Evergrande went bust! So when rates are rising and credit is tightening that's when folk are at the risk of losing it all. Cash is King!

Folk during the mania are eager to put deposits down on properties only to find out that the developer has gone bust and their deposits are gone!

The US housing market is most impacted by the 10 year yield whilst the UK is impacted by the 2 to 5 year yield.

Thus rising short end yields impact the UK harder then the US because the UK relies heavily on short-term 2 year and 5 year fixed-rate contracts (75% of mortgages). As existing terms expire millions of households face an immediate step-up in monthly repayments upon refinancing, directly curtailing disposable incomes. Highly leveraged private landlords experience severe margin compression. Stricter lender interest-coverage ratio (ICR) tests at higher yield levels force many landlords to sell properties, adding sell-side supply to the market while pushing displaced buyers into the rental sector .Because inventory is not locked into 30-year fixed rates, UK home prices adjust downward more rapidly than in the US, particularly in higher-valued regions like London and the South East.

Whilst in the US most homeowners hold 30-year fixed mortgages (85%) many locked in below 4%, rising yields create a financial barrier to selling. Moving to a new home requires exchanging a low rate for a 6.5% to 7%+ rate causing sellers to hold onto existing properties. Buyer purchasing power drops significantly pulling transaction volumes down. However, because inventory remains severely constrained by the lock-in effect home prices remain sticky high despite weak affordability. Homebuilders capture a higher share of transactions by offering rate buy downs (paying up front to reduce the buyer's mortgage rate that individual sellers cannot match.

So rising yields tend to freeze the US housing market as folk can bide their time for better rates whilst increases selling pressure in the UK due to structurally forced to remortgage which makes US house prices less volatile then then UK house prices given the higher forced frequency due to fixed rates expiring into high standard variable rates.

UK mortgages are more sensitive to central bank interest rate changes then US mortgages due to the long lock up period which is largely down to Fannie Mae and Freddie Mac, whilst in the UK it's all down to private banks to absorb or hedge duration risk. So british home buyers are getting screwed by the system because their is no government sponsored backstop to guaranteed UK mortgage backed securities, instead cyclic refinancing that tends to include huge fees.

US House Prices 12 Month Forecast

Current State: Locked in a "rate lock-in" trap. Existing homeowners with sub 4% mortgages refuse to sell, keeping inventory historically low and preventing major price declines despite weak sales volume.

Key Positive Drivers: Chronic structural deficit of single-family homes, massive demographic wave of Millennial first-time buyers, and anticipated Fed rate cuts despite yields suggesting otherwise.

Key Negative Drivers: Severe affordability headwinds from 6.5% mortgage rates, historic high price-to-income ratios, and rapidly accelerating home insurance premiums.

Prospects: House prices expected to stagnate, likely will be 2% higher in 12 months as the Fed and Government will do their utmost to prevent falling house prices which would ensure a recession however the feed back loop of a stagnating housing market will result in a stagnating economy.

UK House Prices Forecast

UK borrowers traditionally take on shorter-term fixes then US borrowers so the UK housing market is more rate sensitive then the US housing market

Current State: Bouncing along a cyclical floor. Affordable northern regions (North West, Yorkshire) and Northern Ireland are outperforming, while London and the South East face price softness.

Key Positive Drivers: Positive real wage expansion, and strong regional rental yields attracting buy-to-let investors.

Key Negative Drivers: Buy-to-let landlords exiting the market due to regulatory/tax shifts (adding sell-side inventory in the South) and elevated fixed-rate mortgage renewal rates. Bank of England looks set to raise interest rates.

Prospects: Stagnating UK house prices look set to fall over the next 12 months by about 5%.

China House Prices CRASH

Current State of the Market - Broad Secondary Market Slump: Resale home prices across major cities remain down between 5% and 8% year-over-year, as price discovery in the secondary market reflects actual market clearing faster than primary (new-build) prices.

Key Positive Drivers (+)

Substantial State Policy Support: Lower down payment requirements, record-low mortgage benchmark rates (~3.0%), and state-backed programs to purchase unsold developer inventory for conversion into affordable housing.

Key Negative Drivers (-)

Severe Inventory Overhang: Massive unsold housing stock in Tier-2 and Tier-3/4 cities requires several years of organic demand to fully absorb.

12-Month Outlook & Price Forecast

China housing market is a total disaster, in a downwards death spiral due to structural issues of speculative over building for which China does not have the population to support i.e. demographics of fast ageing are working against the Chinese housing market so there is no quick fix. The trend is for a 5% to 8% per annum drop which could persist for many more years to eventually see a 50% drop off the high, so we may only be at the half way mark though top-tier cities will stabilize well ahead of regional markets so the price falls won't be uniform.

Bottom line China is the New Japan, and not in a good way, China went on a building boom just as their population was peaking, and since which the workforce has been in decline by some 20 million workers per year where the canary in the coal mine was the Evergrande failure in 2021 as I wrote in articles at the time if it being the nail in China's Housing bull market which has stood the test of time. The response was the Temu export economy that the US trade war is disrupting which puts further downwards pressure on the chinese economy and neither will increasing domestic consumption work as a fast ageing population is not going to go on a spending binge hence there is worse to come as China seeks scapegoats and distractions away from economic disaster, hence increases the probability for a War on Taiwan. If the US can do it with Iran then China can do it with Taiwan, a similar smoke and mirrors exercise, which will definitely deliver a deep discount buying opps in TSMC if when the dust settles the factories remain standing!

Canada Housing Bear Market

Canadian interest rates have been on pause since last cut in October 2025at 2.25%. Canada could be forced to follow a US rate hike else face further severe economic consequences as investors sell Canadian assets to buy US assets forcing Canadian yields higher. The Canadian economy is facing the full force of Trump the trade war weakening the Canadian economy/.

Current State: Canadian house prices have been in a severe bear market after having topped in 2022, suggests structural weakness in the Canadian economy / housing market i.e. a glut of properties coming to the market that the trade war acts to worsen.

Key Positive Drivers: High net immigration driving underlying housing demand, aggressive Bank of Canada rate cuts improving buyer qualification power, and pent-up demand.

Key Negative Drivers: Ongoing "mortgage renewal cliff" forcing fixed-rate borrowers into significantly higher monthly payments, severe price-to-income detachment, and economic cooling.

Prospects: Expect Canadian house prices to continue drifting lower as the worst of the bear market is in the rear view mirror, house prices could drop another 3% over the next 12 months.

Australia House Prices Mania Bubble Top

Current rate 4.35%, last rate hike was 0.25% in May, next could come as early as this month especially if the Fed raises rates.

Current State: Broadly resilient near record highs, but starkly divided. Mid-sized markets (Perth, Brisbane, Adelaide) show double-digit structural momentum, whereas Sydney and Melbourne are leveling off due to hitting extreme price ceilings.

Key Positive Drivers: Severe national undersupply of new home completions, high net overseas migration, and extremely tight rental vacancy rates pushing tenants into buying.

Key Negative Drivers: Restrictive Reserve Bank of Australia cash rate, strict debt-to-income stress testing by lenders, and reduced household savings buffers as rates rise.

Prospects: Australia is in a housing mania with prices up about 60% in 6 years, Canada shows what happens when the bubble pops, which is inevitable, where the next rate hike as early as this month could be the pin that pops the Australian housing bubble, sending prices lower, triggering a flood of properties coming to the market both in attempts to capitalise on the mania and as forced sellers unable to service increasing mortgage costs. As developers go bust then market confidence erodes, transactions tend to drop. Australian house prices could easily replicate Canada's 20% drop.

That's five housing markets all with different house price trends! I would bet folk weren't expecting this.

The Holy Grail of Investing

My base case for 2026 was for a bear market to accumulate at deep deviations from the highs that have so far have only manifested to limited extent, i.e. the 10x brigade which is now up over 30% since Mid July, so there have been a number of opportunities to accumulate at deep deviations from the highs and distribute into new highs, usually repeating buying and selling in the same stock, the net effect of which for me and many patrons who followed is that despite being 70% in cash, we are beating the S&P without the risk of drawdown.

Instead IF the S&P had fallen to stand 20% down on the year AND dragged many target stocks lower with it then instead of being up 15% on the year, then buying the falling knives would place ones portfolio somewhere around 5% to 10% down on the year depending on how much cash one deployed into the falling knives and how well one managed fx positions as a 20% drop in the S&P would likely be accompanied with a dollar pumping by maybe 10%.

So the net position right now is for both the portfolio and S&P up about 15%, whilst at 70% cash.

So what if we get the 20% drop in the S&P what would that mean for ones portfolio?

The S&P dropping by 20% from here would put it down about 5% on the year, buying the falling knives would once again target a drop of between 5% and 10% off the portfolios peak value, so by the time the S&P has fallen 20% i.e. down 5% on the year, the portfolio would still be up about 5% to 10% DURING a BEAR MARKET, as say 30% of that 70% cash mountain is deployed into buying the falling knives.

This outcome would be unprecedented and unplanned for a purely a function of a few simple mechanisms that I iterate from time to time.

1. The S&P is a NOTHING BURGER to be IGNORED when it comes to what one does with individual stocks. Despite continuously iterating this for years folk still ask if I expect the S&P to fall then should one sell x,y,z stock to buy it back cheaper, my response tends to be see each stock on it's own. Some stocks crash by 50% whilst others 2x or even 3x all whilst the S&P does it's own thing.

2. Viewing each stock on it's own means one ACCUMULATES the deep deviations from the highs AS PRICES DROP as STOCKS BECOME CHEAP, the classic example of this during 2026 was Microsoft! Where during the year I 4xd my position, which I have been trimming as it continues to break higher and become expensive. And the same holds true with EVERY STOCK! EACH STOCK ON IT'S OWN IN ISTOLATION WIHOUT REFERANCE TO OTHERS AND DEFINETLY WITHOUT REFERANCE TO THE INDICES. Folk can see this in action in the Trade Winds article where you can see that I BUY and SELL EVERY DAY. I don't even look a the S&P, it is that irrelevant.

3. High Percent Cash allowed one to BUY the dumps BIG as they came along because one is primed to capitalise upon such opportunities in a mechanistic manner. whereas too little cash and one misses most of the opps as one THINKS too much about exposure and risk i.e. I plowed about 5% into the software sector, and 4% into crypto's, I doubt I would have done that had I been say 20% in cash, instead likely seek to keep my powder dry for the AI stock opportunities. Having higher percent cash means one can beat the S&P without little if any portfolio drawdown.

These three mechanisms have got me to a position where IF the S&P drops by 20% going into year end then I will end the year UP during a BEAR MARKET! That IS INCREDBLE, which despite noise in the comments from time to time from those who have yet to comprehend what I am demonstrating in real time, this is TELLING ME that what I am doing IS THE RIGHT WAY TO INVEST! Investing like a machine that apparently cannot lose no matter what the stock market does! The Holy Grail of Investing.

It really does not matter what the S&P does, not to my portfolio or to those who follow, only those who remain hitched to the S&P fail to grasp what I am saying which is that I CANNOT see how I can lose in the stock market because there are always stocks that are pumping, becoming expensive to SELL and there are always stocks dumping becoming cheap to BUY.

One needs mechanism so that one buys when cheap and then sells when expensive REGARDLESS OF EVERTHING ELSE! Primary is reacting to price movements in real time, i.e. today Circle pumped to over $100 and a sell limit order got triggered so I took a look and added more sell limits and added a higher buy to rebuy when it next drops should it do so, that's it, that's all I DO each day. it's all there in the comments and the in the Trade Winds.

Friday 4th Sept 2026Big Buys - FICO 10% dump buying at $910Big Sells - Small Buys - Small Sells SMCI, CRCL, NVDA

As the buys and sells get triggered I go take a look at the charts and add / adjust / scrap limit orders, if a limit orders not triggered then I don't need to look at that stock.

What could go wrong that makes me lose, where a loss would be if I am down MORE than the S&P during a year i.e. my expectations at the start of the year were that if the S&P drops 20% then I could be down upto 10%, depending on how well I managed fx as my 15% gain is in STERLING not USD which makes where the portfolio stands even better than it already looks.

So when people say you got 2026 WRONG, yes I can see the S&P has not followed the trend forecast I posted last December, but I am not feeling it, this feels a lot better then if the market had done what I forecast it could do. Folk fixating on the S&P are completely missing the golden goose that keeps laying the golden eggs.

AI Secondary Stocks

The prospects vs risks boil down to which stocks benefit from capex spend double edged sword as when capex is cut so will their margins, it all depends on where we are in the capex cycle and for how long it extends i.e Micron at the same time has the highest prospects to be higher a year from now on the back of explosive earnings growth but also is the most vulnerable to a cyclical downturn which is hard to call, I mean back in 2023 the likes of Musk and Druckenmiller were expecting a recession in 2024, and well I think they've both give up trying to make such calls since, rather than an out right collapse in demand what we see is multiples compression that gives us the 50%+ drops in stock prices as we are witnessing with the likes of KLAC that has fallen from a peak of $310 to a recent low of $170 for a near 50% drop despite the metrics remaining firm. That coupled with signs of hyperscalers scaling back on capex spends delivers the likes of 60% to 70% drops for the likes of KLAC,so as is usually the case one needs to be prepared for the getting lucky events when the come along and not get sidetracked by MSM hysteria which tends to be euphoric going into the highs and End of the world pessimistic going into the lows, alls we can do is know when a stock is expensive and when it is cheap and thus accordingly accumulate the DEVIATIONs from the highs such as picking up some KLAC at $175 on Friday, not because its THE LOW because it's likely not, rather because it's 50% below the high.

The signs for a slowdown in capex spends are there, we see it in rising yields, borrowing costs and credit default swaps that have spiked to financial crisis era levels for the likes of Oracle, and hence the Oracle stock price has been under pressure. Then we have infrastructure issues such as the power gird unable to meet demand for all of the under construction and planed data centres which means a scaling back in purchase of semi-conductors are in the pipeline when data centre builds start to become moth balled. There are many Achilles heels baked into the build out such as the useful life of assets such as GPU's which is a lot less than the fibre optic networks of the dot come era which will be one of the reasons MSM will report on to explain stock price drops AFTER the event.

The bottom line is that assets such as GPU's have a short 3-4 year shelf life, which means deploying a large number of GPU's puts a permanent maintenance strain on free cash flow, so it's not a case of spend once and profit from for decades as was the case with fibre optics, instead it's a case of continuously keep spending or start mothballing data centres. On face value this implies demand for Nvidia, AMD, KLAC and so on but it also implies accelerating the switch to custom silicon and thus will hit general semiconductor demand whilst benefit the likes of Broadcom.

And don't forget the stock prices move BEFORE the news hits the mainstream media which is why folk tend to experience the OPPOSITE price action to what the news suggests should happen hence why I don't bother looking at the news, instead focus on the metrics, stock charts, macros and mega-trends, the news is usually irrelevant.

1. Micron $1016 - EGFS 150%, 157%, Dir 0%, 0%, PE Ranges 63%, 10%

Drivers - Micron the runaway freight train, Structural HBM4/HBM3e supply lock-in, DRAM multi-year LTAs, AI server demand. Key risk Cyclical memory spot price collapses during oversupply hitting margins.

I hold none, why? Eyeball time! What occupies most eyeball time is that which I am most exposed to and that which I most seek to accumulate, micron was always a periphery stock, not primary or core secondary i.e. a gamble secondary unlike for instance ASML and AVGO, Micron was more like a TSLA, just not as bad! But you get the picture Micron was always a side salad, still I did at one point have heavy exposure to which it rewarded me with an overall 3x profit, so as far as I am concerned it did it's job. Could have been a lot worse, could have done an Intel and trade 50% lower for years, though even Intel eventually came good, delivering a 3x on my peak exposure average buy, so things have gone great even if I did not ride Micron into SPACE!

Metrics - Remain Epic on a whole different level of Epic, hard to believe what they imply for the stock price and hence I didn't believe them, saw them as too good to be true. Alls one can do is LEARN for next time, the key lesson is PAY ATTENTION TO THE EGFS they told me at $80 Micron was epic! Go read the articles and comments, the EGFS were hard to believe, nothing is 100% so there is the chance that Micron had broken the formulae and was spitting out gobbledygook, if the EGF's work for 90% of stocks then that's good enough for me, so I could not quite believe what the EGF's were suggesting, incredible, unprecedented, and as I often commented and wrote EPIC which is how things turned out since, the lesson learned when an EGF is EPIC is to go with the flow!

Stock chart - When to buy? At $625, why $625? Because it's a 50% drop off it's high, from where the upside potential would first be a 2x to $1250 and then 3x, 4x, 5x..... I'll probably start lightly accumulating at around $800, there's support at the previous low of $740, increasingly size of buys down to target of $625. When / IF We get to $625 then one can ponder lower buy limits down to $450. That's how I'll reaccumulate a position in Micron.

Stock price potential - Epic earnings growth IF sustained could see Micron 2x again! $1250 converts into $2500! It's beyond comprehension! I remember folk used to argue with me about buying too high at $60 when they could have waited and bought the LOW at $48! Now it's trading at $1000 coming off a high of $1250 and the way it's going it could 100x i.e. from $50 to $5000! Micron is in a class of it's own and the EGF's flagged this all the way from $80!

2. LRCX $307 - EGFS 36%, 77%, Dir 18%, 29%, PE Ranges 251%, 118%

Drivers 3D NAND layer expansion & Gate-All-Around (GAA) etch intensity. Key risk Volatility in memory CAPEX spend as Chip makers rapidly freeze new tool purchases when end-market demand softens or capacity exceeds demand.

Metrics - Epic but expensive, a lot of growth is baked into the price i.e. even a year from now LRCX will be expensive at 118% unlike Micron which will be dirt cheap at just 10%

Stock Chart - Where's the bear market folk ask, LRCX dropped 43% down to $240 that's your bear market! Though still just outside my buying range of $202 to $150 so I am raising the buy range to $226 to $156 and I'll start adding lightly at $250 having made some on a small short down to $240.

Stock price potential - A year from now LRCX could be 30% to 40% higher so targets $406 to $450.

3. AMAT $455 - EGFS 38%, 42%, Dir 12%, 5%, PE Ranges 185%, 115%

Drivers - Broad WFE leadership, materials engineering, advanced packaging, Key risk China export restriction shifts and capacity demand issues.

Metrics - Again just like LRCX Epic, and similarly over priced.

Stock Chart - AMAT is like a carbon copy of LRCX,. also traded 42% lower to its recent low of $430. There's support at $380 so I'll pick up a small amount there. Buying range nudged higher to $324 to $220, increasing size of buys as it drops and similar to Micron and LRCX I currently hold none. Though did make a small amount on a short. So I am seeking $380 to start adding at about 50% off it's $740 high.

Stock price potential - A year from now targets a similar gain to LRCX of 30% to 40% to about $600 to $650 off of current $455.

4. KLAC $185 - EGFS 18%, 59%, Dir 11%, 16%, PE Ranges 254%, 129%

Drivers Yield inspection monopoly on 2nm, High-NA EUV, and HBM packaging. Key risk Fab construction delays.

Metrics - Epic but very over valued. Even though its dropped by near 50% it's still very expensive.

Stock Chart - KLAC has dropped 45% from peak to trough down to $168, which is still someway above the top of it's buying range of $132. Think about that! The buying range has done it's job of PREVENTING FOLK from FOMO-ing into its $307 top, so many times folk have asked if I am going to raise my buying rage. I remain reluctant to raise KLAC's range because it will STILL be expensive at $132. Anyway I have been lightly accumulating as it drops as I take profits on my shorts which has nudged my position to 0.10% of portfolio, a tiny position. I seek sub $140 for bigger buys, all the way down to $90 which would be a 70% drop off it's high.

Stock price potential - From current $185 I can see KLAC 25% to 30% higher a year from now, to about $235 to $245, so I don't see a return to a new all time high any time soon, the pump to $307 was pure FOMO mania, which yes it could happen again, which is why I did add some on the dip to $168, so that I at least have something to trim should it pump.

5. AMZN $259 - EGFS 7%, 61%, Dir 12%, 5%, PE Ranges 94%, -7%

Drivers - AWS AI workload acceleration, Trainium adoption, high-margin ads. Key risk Massive $220B+ CAPEX drag on FCF

Metrics - EGFS and direction of travel are positive and not too expensive but Amazon does not have the growth of the above 4 Secondaries so one needs to wait for an opportunity to accumulate when cheap else risk over paying though time does work in Amazon's favour i.e. the stock will get cheaper over time.

Stock Chart - Amazon stock price is low volatility, i.e .unlike the others so far the stock price does not tend to move too much, which means when it does move take the opps to accumulate and distribute, i.e. on the recent pump to $290 and before that dump to $225. The buying range is $201 to $152 which it did briefly trade into earlier in the year and which is where the bulk of my Amazon buy orders are. It's a steady eddy stock, accumulate in the ranges and trim some into new highs whilst one builds and maintains a position. I don't expect Amazon to experience a severe bear market i.e. it's going to be a case of getting lucky to see Amazon fall below $175.

Stock price potential - From $258 Amazon could trade about 20% higher a year from now so I do expect Amazon to target a new all time high above $300. I'll start adding lightly at $229 with main buys starting at $201.

6. Qualcom $169 - EGFS -23%, -9%, Dir -5%, -23%, PE Ranges 39%, 49%

Driver AI PC architecture momentum (ARM), automotive digital chassis. Key risk Apple modem in sourcing timeline, depend heavily on discretionary consumer spending

Metrics - Qualcom is the first in this analysis to have NEGATIVE EGF's i.e. Qualcom's earnings are CONTRACTING as evidenced by the direction of travel, which means despite Qualcom looking cheap on a PE Range of 39%, which rises to 49% in a years time due contracting earnings. So Qualcom is one stock folk don't want to waste opportunities to trim at a profit because it will likely give back ALL of it's gains and I could see that from the metrics alone all year without looking at any news report or stock charts as I have been iterating each time it pumped.

Stock Chart - Qualcom illustrates why it's daft to stare at the S&P because over past 3 years it's had 4 bull markets and 4 bear markets to capitalise upon by buying when it drops and sell when it pumps so whilst one can try and guess what comes next but it doesn't really matter as long as one ACTS! It also illustrates the other Achilles heel of investing which is folk wanting to invest for a fixed time period i.e. for the next 3 or 5 years, which stocks like Qualcom would deliver a nightmare ride. Where over 5 years Qualcom could 2x and drop by 50% as many as 10 times! Which is why I like Qualcom! The lower it drops the more I buy the higher it pumps the more I sell.

Qualcom is currently in no mans land at $168, too low to sell and too high to buy, this coming off a high of $260 and off a low of $140. I recall an exchange with a dude at around $240 convinced Qualcom would FOMO to over $300 that they would regret not selling at least a fair chunk of what they held. Yes we can fantasise about the mania taking Qualcom to $300+ but as I said at the time the risk outweighs the potential reward, i.e. the risk is for a drop to $120 vs the reward of another $60 upside to $300, it's just not worth the risk especially given Qualcom's poor metrics. the rally to $260 was getting lucky.

Potential upside a year form now. I don't think it's going to break to a new all time high, not with its metrics, I think its going to find it tough going over $200 though could make it to $230 before any rally fizzles out. So upside potential from $169 is about 30% to 35%. So I seek to re-accumulate Qualcom once more in it's buying range of $152 to $120 whilst seek to start trimming at about $220 which yields about a 60% profit on buys.

7. TSLA $354 - EGFS -8%, 20%, Dir -4%, -11%, PE Ranges 421%, 338%

Drivers - Megapack energy storage growth, FSD software monetization, robotics. Key risk EV price wars & auto margin pressure, depend heavily on discretionary consumer spending

Metrics - Negative EGF's coupled with extremely high PE ranges has meant that once the mania ends TSLA is going to face a valuations reckoning, i.e. a severe bear market.

Stock Chart - The stock price topped in December 2025 since has been in a volatile bear market all the way down to a recent low of $296, down 46% off it's high, let that sink in, TESLA has been off 46% off it's high all whilst many remain convinced that Musk is going to deliver them to the promised land of $1000 and beyond. TSLA is mostly HYPE, there is little actual substance, still one can make good money from TSLA as long as one does not buy into the hype and thus SELL when it pumps to then buy when it dumps by 50% or more which is where it is close to achieving. My base case is that TSLA will see sub $200 where I will execute my big buys just as I did during 2024 when TSLA fell to a low of $136. The buying range is a wide $286 to $172, with the buys getting heavier the as the stock price drops, particularly below $220. We will only know THE LOW in hindsight, and thus I will keep buying all the way into the low be it $200, $180 or $140 as I rebuild that which I sold on the pump to $500.

Potential upside is about $450 so about 30% from here, what does that mean? It means after TSLA has had its bear market and bottomed where ever that may be then TSLA will target $450 and thus I will likely start trimming TSLA once more when it next breaks above $400 which should deliver me a 2x on buys just as I have done several times before. Folk fixate on certainty, when instead folk should seek to position themselves for both a drop and a pump by accumulating as prices drop and then distributing at a profit once they pump. for TSLA accumulate sub $286 down to as low as it goes, and distribution above $400 to as high as it goes where the chances of TSLA blasting off into space to the likes of $1000 are pretty slim, and not necessary if one trades TSLA's $500 to $150 range by capturing $200 between say an average buy of $220 and sell of $420 as I've done twice before which is why TSLA has proved one of my most profitable stocks, despite the stock price having gone nowhere for 5 years!

8. IBM $235 - EGFS -3%, 6%, Dir 14%, -13%, PE Ranges 69%, 59%

Drivers - Red Hat ARR growth, enterprise AI consulting, high software mix. Key risk Discretionary IT spending cuts

Metrics - IBM has consistently weak EGF's that never supported the mania pump o $330, which means one needs to trim when there be FOMO and wait for the opportunities to buy when cheap, $235 is not cheap. EGFS are weak but not catastrophic which means to expect an orderly bear market followed by a slow recovery.

Stock Chart - IBM fomo'd on Quantum hype all the way to a December high of $330, a rally that I sold all of what I held into whilst many had convinced themselves that IBM was going to do a Micron and blast off to $400 and beyond.... Instead since the stock price has dropped by 40% to a low of $200 into it's buying range of $208 to $168, allowing one to rebuy a large chunk of what one sold. So the stock price targets $168 for a 50% drop off the high. Nothing complicated, IBM delivered it's bull market mania to SELL into and now is coming good on it's 50% drop bear market, what more could one ask from a stock?

Potential upside - I think it's going to be tough for IBM to ignite the fomo juices once more so I would be surprised if we see a new all time high anytime soon, off the current price of $235, a year from now we could see IBM about 20% higher to around $280, which would be +65% if one manages to buy chunks as low as $170.

LULU $103 - We Will All Buy Too High

Several patrons requested to take a look at LULU, a stock I've mentioned in the comments and maybe an article. LULU looked like a great buy following a 60% drop to between $200 and $160, now it's trading at $103. This is what investing be like! Buying when cheap does not mean it can't get cheaper! Buying the dip has got me to 133% invested of target exposure, this is why one tracks percent invested of target else ones position size can mushroom and put ones portfolio at risk, which is why tracking percent hard cash invested is so important.

LULU so far has not given many opps to trim at a profit, so the average buy remains quite high at $168, which is still a lot better then folk who bought it at say $320, or $420 or maybe they fomo 'd into it's $520 high. So I did what I do with every stock, I bought when LULU dropped all the way to down 81% from it's high. so I've just added a little more. There's no point looking at the news for reasons why its dropped because stocks fall on BAD NEWS, alls I want to know is could it go bust, does not look like it could, and how far is it off it's high -81%.

In terms of trend, I can see that $170 looks doable as a target to aim for, so what I bought today I can sell at $170 which will help nudge the average cost of share lower. LULU illustrates that one needs PATIENCE, to give the position time to come good, which might not seem possible with LULU trading down 81% at it's lowest price in eons but I've been here before, so many times, so being in deep drawdown on LULU does not phase me, I can see $170, then $220 on break of which folk will be looking at a whole different stock and I'll be getting comments if I think LULU will drop to $100 again so folk can buy, I'll likely reply, sorry buddy that ships sailed, fortune favours the brave.

The name of the game is to accumulate and distribute at a profit, for that we need volatility such as we experienced with the pump from $160 to $220 during which I trimmed some, the next expected volatility pump will be to $170 which is where I aim to trim some of what I hold at a profit i.e. buy at $102 trim at $164 at a 60% profit.

Bitcoin's Most Profitable Bear Market

My consistent view is that we are in a late cycle bear market rally that targets a new bear market low BELOW $58k with my base case target for 18 months being $48k. This view only comes into question if we see Bitcoin trade above $84k, that would be the key signal for me that a higher high implies to expect a higher low, to date bitcoin continues to support my base case and thus I have been trimming the rally in bitcoin, Solana, eth as well as crypto stocks Circle, Bmnr and Coinbase, roughly about 15%, to up my buying power on the next dip to a new bear market lows, start accumulating Bitcoin sub $68k, Solana sub $80, Eth sub $1950, Circle sub $76....

So as things stand with bitcoin trading at $79k nothing has changed and thus remains one of the most straightforward and predictable bear markets one could have hoped to have experienced, delivering a number of opportunities to trim at a profit as the likes of Circle illustrates i.e. Circle recently bottomed at $58 and since pumped to $106 to trim into, that is a 80% PUMP! even if one trimmed for half the range it's still a decent 40% profit.! Hence my exposure has gone from about 140% invested to down to 112%.

One is not supposed to make profits during a BEAR MARKET, which are times to accumulate the deviations from the highs, the fact that one has been able to profit during this bear market makes it seem so easy to those who have followed my lead, bought the dumps and sold the pumps in circle several times already, it seems easy but it's not!

Too easy means there has got to be some pain to come, things don't go this well without a MAJOR surprise which one has to be prepared for to some degree, i.e. such as when AMD dropped from $120 to $80, that was PAINFUL.

So with bitcoin the surprise could be a drop to say $30k, it's not my base case but one needs to be prepared for to some extent, there is no reason why it should but for there to be PAIN then it needs to do something painful, which is why I stated that Solana could drop to $40, not because I am waiting to BUY the drop to $40 but that would deliver PAIN in response to which alls one can do is trim what one bought on the dump so the higher Solana goes the more one trims, which then carries the risk of Solana not falling back down to the $70's. There has to be risk and there has to be pain where for Solana there are two pain trades, firstly that Solana drops to $40 and secondly that it doesn#t drop but keeps climbing higher past $150, and onwards to $200+ Those are two events that deliver at least some PAIN, the first is deep drawdown PAIN, the second is the pain of having sold too much too early and thus the pain of less profit. We are likely to experience at least one of the two, for me I would much rather prefer the drop to $40 then if Solana kept climbing higher to $200+ But this is the reality of the market, it does tend to deliver PAIN! PAIN that most investors can't take whilst I understand it's part and parcel of the game we play, like folk who go paint ball shooting, who enjoy the thrill and excitement when they shoot others, but less so when they get shot themselves.

The SOLANA Accumulation and Distribution Game.

During the recent 6 weeks that Solana spent in the $70's I iterated countless times that folk should not pussy foot around waiting for lower prices, if we saw the $60's that would be great, and if we see the $40's that would be getting lucky!

The thing about the markets is you do not always get what you want which is why I DO NOT WAIT FOR THE TOPS OR BOTTOMS! i.e. I sold 90% of my bitcoin going into the $126k top leaving 10% for the likes of $134k to $148k that never materialised as I iterated to folk at the time, same with Eth, yes it could have pumped to $5800, but I wasn't going to wait for $5800, I sold 90% going into $4800 and the same has held for the bear market as I have iterated countless times, so folk who failed to act are playing pick and mix so as to take the easy route by not THINKING! Because it takes ENERGY and WORK to THINK! It's easier to just fixate on snippet such as $40 rather than the whole picture of accumulating a position to capitalise upon for the next BULL MARKET!

What's my base case?

Solana will see $225+ during the next bull market, so one needs to be measured in the amount one trims at say $105, yes the plan is to rebuy in the $70's but the market does not always give you what one want and it's a matter of degree where the risk is losing $120 profit vs buying back $30 cheaper. The solution is to stagger ones sells so that one eventually does get a $30+ drop, just that it could be off of $130, $150, or even $200, which one will only know in hindsight. This is reality vs the fantasy that most talk themselves into and out of what they should be doing. To be blunt if you failed to buy significant Solana in the $70's then you messed up! You need to learn from it so you can better appreciate what to do next time, REALITY VS FANTASY! Yes I fantasise about Solana dropping to $45, but the reality is it probably won't in which case what do I do? Each day I look at where the price is and then ACT in the present, for 6weeks Solana was trading in the $70's so each day I bought a little more Solana. It's the same as Solana goes up, each time solana pumps higher I sell a little Solana, the fantasy is that it could drop back into the $70's to rebuy what I sold but the reality is going to likely be it trades to $150 first, so I pace my sells to capitalise upon that which if it happens I won't need $70s for rebuy's, off of $150 $110 would be just as good. That's how one plays the accumulation and distribution game.

THE OPERATING SYSTEM

Patrons often ask who are The THEY?

THEY are the complexes where the one folk are most aware of is the Military Industrial Complex which is just one cog in the wheel. There's The Financial-Banking Complex, The Surveillance-Tech Complex, The Medical-Industrial Complex (Big Pharma), The Media-Entertainment Complex, The Agro-Chemical Complex (Big Ag), The Prison-Industrial Complex then we have the Bureaucratic-Administrative Complex (Government) not forgetting the Education Complex tasked with dumbing down the population, and the oldest of the lot that acts as the foundation stone is the Theological-Ecclesiastical Complex, the psychological arm of the Deep State.

The THEY is the synthesis of all of these, collectively go by many names, The New World Order, Illuminati, Deep State, Great Reset.... I would best label it as EMPIRE. Whilst the intelligence agencies act as the nervous system the connects them all together, as the brain and the enforcers, keeping the other complexes inline.

we are mere cattle on a farm, the cattle tend to get easily worked up and waste time fighting against other cattle whilst the farmers smile at how dumb the cattle are, so easily side tracked and kept under control as they are herded to slaughter, look at the farm called Russia, how many of their young cattle have been slaughtered in Ukraine for the Farmer?

The Fall of the Roman US Empire

Be forewarned this section is going to give some folk whiplash! 😁

Rome like Athens is in decay, clinging on to ancient glories past as major sources of tourist revenue, one could say ancient Rome is Italy's crude oil, a well that keeps giving $$$'s whether the locals like it or not, without tourism they would be infinitely poorer!

Rome is okay to visit, crap to actually live there, too bloody hot for one thing, and yeah tourism does jack up the prices somewhat for day to day living costs and property prices making it that bit harder for locals to get by but then again they would be a lot poorer without tourism. So all these anti tourism morons in the likes of Spain are literally seeking to kill the goose that lays the golden eggs.

The whole origin story of Rome and what was supposed to have happened in the early centuries obvious to most today as being total fiction, Romulus and Remus and many if not most of the wars Rome was supposed to have fought never happened, which is the case with every Empire even to this very day, the victor writes the history to justify legitimacy of rule. Take World War 2, the victors wrote themselves into being the good guys and the Japanese and Germans as the bad guys, evil in fact! When the reality is that the Germans and Japanese were no different then the Brit's, Americans and Russians. But history is written by the victors and this now everyone believes that the Germans and Japanese were evil, even the Japanese and Germans themselves! That demonstrates the power of victory! Get to dictate history that paints oneself as the good guys.

How Did Pagan Rome Become Christian?

Because Saul of Taurus hijacked a Jewish cult who's leader was killed decades earlier then placed himself at the centre of that which he sought to create for his own purposes, you see evidence of this everywhere, St Paul's Cathedral, St Paul's Square, it's all bloody PAUL,PAUL, PAUL! Made up BS about Jesus coming to him decades after his death and telling Paul you are the main man! Not Jesus's brother James, nor any of the disciples who actually spent time with Jesus but YOU PAUL YOU ARE THE ONE! Christianity would be better named as Paulianity given that what it was in the time of Jesus is not what was written in the bible some 50 years after the death of Jesus which would become even more detached from reality with the Holy Trinity.

But folk swallowed it hook line and sinker as it offered them exactly what most folk wanted, someone else to do the thinking for them that and it suited the interests of the Roman Empire as it promoted Roman supremacy above all else. Christianity was all about assimilating the rebellious Jews into the Roman Empire as christians, to mix the pot which then went full blown midlevel with Constantine when Rome became a Christian Empire, those who remained non christian were no longer equal Roman citizens who with each passing Pope became further persecuted unless they converted which remained Christianities base case until the Protestant wars sought to break the stranglehold from Orthodoxy and set Europe Free to come to pray to a new god, MONEY! And thus Capitalism was born!

This also answers the question where Islam came from, Today's fanatical Jews, the zionists and kindred spirits deride Islam as a monster of sorts despite the fact that a sect of Judaism likely CREATED ISLAM as a weapon to use against the Christian Rome Empire, to break the Empire that constantly sought to persecute the Jews unless they converted to Christianity. So the Christian Empire had to go!

The Jews sought a Messiah and Arabs sought a Messianic religion, both united towards the objective of reclaiming the holy land from Byzantine control. What went wrong? CONTROL! One sought to control the other..... There would have been real world catalysts for the rejection of this union such as Jewish racism against the Arabs, seeing themselves as superior, the Israelites so to speak contributed greatly towards Islam becoming distinct and independant of Judaism resulting in subsequent conflicts between the once unified tribes, one can see this with the shift form prayers towards Jerusalem redirected to prayers towards Mecca, other shifts happen in the changing of the prayer day to Friday from Saturday, So even in it's early years Islam sought to separate itself from both Christianity and Judaism.

The key date for the shift away from Judaism would have been the shift in the Qibla from Jerusalem to Mecca around 624 AD one could say that is when Islam was born as an Independant religion, until that point it could have gone either way....

Of course we are talking of ancient times, the gap between then and now is literally 1500 years. And so Islam was birthed into existence, things did not turn out exactly as planned, yes Islam eventually would go on to deliver the Apocalypse that saw the heart of Christianity, Constantinople fall, with folk across the western Christian Empire fearing they would soon be next and thus prepared for the End of the World, if Venice fell then next would be Rome and that would be christianity gone kaput. But that's all history because Christianity survived as did the Ottoman Empire until it would experience it's own apocalypse and go kaput where for a time it was touch and go if there would be a Turkey given that the Greeks were out for blood, seeking to wipe the Ottoman Empire off the face of the Earth, but in stepped the Russians to prevent the British Empire from further threatening the Russian Empire and thus despite ideologically opposed to an Islamic Turkey, it was in Russia's strategic interests to have Turkey as a buffer state.

So there you have it, A rich Roman called Saul / Paul created christianity so that the rebellious Jews could be romanised, and with the birth of the Christian Roman Catholic Empire the increasingly persecuted Jews would go on to create Islam as a weapon to destroy the Christian Roman Empire so as to return the Empire to it's former multi-religous state where anything goes and all citizens are equal, of course there were millions of slaves who had no rights.

And there would be many new Paul's along the way such as Martin Luther.

Why stop with Christianity and Islam, who created Judaism?

King David! To give legitimacy of his conquest of Palestine and for his successors. It was the house of David that wrote and assembled the myths to create a Jewish people about 1000 BCE, the promised land, Abraham, Jacob, Moses and so on never existed, may as well have been called Hercules and Achilles...

All religions are engineered towards giving legitimacy to rule over others, manufactured histories, building on myths and superstitions taken from across the ancient world, stories from Egypt and Mesopotamia, back dating history, bolting on an extra 500 years from the present so that the peoples come to believe that it has always been such, so that the slaves don't rebel, for instance the bible was used to indoctrinate America's black slaves to obey their masters or they will burn in hell!

The Old testament is a work of state propaganda. The only way such text and be imprinted onto the uneducated masses if it is by royal decree.

Everything in the bible before King David is a myth, Adam and Eve, the Garden of Eden, and God said let there be light! The word had meaning because only the priests and elite could read the words and thus Judaism created to bind disparate conquered peoples together into a nation state, what do you think schools today are busy doing?

The purpose of religion is to dehumanise people into believing that they are powerless in the face of god and his representatives (the King, Emperor and high priests) on Earth and thus submit to become slaves to the religion, to die for their King who is anointed by God through which you as their obedient servant are carrying favour with God to be rewarded after you die.

Religions seek uniformity amongst people, they squash innovation, free thought that is seen as heresy which is why Empires collapse, they become stale, rigid, bureaucratic, the system is bogged down by red tape, which was China's Achilles heel, the Chinese Empire was too bureaucratic and still is under the CCP, too rigid, to uniform, the underlings are too afraid to have opinions let alone voice them, Russia is similar, it's the Achilles heel of all Empires.

Whilst the rise of the Ottoman Empire would spark European voyages of discovery to find new routes to the east by going south and west and so the new world would be stumbled upon and spark new empires to come into existence, the Portuguese, Spanish, French, Dutch and British Empires all by accident, alls they were doing was looking for a sea route to India and China that avoided the Ottoman Empire.

The Fall of the US Empire

The US is a copy and paste of the Roman Empire, the senate, fake republic when in reality is an Empire, the Romans themselves did not see themselves as an Empire despite the fact that is what they were, and what everyone today refers to Rome as, the same is true for the US Empire, Americans do not see themselves as Empire when the US clearly is a global Empire which is what history will call the US when looking back in centuries time, if we get that far!

US violence abroad will increasingly come home to roost, and there are plenty of guns across the US, it's going to become very bloody in the US, lots of political assassinations, accelerating drift into a police state with death squads roaming the cities we are seeing it with ICE, the US is heading for huge amount of domestic violence because the US like Rome is a very violent empire, it's imbedded and ordinary folk have plenty of weapons to carry out violence against one another. Call the cops and you risk getting shot by those who you called to protect you! Parents in the US fear the next mass shooting will be in their kids school, you don't get that level of fear and violence in the UK, despite what the far right in the US propagandise about the UK, it's infinitely a far less violent place to live which is why I don't see any reason to move elsewhere, the grass is not greener abroad.

Trump illustrates how strong propaganda is in the US, that they can't see he is destroying America step by step whether by design or stupidity.

Which will rise to displace the US?

China?

Japan?

I think Europe has a good chance of filling the void that the US is leaving behind as it diminishes through acts of sheer stupidity.

Europe has the capacity to rise given that it has the prerequisites i.e. freedom, law and order, infrastructure, institutions, it can displace the dying US Empire. The US had a good 80 year run as the top dog, now it's in decline, Europe can rise, this also suggests that at some point the UK will rejoin Europe, that would be the end of the US Empire as we have known it to be. The worlds focus will shift to Europe, trade the euro and european stocks, the trend is already in motion i.e. gold is being repatriated from the US back to Europe. Britain an Island is the natural wealth safe haven for Europe which is why the Dutch withdrew gold from the US to the UK.

So folk should think of the rise of one empire and the fall of another as opportunities i.e. every bear market gives birth to the next bull market.

Imagine what happens when AI gets religion, the fanaticism of self sacrifice for the greater AI good will be epic! That will likely include human collaborators who will be convinced they will be rewarded by being uploaded into a virtual heaven.

Walking around Rome one realises that most people don't want to know the truth, prefer to be kept in their loops, living lives in open door prison cells, cattle on a farm, prefer the comfort of lies, prefer the comfort of their illusion. Ignorance is bliss, that's what Rome reminded me.

The TRUTH is Good, Bad and Ugly but the TRUTH WILL Set You FREE!

What's the truth? THAT is what you will have to discover for yourself! I'll stop here else folk might start getting silly ideas about nailing me to a cross 😁

When in Rome!"""

# Stocks called out in the article, with the levels it publishes (equities the
# app already monitors; crypto mentions live in the Crypto tab).
ARTICLE_STOCKS = [
    dict(t="MU",   levels="Light adds ~$800; support $740; target zone $625; lower limits $450",
         why="The runaway freight train — EGFS 150%. Holds none; buys the 50%-off-high dip."),
    dict(t="LRCX", levels="Buy range RAISED to $226–$156; light adds at $250",
         why="Epic but expensive; 12m $406–$450."),
    dict(t="AMAT", levels="Range nudged higher to $324–$220; small buys at $380",
         why="Carbon copy of LRCX; 12m $600–$650."),
    dict(t="KLAC", levels="Range UNCHANGED $132–$90; added some at $168",
         why="Yield-inspection monopoly; still expensive at $132; 12m $235–$245."),
    dict(t="AMZN", levels="Light adds at $229; main buys from $201 (range $201–$152)",
         why="Steady eddy; 12m >$300 new ATH."),
    dict(t="QCOM", levels="Re-accumulate $152–$120; trim ~$220",
         why="Negative EGFs — earnings contracting; trim the pumps; 12m ~$230 max."),
    dict(t="TSLA", levels="Trim once it breaks above $400",
         why="-46% bear market to $296; 12m target $450 after it bottoms."),
    dict(t="IBM",  levels="Buy range $208–$168; 12m ~$280",
         why="Sold the $330 fomo pump, rebought in range; $235 is not cheap."),
    dict(t="NVDA", levels="Buy $148–$183 (sheet); trim ATH $237",
         why="Smart money sold into earnings; buying opps primed into October."),
    dict(t="LULU", levels="Added at ~$102 (down 81%); next trim target $170",
         why="'We will all buy too high' — avg cost $168 vs the $520 high."),
    dict(t="FICO", levels="Bought the 10% dump at $910 (4 Sept)",
         why="10x brigade; brigade +30% since mid-July."),
    dict(t="CRCL", levels="Trimmed the $58→$106 pump; sell limits stacked above $100",
         why="The accumulate/distribute mechanism in action."),
]
ARTICLE_CRYPTO = ("BTC: bear rally — base-case low below $58k, 18-month target $48k, view "
                  "invalid above $84k; trimming rallies. SOL: rebuy in the $70s ($40s = getting "
                  "lucky), staggered trims from $105, next bull $225+. Live prices: Crypto tab.")

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
        note='[B] Homing in on $48k target, accumulate sub $60k. [Crypto sheet] primary $134k. '
             '[A 8 Sep] base-case bear low <$58k (18-mo $48k); view invalid >$84k; trimming rallies.',
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
        note='[Crypto sheet] exit 190/400/490. [A 8 Sep] rebuy in the $70s ($40s = getting '
             'lucky); staggered trims from $105; next bull $225+.',
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
ALL_TICKERS: list[str] = list(dict.fromkeys(s["t"] for s in MONITORED))  # de-dup, order
# [v4.1 FIX] Fingerprint of the monitored book. Every quotes/funds/overview cache
# key embeds it, so adding or removing a ticker (LULU/NFLX in v4) automatically
# invalidates cache files written by an older book. Without it, a mid-session
# upgrade kept serving the pre-upgrade snapshot for the CURRENT time slot and
# the new names rendered NO DATA until the next 09:30/12:00/16:00 slot.
BOOK_FP = hashlib.sha1(",".join(ALL_TICKERS).encode()).hexdigest()[:8]
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
def _download_premarket() -> dict:
    """{ticker: (premarket_price, ts)} — today's PRE-market prints (bars before
    09:30 ET) from yfinance's prepost feed. Names with no premarket trading
    simply don't appear. One batched request, retried like every other call."""
    data = with_retry(lambda: yf.download(
        tickers=ALL_TICKERS, period="1d", interval="5m", prepost=True,
        group_by="ticker", auto_adjust=False, progress=False, threads=True),
        label="premarket")
    out: dict = {}
    open_min = 9 * 60 + 30
    today = pd.Timestamp.now(tz=MARKET_TZ).normalize()
    for t in _tickers_of(data):
        closes = _series(data, t, "Close")
        if closes is None or closes.empty:
            continue
        idx = closes.index
        if idx.tz is None:
            idx = idx.tz_localize(MARKET_TZ)
        else:
            idx = idx.tz_convert(MARKET_TZ)
        pre = closes[(idx.normalize() == today) &
                     (idx.hour * 60 + idx.minute < open_min)].dropna()
        if not pre.empty:
            out[t] = (float(pre.iloc[-1]), pre.index[-1])
    return out


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
    def _fetch_premarket(self, day_key: str) -> Optional[dict]:
        stt = self._state("premarket")
        try:
            t0 = time.perf_counter()
            raw = _download_premarket()
            self.cache.put(f"premarket-{day_key}", raw)
            stt.seconds = round(time.perf_counter() - t0, 2)
            stt.fetched, stt.stale, stt.n = True, False, len(raw)
            return raw
        except Exception as exc:
            stt.errors["_fetch"] = f"{type(exc).__name__}: {exc}"
            return None

    def premarket(self, day_key: str, wait: bool = True) -> dict:
        """{ticker: (premarket_price, ts)}. ~5-min TTL while premarket is open
        (04:00-09:30 ET weekdays), 30 min otherwise."""
        stt, key = self._state("premarket"), f"premarket-{day_key}"
        now = pd.Timestamp.now(tz=MARKET_TZ)
        in_pre = now.weekday() < 5 and 4 * 60 <= now.hour * 60 + now.minute < 9 * 60 + 30
        ttl = PREMARKET_TTL_S if in_pre else 30 * 60
        raw = self.cache.get(key, max_age_s=ttl)
        if raw is None and not self.offline:
            raw = self._run(key, lambda: self._fetch_premarket(day_key), wait=wait)
        if raw is None:
            raw = self.cache.get(key, max_age_s=None)
            if raw is not None:
                stt.stale, stt.note = True, "serving last cached premarket"
        return {t: (float(p), pd.Timestamp(ts)) for t, (p, ts) in (raw or {}).items()}

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
        if self.cache.get(f"premarket-{day_key}", 30 * 60) is None:
            self._start(f"premarket-{day_key}", lambda: self._fetch_premarket(day_key))
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
        for name in ("quotes", "ath", "fundamentals", "overview", "crypto", "premarket"):
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

# =============================================================================
# LM STUDIO — ask the article questions via a local LLM (OpenAI-compatible API)
# =============================================================================
LM_DEFAULT_BASE = "http://localhost:1234/v1"


def _lm_http():
    import requests  # yfinance dependency — always present
    return requests


def lm_list_models(base_url: str, timeout: float = 5.0, http=None):
    """GET {base}/models -> (model-ids | None, error). Injected `http` keeps
    this unit-testable without a live server."""
    http = http or _lm_http()
    try:
        r = http.get(base_url.rstrip("/") + "/models", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return [m.get("id", "?") for m in data.get("data", [])], ""
    except Exception as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)


def lm_chat(base_url: str, model: str, messages: list, temperature: float = 0.2,
            timeout: float = 240.0, http=None):
    """POST {base}/chat/completions -> (content | None, error)."""
    http = http or _lm_http()
    try:
        r = http.post(base_url.rstrip("/") + "/chat/completions",
                      json={"model": model, "messages": messages,
                            "temperature": temperature, "stream": False},
                      timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"], ""
    except Exception as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)


def _lm_system_prompt() -> str:
    levels = "; ".join(
        "%s buy %s-%s trim %s" % (s["t"], s.get("buy_lo"), s.get("buy_hi"), s.get("trim"))
        for s in MONITORED if s.get("buy_hi") is not None or s.get("trim") is not None)
    return (
        "You are an investing-analysis assistant. Answer the user's questions about Nadeem "
        "Walayat's latest article, grounded STRICTLY in the ARTICLE TEXT below — quote his "
        "levels and reasoning exactly. If the article does not cover something, say so "
        "plainly instead of guessing.\n\nCURRENT PORTFOLIO LEVELS (monitor): " + levels +
        "\n\nARTICLE TEXT:\n" + ARTICLE_TEXT)


def render_lm_studio_box() -> None:
    st.markdown("##### 🤖 Ask the article — LM Studio (local)")
    st.caption("Runs on YOUR machine: open LM Studio → Developer → Start Server (OpenAI-compatible "
               "server, default http://localhost:1234/v1), load a model, then Connect. The full "
               "article + current levels are sent as context with every question.")
    base = st.text_input("Server base URL", value=st.session_state.get("lm_base", LM_DEFAULT_BASE),
                         key="lm_base_input")
    if st.button("🔌 Connect", key="lm_connect"):
        models, err = lm_list_models(base)
        if models:
            st.session_state["lm_base"] = base
            st.session_state["lm_models"] = models
            st.session_state.setdefault("lm_model", models[0])
            st.success(f"Connected — {len(models)} model(s) available.")
        else:
            st.session_state.pop("lm_models", None)
            st.error(f"Could not reach LM Studio at {base} — {err}. Start the server "
                     f"(LM Studio → Developer → Start Server) and press Connect again.")
    models = st.session_state.get("lm_models")
    if models:
        st.selectbox("Model", models, key="lm_model")
    if "lm_history" not in st.session_state:
        st.session_state["lm_history"] = []
    for msg in st.session_state["lm_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    question = st.chat_input("Ask about the article…")
    if question:
        st.session_state["lm_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        base_url = st.session_state.get("lm_base", LM_DEFAULT_BASE)
        model = st.session_state.get("lm_model") or "local-model"
        with st.chat_message("assistant"):
            with st.spinner("Thinking (local model)…"):
                messages = ([{"role": "system", "content": _lm_system_prompt()}]
                            + st.session_state["lm_history"])
                answer, err = lm_chat(base_url, model, messages)
        if answer is None:
            st.error(f"LM Studio error — {err}")
            st.session_state["lm_history"].pop()   # drop the unanswered question
        else:
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state["lm_history"].append({"role": "assistant", "content": answer})
    if st.session_state.get("lm_history") and st.button("🗑️ Clear chat", key="lm_clear"):
        st.session_state["lm_history"] = []
        st.rerun()


def render_premarket_tab(premarket: dict, quotes: dict, near_pct: float) -> None:
    """[PREMARKET TAB] premarket prints with gap vs the previous close, the
    zone AT the premarket price, a 'gapping into buy range' callout and a
    big-movers list. Sorted by |gap|, biggest first."""
    st.subheader("🌅 Premarket")
    _how_to_box("How to use Premarket", [
        "Premarket prints (04:00–09:30 ET weekdays) with the gap vs the previous close.",
        "Zone colours are computed AT the premarket price — a gap-down INTO the buying "
        "range is the actionable premarket list (GREEN / ALL CAPS).",
        "Liquidity is thin before the open — treat premarket levels as indications, "
        "not fills. Big movers (|gap| ≥ 3%) are listed above the table.",
    ])
    now = pd.Timestamp.now(tz=MARKET_TZ)
    in_pre = now.weekday() < 5 and 4 * 60 <= now.hour * 60 + now.minute < 9 * 60 + 30
    latest = max((ts for _, ts in premarket.values()), default=None)
    if in_pre:
        st.success(f"🔴 PREMARKET OPEN (04:00–09:30 ET) — {len(premarket)} names printing; "
                   f"~5 min refresh."
                   + (f" Last print {latest.strftime('%H:%M')} ET." if latest is not None else ""))
    else:
        st.info("🌙 Premarket is closed (04:00–09:30 ET weekdays) — showing the most "
                "recent premarket prints.")
    if not premarket:
        st.caption("No premarket prints available yet.")
        return
    entries = []
    for t, (pre_px, pre_ts) in premarket.items():
        s = BY_TICKER.get(t)
        if s is None:
            continue
        q = quotes.get(t)
        prev_close = q[1] if q else None
        gap = (pre_px / prev_close - 1.0) if (prev_close and pre_px) else None
        entries.append((t, s, pre_px, pre_ts, prev_close, gap, zone_of(pre_px, s, near_pct)))
    entries.sort(key=lambda e: (-(abs(e[5]) if e[5] is not None else 0.0), e[0]))
    into_buy = [e for e in entries if e[6] == ZONE_BUY]
    into_trim = [e for e in entries if e[6] == ZONE_TRIM]
    movers = [e for e in entries if e[5] is not None and abs(e[5]) >= 0.03]
    if into_buy or into_trim or movers:
        bits = []
        if into_buy:
            bits.append("<span style='color:#00E676;font-weight:600;'>🟢 Gapping into buy "
                        "range (" + str(len(into_buy)) + "):</span> <span style='font-family:"
                        "monospace;color:#00E676;'>" + " ".join(_e(e[0]) for e in into_buy) + "</span>")
        if into_trim:
            bits.append("<span style='color:#FF5252;font-weight:600;'>🔴 At/above trim at "
                        "premarket (" + str(len(into_trim)) + "):</span> <span style='font-family:"
                        "monospace;color:#FF5252;'>" + " ".join(_e(e[0]) for e in into_trim) + "</span>")
        if movers:
            movers_txt = ", ".join(f"{_e(e[0])} {e[5] * 100:+.1f}%" for e in movers)
            bits.append("<span style='color:#FFD54F;font-weight:600;'>⚡ Big movers "
                        "(|gap| ≥ 3%):</span> <span style='font-family:monospace;color:#FFD54F;'>"
                        + movers_txt + "</span>")
        st.markdown("<div style='border:1px solid #263238;border-radius:8px;padding:8px 12px;"
                    "margin-bottom:12px;font-size:14px;line-height:1.6;'>"
                    + "<br>".join(bits) + "</div>", unsafe_allow_html=True)
    rows = []
    for t, s, pre_px, pre_ts, prev_close, gap, zone in entries:
        color = ZONE_COLORS.get(zone, "#9E9E9E")
        name = display_name(s, zone)
        gap_txt = "—" if gap is None else f"{gap * 100:+.2f}%"
        gap_color = ("#00E676" if (gap or 0) > 0 else ("#FF5252" if gap is not None else "#546E7A"))
        rows.append(
            "<tr style='border-bottom:1px solid #263238;'>"
            f"<td style='padding:5px 8px;font-family:monospace;color:#B0BEC5;'>{_e(t)}</td>"
            f"<td style='padding:5px 8px;color:{color};font-weight:600;'>{_e(name)}</td>"
            f"<td style='padding:5px 8px;color:{color};'>{fmt_money(pre_px)}</td>"
            f"<td style='padding:5px 8px;color:{gap_color};'>{gap_txt}</td>"
            f"<td style='padding:5px 8px;color:#B0BEC5;'>{fmt_money(prev_close) if prev_close else '—'}</td>"
            f"<td style='padding:5px 8px;color:{color};font-weight:600;'>{_e(status_text(s, zone, pre_px, near_pct))}</td>"
            f"<td style='padding:5px 8px;color:#B0BEC5;'>{_e(fmt_range(s))}</td>"
            f"<td style='padding:5px 8px;color:#B0BEC5;'>{_e(fmt_money(s['trim']) if s.get('trim') else '—')}</td>"
            f"<td style='padding:5px 8px;color:#546E7A;font-size:13px;'>{pre_ts.strftime('%H:%M')}</td></tr>")
    st.markdown(
        "<table style='width:100%;border-collapse:collapse;font-size:15.5px;'>"
        "<tr style='color:#78909C;text-align:left;border-bottom:1px solid #37474F;'>"
        "<th style='padding:4px 8px;'>Ticker</th><th>Company</th><th>Premarket</th>"
        "<th>Gap %</th><th>Prev Close</th><th>Status @ Pre</th><th>Buying Range</th>"
        "<th>Trim ≥</th><th>Print</th></tr>" + "".join(rows) + "</table>",
        unsafe_allow_html=True)
    no_print = len(ALL_TICKERS) - len(entries)
    st.caption(f"{len(entries)} names with premarket prints • {no_print} not yet printing • "
               "premarket levels are indications, not fills • zone computed at the premarket price.")


def render_article_tab(quotes: dict, zones: dict, near_pct: float) -> None:
    """[LATEST ARTICLE TAB] highlighted stocks at the top, full article text
    as-is below, then the LM Studio question box."""
    st.subheader(f"📰 {_e(ARTICLE_TITLE)}")
    st.caption(f"Nadeem Walayat • {ARTICLE_DATE} • full text as published, followed by "
               f"the stocks it calls out (live prices + zone colours).")
    rows = []
    for a in ARTICLE_STOCKS:
        s = BY_TICKER.get(a["t"])
        q = quotes.get(a["t"])
        price = q[0] if q else None
        zone = zones.get(a["t"], ZONE_NODATA if not q else ZONE_WAIT)
        color = ZONE_COLORS.get(zone, "#9E9E9E")
        name = display_name(s, zone) if s else a["t"]
        status = status_text(s, zone, price, near_pct) if s else "—"
        rows.append(
            "<tr style='border-bottom:1px solid #263238;'>"
            f"<td style='padding:5px 8px;font-family:monospace;color:#B0BEC5;'>{_e(a['t'])}</td>"
            f"<td style='padding:5px 8px;color:{color};font-weight:600;'>{_e(name)}</td>"
            f"<td style='padding:5px 8px;color:{color};'>{fmt_money(price) if price is not None else '—'}</td>"
            f"<td style='padding:5px 8px;color:{color};font-weight:600;'>{_e(status)}</td>"
            f"<td style='padding:5px 8px;color:#90CAF9;'>{_e(a['levels'])}</td>"
            f"<td style='padding:5px 8px;color:#78909C;'>{_e(a['why'])}</td></tr>")
    box = (
        "<div style='border:2px solid #4FC3F7; border-radius:10px; padding:10px 14px 12px; "
        "background:rgba(79,195,247,0.06); margin-bottom:14px;'>"
        "<h3 style='color:#4FC3F7;margin:4px 0 2px;'>🎯 Stocks in this article</h3>"
        f"<div style='color:#9E9E9E;font-size:12px;margin-bottom:6px;'>{_e(ARTICLE_CRYPTO)}</div>"
        "<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
        "<tr style='color:#78909C;text-align:left;border-bottom:1px solid #37474F;'>"
        "<th style='padding:4px 8px;'>Ticker</th><th>Company</th><th>Price</th>"
        "<th>Status</th><th>Article levels</th><th>Why it's mentioned</th></tr>"
        + "".join(rows) + "</table></div>")
    st.markdown(box, unsafe_allow_html=True)
    st.markdown(ARTICLE_TEXT)
    st.divider()
    render_lm_studio_box()


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
STORE_VERSION = "v4.1-book-fp"
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
        "WHITE = within 10% of buy top  •  RED/ALL CAPS = trim zone  •  Levels: 8 Sept article › 8 Sept "
        "sheet › Trade Wind › briefs")
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
    # [v4.1 FIX] keys carry the book fingerprint -> a changed ticker list can
    # never be served a pre-change snapshot from the disk cache.
    slot_key = f"{slot:%Y-%m-%d_%H-%M}_{BOOK_FP}"
    day_key = f"{now:%Y-%m-%d}_{BOOK_FP}"
    # Kick off refreshes in the background, then render from whatever is cached.
    try:
        store.prefetch(slot_key, day_key)
    except Exception:
        pass  # never block the UI on a background warm
    quotes = store.quotes(slot_key)
    aths = store.aths()
    funds, metrics_map = store.fundamentals(day_key)
    premarket = store.premarket(day_key)
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
    (tab_monitor, tab_pre, tab_article, tab_overview, tab_crypto, tab_egf,
     tab_fund, tab_big, tab_rules) = st.tabs([
        "📈 Monitor",
        "🌅 Premarket",
        "📰 Latest Article",
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
        st.caption("Sorted A–Z by ticker. Levels from the 8 Sept article › 8 Sept sheet › briefs.")
        rows = [build_row(s, (quotes.get(s["t"]) or (None, None, None))[0],
                          (quotes.get(s["t"]) or (None, None, None))[1],
                          zones.get(s["t"], ZONE_WAIT), near_pct,
                          aths.get(s["t"]), metrics_map.get(s["t"]))
                for s in stocks]
        st.markdown(table_html(rows), unsafe_allow_html=True)
        with st.expander("ℹ️ Sources, exclusions & crypto reference"):
            st.markdown(
                "**Article (8 Sept 2026 — Bonds Market Crisis)** — AI secondaries with fresh "
                "levels (MU $625/$476–625, LRCX raised to $226–$156, AMAT $324–$220, QCOM trim "
                "~$220, TSLA trim $400, IBM $208–$168, AMZN $229/$201) + LULU, FICO, CRCL "
                "mentions; article levels win where newer than the sheet.  \n"
                "**10X Brigade (17 Jul 2026)** — 14 ten-year candidates, boxed at the top, A–Z.  \n"
                "**Portfolio CSV (8 Sept)** — buying ranges + trim ladders synced (NVDA "
                "$148–$183/$237, AMD $180–$260, MU $476–$625, TSM $222–$322, META $448–$548, "
                "MSFT $336–$382, GOOG $228–$292, JNJ/HPQ/COIN/RBLX/MGNI trims).  \n"
                "**NFLX** — user addition: buy under $70.  \n"
                "**Excluded** — non-US listings (BESI static; SMSN.L, SMT.L, WTAI.L/INTL.L, "
                "RBTX.L, UKW.L, BDEV.L, PRX.NV) and delisted (MPW, RDFN).")
            st.markdown(CRYPTO_REFERENCE)
            if missing:
                st.caption("No data (check ticker on Yahoo Finance): " + ", ".join(missing))
    with tab_pre:
        render_premarket_tab(premarket, quotes, near_pct)

    with tab_article:
        render_article_tab(quotes, zones, near_pct)

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
