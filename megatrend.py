"""
Nadeem Walayat AI Portfolio Monitor — buying ranges & trim levels
Streamlit app powered by yfinance (REAL market data — no synthetic fallback).

DATA / REFRESH POLICY
---------------------
Prices update exactly THREE times per trading day: 09:30, 12:00 and 16:00 ET
(one batched yfinance download per snapshot). Outside those windows the app
shows the latest snapshot; a NEW browser session triggers exactly one fresh
fetch on open (so re-opening the app always shows current data), and an open
app auto-wakes at the next snapshot time. All-time-high distances refresh
once per day.

LAYOUT
------
Tab 1 "📈 Monitor"
  Compact ticker-strip header: the SYMBOLS in each zone (small font), not
  big number metrics.
  ⭐ 10X BRIGADE — hard-coded, always top & center (amber box, 14 ten-year
     10x candidates from the 17 Jul 2026 article, initial buying ranges +
     10-year targets). No toggle — it is always shown.
  Then ONE continuous portfolio table (no sub-sections): latest-article
  stocks first, then Trade Wind mention, portfolio sheet, Stocks Briefs
  additions, then small positions.
Tab 2 "📖 Rules to Remember" — the Investing Guide + Real Secret distilled.

COLUMNS: Ticker • Company • Price • Day % • Status • Buying Range • Trim ≥
• Target • Fwd P/E • PEG • EGF proxy • FScore • % from ATH. No notes column.

FUNDAMENTALS (from yfinance, refreshed once per day — slow-moving data):
  EGF proxy = forward vs trailing EPS growth (what his EGF measures; the exact
  EGF needs next-quarter estimates yfinance doesn't expose). FScore = a
  transparent 0-10 fundamentals score (EPS growth, EPS YoY, revenue growth,
  profit margin, ROE, forward PE < trailing PE, cash > debt, FCF > 0, gross
  margin, PEG ≤ 2). Tab "🔬 Fundamentals" carries the FULL metric set:
  trailing/forward P/E, PEG, EGF proxy, EPS YoY, revenue growth, gross /
  operating / profit margins, ROE, P/B, P/S, D/E, cash, debt, FCF, market cap.

ZONES:  GREEN + name in ALL CAPS = in the buying range (or below, "getting
lucky");  WHITE = within 10% above the buy top (adjustable);  RED + ALL CAPS
= at/above the trim level;  dim gray = wait.

Run:  pip install yfinance  &&  streamlit run ai_stocks_monitor.py
"""

import time
from typing import Optional

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
    YF_OK = True
except Exception:  # pragma: no cover — environment dependent
    yf = None
    YF_OK = False

MARKET_TZ = "America/New_York"
NEAR_PCT_DEFAULT = 10.0
SNAPSHOT_TIMES = [(9, 30), (12, 0), (16, 0)]   # the only price updates of the day


# ═══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT SCHEDULE — prices change only at 09:30 / 12:00 / 16:00 ET
# ═══════════════════════════════════════════════════════════════════════════════
def _prev_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    d = d - pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def current_snapshot_ts(now=None) -> pd.Timestamp:
    """Timestamp of the most recent scheduled snapshot (the data we display).
    Before 09:30 on a trading day -> previous trading day's 16:00. Weekends ->
    Friday 16:00."""
    if now is None:
        now = pd.Timestamp.now(tz=MARKET_TZ)
    if now.weekday() < 5:
        for h, m in reversed(SNAPSHOT_TIMES):
            slot = now.normalize() + pd.Timedelta(hours=h, minutes=m)
            if now >= slot:
                return slot
    prev = _prev_trading_day(now)
    return prev.normalize() + pd.Timedelta(hours=SNAPSHOT_TIMES[-1][0], minutes=SNAPSHOT_TIMES[-1][1])


def next_snapshot_ts(now=None) -> pd.Timestamp:
    """Timestamp of the next scheduled snapshot (when the app will wake)."""
    if now is None:
        now = pd.Timestamp.now(tz=MARKET_TZ)
    d = now.normalize()
    if now.weekday() < 5:
        for h, m in SNAPSHOT_TIMES:
            slot = d + pd.Timedelta(hours=h, minutes=m)
            if now < slot:
                return slot
    nxt = d + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt.normalize() + pd.Timedelta(hours=SNAPSHOT_TIMES[0][0], minutes=SNAPSHOT_TIMES[0][1])


def ms_until_next_snapshot(now_ms: float) -> int:
    nxt = next_snapshot_ts()
    return max(1000, int((nxt - pd.Timestamp.now(tz=MARKET_TZ)).total_seconds() * 1000) + 2000)


# ═══════════════════════════════════════════════════════════════════════════════
# ⭐ 10X BRIGADE — hard-coded top & center (17 Jul 2026 article). Buy ranges
# and 10-year targets exactly as published. No trim levels: long-run
# accumulation plays. BESI is Amsterdam-listed (€) — static, not monitored.
# ═══════════════════════════════════════════════════════════════════════════════
BRIGADE = [
    dict(t="CRM",   name="Salesforce",   buy_lo=130.0,  buy_hi=163.0,  target="720",
         note="[A comment] '$230 pumping', trim level asked — unanswered; exposure 125%."),
    dict(t="CRCL",  name="Circle",       buy_lo=50.0,   buy_hi=66.0,   target="640",
         note="[TW] trimming cryptos; exposure 126%."),
    dict(t="BESI",  name="BESI",         buy_lo=145.0,  buy_hi=194.0,  target="2000", static=True,
         note="Amsterdam-listed (€192.10 on the sheet) — not US, not live-monitored."),
    dict(t="DUOL",  name="Duolingo",     buy_lo=65.0,   buy_hi=105.0,  target="800",
         note="Exposure 48%."),
    dict(t="NOW",   name="ServiceNow",   buy_lo=68.0,   buy_hi=98.0,   target="1040",
         note="[TW] small sells; exposure 96%."),
    dict(t="NVO",   name="Novo Nordisk", buy_lo=35.0,   buy_hi=48.0,   target="250",
         note="[TW 11 Aug] in buying range; exposure 58%."),
    dict(t="FICO",  name="Fair Isaac",   buy_lo=830.0,  buy_hi=1170.0, target="6250",
         note="[TW 11 Aug] in buying range; exposure 124%."),
    dict(t="INTU",  name="Intuit",       buy_lo=235.0,  buy_hi=292.0,  target="1455",
         note="[TW] small sells; exposure 109%."),
    dict(t="VEEV",  name="Veeva",        buy_lo=138.0,  buy_hi=166.0,  target="1000",
         note="[TW] big sell 8% + trims; exposure 90%."),
    dict(t="ADBE",  name="Adobe",        buy_lo=190.0,  buy_hi=235.0,  target="1400",
         note="Exposure 137%."),
    dict(t="CLX",   name="Clorox",       buy_lo=82.0,   buy_hi=93.0,   target="480"),
    dict(t="SMCI",  name="SMCI",         buy_lo=18.0,   buy_hi=24.0,   target="240",
         note="Exposure 19%."),
    dict(t="QBTS",  name="QBTS",         buy_lo=4.0,    buy_hi=8.0,    target="98"),
    dict(t="PATH",  name="PATH",         buy_lo=8.0,    buy_hi=12.0,   target="120",
         note="[TW 18 Aug] small sell."),
]
BRIGADE_NOTE = ("Special section from the 17 Jul 2026 '10x Stocks to Accumulate' article • "
                "brigade +24.5% since mid-July • no trim levels — long-run accumulation "
                "(GREEN = in buying range, WHITE = within 10% of the top).")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LIST — ONE table: latest-article stocks first (26 Aug order), then the
# Trade Wind mention, the portfolio sheet (25 Aug), Stocks Briefs additions,
# then small positions. Brigade tickers are NOT repeated here. `note` fields
# are provenance for maintenance only — NOT rendered.
# ═══════════════════════════════════════════════════════════════════════════════
STOCKS = [
    # ── most recent article first (26 Aug order) ──────────────────────────────
    dict(t="AVGO", name="Broadcom", buy_lo=272.0, buy_hi=336.0, trim=495.0,
         mech="Only at ATH", target="470–500+",
         note="[A] support $355 — break targets $300/$288. [CSV] trim ATH 495."),
    dict(t="NVDA", name="NVIDIA", buy_lo=174.0, buy_hi=190.6, trim=219.0,
         mech="Author ladder", target="~275",
         note="[A 26 Aug] buys 190.6/188/186/181/177/174, sells 219/226/231/236/248."),
    dict(t="META", name="META", buy_lo=360.0, buy_hi=548.0, trim=717.0,
         mech="Within 10% of High", target="~730",
         note="[A] 450s → below 400 → as low as 360 (puke case 240)."),
    dict(t="AMD", name="AMD", buy_lo=180.0, buy_hi=300.0, trim=585.0,
         mech="Only at ATH", target="600",
         note="[A] won't add much above $300; dream drop $200."),
    dict(t="TSM", name="TSMC", buy_lo=315.0, buy_hi=390.0, trim=479.0,
         mech="Only at ATH", target=">500",
         note="[A] lightly adding sub 390, sweet spot ~330, support 315."),
    dict(t="ASML", name="ASML", buy_lo=1000.0, buy_hi=1326.0, trim=1900.0,
         mech="Within 5% of High", target=">2150",
         note="[A] 1300 then getting-lucky 1000."),
    dict(t="GOOG", name="Google", buy_lo=240.0, buy_hi=292.0, trim=404.0,
         mech="Only at ATH", target="430",
         note="[A] support 332 — break targets 300/272/240."),
    dict(t="MSFT", name="Microsoft", buy_lo=336.0, buy_hi=400.0, trim=500.0,
         mech="Within 10% of High", target="600",
         note="[A] buying opp toward 400, below = getting lucky."),
    dict(t="BIDU", name="Baidu", buy_lo=88.0, buy_hi=108.0, trim=None,
         note="[A] 'buy the dumps such as BIDU'; 18 Aug mega buys to $88."),
    dict(t="MRNA", name="Moderna", buy_lo=None, buy_hi=None, trim=None,
         note="[A] 'sell the pumps' — sold 70% into the 3x pop. No levels."),
    # ── Trade Wind mention ────────────────────────────────────────────────────
    dict(t="FCX", name="Freeport-McMoRan", buy_lo=None, buy_hi=50.0, trim=None,
         note="[TW 11 Aug] trim zone. [B] accumulate sub $50."),
    # ── portfolio sheet (25 Aug) ──────────────────────────────────────────────
    dict(t="QCOM", name="Qualcomm", buy_lo=122.0, buy_hi=152.0, trim=234.0, mech="Within 10% of High"),
    dict(t="LRCX", name="Lam Research", buy_lo=138.0, buy_hi=202.0, trim=417.0, mech="Within 5% of High"),
    dict(t="IBM", name="IBM", buy_lo=168.0, buy_hi=208.0, trim=299.0, mech="Within 10% of High"),
    dict(t="KLAC", name="KLAC", buy_lo=90.0, buy_hi=132.0, trim=292.0, mech="Within 5% of High"),
    dict(t="AMAT", name="AMAT", buy_lo=202.0, buy_hi=276.0, trim=703.0, mech="Within 5% of High"),
    dict(t="AMZN", name="Amazon", buy_lo=152.0, buy_hi=201.0, trim=258.0, mech="Within 10% of High"),
    dict(t="MU", name="Micron", buy_lo=132.0, buy_hi=312.0, trim=1192.0, mech="Within 5% of High"),
    dict(t="INTC", name="Intel", buy_lo=28.0, buy_hi=60.0, trim=135.0, mech="Within 5% of High"),
    dict(t="TSLA", name="Tesla", buy_lo=172.0, buy_hi=286.0, trim=449.0, mech="Within 10% of High"),
    dict(t="AAPL", name="Apple", buy_lo=190.0, buy_hi=226.0, trim=310.0, mech="Within 10% of High"),
    dict(t="LMT", name="Lockheed Martin", buy_lo=422.0, buy_hi=458.0, trim=623.0, mech="Within 10% of High"),
    dict(t="RTX", name="RTX", buy_lo=116.0, buy_hi=144.0, trim=204.0, mech="Within 10% of High"),
    dict(t="ARW", name="Arrow Electronics", buy_lo=106.0, buy_hi=132.0, trim=214.0, mech="Within 10% of High"),
    dict(t="FLEX", name="FLEX", buy_lo=42.0, buy_hi=70.0, trim=150.0, mech="Within 10% of High"),
    dict(t="GPN", name="GPN", buy_lo=62.0, buy_hi=68.0, trim=None),
    dict(t="JBL", name="Jabil", buy_lo=180.0, buy_hi=238.0, trim=386.0, mech="Within 10% of High"),
    dict(t="WDC", name="Western Digital", buy_lo=156.0, buy_hi=252.0, trim=720.0, mech="Within 10% of High"),
    dict(t="DIOD", name="Diodes", buy_lo=42.0, buy_hi=66.0, trim=113.0, mech="Within 10% of High"),
    dict(t="ON", name="ON Semiconductor", buy_lo=38.0, buy_hi=62.0, trim=121.0, mech="Within 10% of High"),
    dict(t="TAK", name="Takeda", buy_lo=12.0, buy_hi=13.0, trim=None),
    dict(t="ADSK", name="Autodesk", buy_lo=186.0, buy_hi=202.0, trim=310.0, mech="Within 10% of High"),
    dict(t="CRUS", name="Cirrus Logic", buy_lo=98.0, buy_hi=126.0, trim=162.0, mech="Within 10% of High"),
    dict(t="GFS", name="GlobalFoundries", buy_lo=32.0, buy_hi=48.0, trim=83.0, mech="Within 10% of High"),
    dict(t="HPQ", name="HP", buy_lo=16.0, buy_hi=18.6, trim=28.0, mech="$28–30"),
    dict(t="LOGI", name="Logitech", buy_lo=66.0, buy_hi=86.0, trim=126.0, mech="Within 10% of High"),
    dict(t="INMD", name="InMode", buy_lo=12.0, buy_hi=14.0, trim=None),
    dict(t="ULH", name="ULH", buy_lo=12.6, buy_hi=14.6, trim=None),
    dict(t="JNJ", name="JnJ", buy_lo=154.0, buy_hi=182.0, trim=249.0, mech="Within 10% of High"),
    dict(t="ABBV", name="AbbVie", buy_lo=155.0, buy_hi=167.0, trim=241.0, mech="Within 10% of High"),
    dict(t="PFE", name="Pfizer", buy_lo=22.0, buy_hi=24.3, trim=None),
    dict(t="FOR", name="Forestar", buy_lo=15.0, buy_hi=20.0, trim=37.0, mech="Within 10% of High"),
    dict(t="IIPR", name="IIPR", buy_lo=38.0, buy_hi=44.0, trim=None),
    dict(t="MPW", name="MPW", buy_lo=3.2, buy_hi=4.0, trim=None),
    dict(t="RDFN", name="Redfin", buy_lo=10.0, buy_hi=12.3, trim=None),
    dict(t="BABA", name="Alibaba", buy_lo=84.0, buy_hi=106.0, trim=None),
    dict(t="TCEHY", name="Tencent", buy_lo=40.0, buy_hi=55.0, trim=89.0, mech="Within 10% of High"),
    dict(t="MGNI", name="Magnite", buy_lo=8.6, buy_hi=11.6, trim=20.0, mech="$20–24"),
    dict(t="RBLX", name="Roblox", buy_lo=33.0, buy_hi=41.0, trim=60.0, mech="$60–69"),
    dict(t="SYNA", name="SYNA", buy_lo=46.0, buy_hi=70.0, trim=None),
    dict(t="DOCU", name="Docusign", buy_lo=40.0, buy_hi=45.0, trim=None),
    dict(t="CRSP", name="CRISPR", buy_lo=34.0, buy_hi=41.0, trim=None),
    dict(t="CSGP", name="CoStar", buy_lo=28.0, buy_hi=33.6, trim=None),
    dict(t="COIN", name="Coinbase", buy_lo=112.0, buy_hi=148.0, trim=211.0, mech="$211–232"),
    # ── Stocks Briefs additions ───────────────────────────────────────────────
    dict(t="OXY", name="Occidental", buy_lo=40.0, buy_hi=74.0, trim=None,
         note="[B] range trade $74–$40."),
    dict(t="SLB", name="SLB", buy_lo=32.0, buy_hi=60.0, trim=None,
         note="[B] $60–$32 range."),
    dict(t="FSLR", name="First Solar", buy_lo=None, buy_hi=200.0, trim=None,
         note="[B] accumulate; sub $200 is getting lucky."),
    dict(t="CCJ", name="Cameco", buy_lo=None, buy_hi=None, trim=None, note="[B] buy deep dip."),
    dict(t="BHP", name="BHP", buy_lo=None, buy_hi=None, trim=None, note="US ADR — briefs watch."),
    dict(t="ALB", name="Albemarle", buy_lo=None, buy_hi=None, trim=None, note="Briefs watch."),
    # ── small positions (price watch only) ────────────────────────────────────
    dict(t="AEHR", name="AEHR"),
    dict(t="AMT", name="American Tower"),
    dict(t="BKNG", name="Booking"),
    dict(t="GSK", name="GSK"),
    dict(t="PINS", name="Pinterest"),
    dict(t="SNAP", name="Snap"),
    dict(t="SNPS", name="Synopsys"),
    dict(t="TMO", name="Thermo Fisher"),
    dict(t="TOELY", name="Tokyo Electron"),
    dict(t="V", name="Visa"),
]

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
    "**Crypto reference (static, from the sheets — not monitored):** BTC $78,817 (27 Aug sheet) • "
    "article buy zones $68–64k / $60–58k / $56–48k • base case resolves lower unless BTC >$84k • "
    "MSTR fair value $99, cheap ≤$90, extreme ≥$207 • MSTR target highs $250/$300 ≈ BTC $95k/$114k."
)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA — yfinance ONLY (real market data). One batched download per snapshot
# for quotes (5d of 30m bars) and one per day for all-time highs (max daily).
# ═══════════════════════════════════════════════════════════════════════════════
ALL_TICKERS = [s["t"] for s in BRIGADE if not s.get("static")] + [s["t"] for s in STOCKS]


def _parse_intraday(data) -> dict:
    """MultiIndex (ticker, field) frame -> {ticker: (price, prev_close, ts)}.
    prev_close = last bar of the previous session."""
    out = {}
    if data is None or data.empty:
        return out
    tickers = list(data.columns.levels[0]) if isinstance(data.columns, pd.MultiIndex) else []
    for t in tickers:
        try:
            closes = data[t]["Close"].dropna()
        except Exception:
            continue
        if closes.empty:
            continue
        idx = closes.index
        price = float(closes.iloc[-1])
        ts = idx[-1]
        if ts.tzinfo is None:
            ts = ts.tz_localize(MARKET_TZ)
        else:
            ts = ts.tz_convert(MARKET_TZ)
        prev = None
        days = idx.normalize()
        earlier = closes[days < days[-1]]
        if len(earlier):
            prev = float(earlier.iloc[-1])
        out[str(t)] = (price, prev, ts)
    return out


def _download_quotes() -> dict:
    data = yf.download(tickers=ALL_TICKERS, period="5d", interval="30m",
                       group_by="ticker", auto_adjust=False, progress=False, threads=True)
    return _parse_intraday(data)


def _download_aths() -> dict:
    """{ticker: all-time-high price} from the full daily history (High)."""
    data = yf.download(tickers=ALL_TICKERS, period="max", interval="1d",
                       group_by="ticker", auto_adjust=False, progress=False, threads=True)
    out = {}
    if data is None or data.empty:
        return out
    for t in (data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else []):
        try:
            highs = data[t]["High"].dropna()
        except Exception:
            continue
        if not highs.empty:
            out[str(t)] = float(highs.max())
    return out


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching snapshot from yfinance…")
def _load_slot(slot_key: str, session_nonce: float) -> tuple:
    """Quotes for one snapshot slot. Keyed by (slot, session) so:
    - a NEW browser session forces exactly one fresh download on open;
    - reruns in the same session reuse the cache;
    - when the slot rolls over (09:30/12:00/16:00) the new key forces a fetch."""
    return _download_quotes(), pd.Timestamp.now(tz=MARKET_TZ)


@st.cache_data(ttl=24 * 3600, show_spinner="Fetching all-time highs from yfinance…")
def _load_aths(day_key: str) -> dict:
    return _download_aths()


# ── FUNDAMENTALS — one .info call per ticker, refreshed once per day ─────────
def _download_fundamentals() -> dict:
    """{ticker: info-dict} from yfinance (forwardPE, PEG, EPS, growth rates,
    margins, ROE, P/B, P/S, debt/cash, FCF, market cap). Per-ticker failures
    return {} and render as '—'."""
    if not YF_OK:
        return {}
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        try:
            return t, (yf.Ticker(t).info or {})
        except Exception:
            return t, {}

    out = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for t, info in pool.map(one, ALL_TICKERS):
            out[t] = info
    return out


@st.cache_data(ttl=24 * 3600, show_spinner="Fetching fundamentals from yfinance…")
def _load_fundamentals(day_key: str) -> dict:
    return _download_fundamentals()


def egf_proxy(info: dict) -> Optional[float]:
    """EGF proxy (%): forward vs trailing EPS growth — the quantity Walayat's
    EGF measures (his exact formula needs next-quarter EPS estimates, which
    yfinance does not expose). forwardEps/trailingEps - 1, equivalently
    trailingPE/forwardPE - 1. Positive = earnings growing into the multiple."""
    ttm_eps = info.get("trailingEps")
    fwd_eps = info.get("forwardEps")
    try:
        if ttm_eps and fwd_eps and float(ttm_eps) > 0 and float(fwd_eps) > 0:
            return (float(fwd_eps) / float(ttm_eps) - 1.0) * 100.0
    except Exception:
        pass
    ttm_pe = info.get("trailingPE")
    fwd_pe = info.get("forwardPE")
    try:
        if ttm_pe and fwd_pe and float(ttm_pe) > 0 and float(fwd_pe) > 0:
            return (float(ttm_pe) / float(fwd_pe) - 1.0) * 100.0
    except Exception:
        pass
    return None


def dcf_fair_value(info: dict, wacc: float = 0.095, terminal_g: float = 0.025) -> Optional[float]:
    """Simple per-share DCF estimate from yfinance fields:
        EV  = FCF x (1 + g) / (WACC - g)        (single-stage perpetuity)
        Equity = EV + cash - debt;  per share = Equity / shares outstanding
    g = earnings growth (revenue growth fallback), floored at 0 and capped at
    6% — no company compounds at 30% forever. Returns None when FCF <= 0,
    shares are missing, or WACC - g is not safely positive. A rough anchor,
    NOT a price target — see the explanation at the bottom of the tab."""
    fcf = info.get("freeCashflow")
    shares = info.get("sharesOutstanding")
    growth = info.get("earningsGrowth")
    if growth is None:
        growth = info.get("revenueGrowth")
    g = min(max(float(growth), 0.0), 0.06) if growth is not None else 0.03
    denom = float(wacc) - float(terminal_g)
    if not fcf or float(fcf) <= 0 or not shares or float(shares) <= 0 or denom <= 0.005:
        return None
    ev = float(fcf) * (1.0 + g) / denom
    equity = ev + float(info.get("totalCash") or 0.0) - float(info.get("totalDebt") or 0.0)
    return equity / float(shares)


def fmt_recommendation(key) -> tuple:
    """Yahoo recommendationKey -> (pretty label, color)."""
    if not key:
        return "—", "#546E7A"
    k = str(key)
    pretty = {"strong_buy": "Strong Buy", "buy": "Buy", "hold": "Hold",
              "underperform": "Underperform", "sell": "Sell",
              "none": "No rating"}.get(k, k.replace("_", " ").title())
    color = "#00E676" if k in ("strong_buy", "buy") else \
            ("#FF5252" if k in ("sell", "underperform") else "#FFD54F")
    return pretty, color


def fundamentals_score(info: dict) -> Optional[int]:
    """Transparent 0-10 'Fundamentals' score computed from yfinance fields
    (mirrors the spirit of his 0-10 column: PE, EPS, revenue, cash flow, ROE).
    +1 for each: forward EPS growth > 0 • EPS YoY > 0 • revenue growth > 0 •
    profit margin > 10% • ROE > 15% • forward PE < trailing PE • cash > debt •
    FCF > 0 • gross margin > 30% • PEG in (0, 2]. Missing fields count as
    failures (score skews conservative)."""
    if not info:
        return None
    peg = info.get("trailingPegRatio") or info.get("pegRatio")
    fwd_pe, ttm_pe = info.get("forwardPE"), info.get("trailingPE")
    checks = [
        (egf_proxy(info) or 0.0) > 0,
        (info.get("earningsGrowth") or 0) > 0,
        (info.get("revenueGrowth") or 0) > 0,
        (info.get("profitMargins") or 0) > 0.10,
        (info.get("returnOnEquity") or 0) > 0.15,
        bool(fwd_pe) and bool(ttm_pe) and float(fwd_pe) < float(ttm_pe),
        (info.get("totalCash") or 0) > (info.get("totalDebt") or 0),
        (info.get("freeCashflow") or 0) > 0,
        (info.get("grossMargins") or 0) > 0.30,
        bool(peg) and 0.0 < float(peg) <= 2.0,
    ]
    return sum(1 for c in checks if c)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE LOGIC (pure — unit-testable)
# ═══════════════════════════════════════════════════════════════════════════════
ZONE_BUY, ZONE_NEAR, ZONE_TRIM, ZONE_WAIT, ZONE_NODATA = "BUY", "NEAR", "TRIM", "WAIT", "NODATA"
ZONE_COLORS = {ZONE_BUY: "#00E676", ZONE_NEAR: "#FFFFFF", ZONE_TRIM: "#FF5252",
               ZONE_WAIT: "#9E9E9E", ZONE_NODATA: "#616161"}


def zone_of(price: Optional[float], stock: dict, near_pct: float = NEAR_PCT_DEFAULT) -> str:
    """BUY: in or below the buying range (green, ALL CAPS).
    NEAR: within near_pct above the buy-range top (white).
    TRIM: at/above the trim level (red, ALL CAPS).
    WAIT: otherwise (dim). NODATA: no quote."""
    if price is None:
        return ZONE_NODATA
    trim = stock.get("trim")
    buy_hi = stock.get("buy_hi")
    if trim is not None and price >= float(trim):
        return ZONE_TRIM
    if buy_hi is not None and price <= float(buy_hi):
        return ZONE_BUY
    if buy_hi is not None:
        near_top = float(buy_hi) + float(buy_hi) * float(near_pct) / 100.0  # exact +N%
        if price <= near_top:
            return ZONE_NEAR
    return ZONE_WAIT


def display_name(stock: dict, zone: str) -> str:
    """Company name in ALL CAPS exactly when in the buy or sell (trim) zone."""
    return stock["name"].upper() if zone in (ZONE_BUY, ZONE_TRIM) else stock["name"]


def fmt_money(v):
    if v is None:
        return "—"
    return f"${v:,.2f}" if v < 1000 else f"${v:,.0f}"


def fmt_range(stock):
    lo, hi = stock.get("buy_lo"), stock.get("buy_hi")
    if hi is None:
        return "—"
    if lo is None:
        return f"≤ {fmt_money(hi)}"
    return f"{fmt_money(lo)} – {fmt_money(hi)}"


def fmt_ath(price, ath) -> str:
    """Distance from the all-time high, e.g. '-12.3%'."""
    if price is None or not ath:
        return "—"
    return f"{(price / float(ath) - 1.0) * 100.0:+.1f}%"


def fmt_g(v) -> str:
    """Growth decimal (0.31 = 31%) -> '+31.0%'."""
    if v is None:
        return "—"
    return f"{float(v) * 100.0:+.1f}%"


def fmt_n(v, digits: int = 1) -> str:
    if v is None:
        return "—"
    return f"{float(v):.{digits}f}"


def fmt_usd_b(v) -> str:
    """Raw dollars -> $bn (e.g. 30e9 -> $30.0B)."""
    if v is None:
        return "—"
    return f"${float(v) / 1e9:,.1f}B"


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
def build_row(stock, price, prev, zone, near_pct, ath, fund=None) -> str:
    color = ZONE_COLORS[zone]
    name = display_name(stock, zone)
    if zone == ZONE_BUY and stock.get("buy_lo") is not None and price < float(stock["buy_lo"]):
        status = "🟢 IN BUY RANGE (below — getting lucky)"
    elif zone == ZONE_BUY:
        status = "🟢 IN BUY RANGE"
    elif zone == ZONE_NEAR:
        status = f"⚪ WITHIN {near_pct:.0f}% OF BUY TOP"
    elif zone == ZONE_TRIM:
        status = "🔴 TRIM / SELL ZONE"
    elif zone == ZONE_NODATA:
        status = "NO DATA"
    else:
        status = "wait"
    day = "—" if (price is None or not prev) else f"{(price / prev - 1) * 100:+.2f}%"
    mech = stock.get("mech")
    trim_txt = "—" if stock.get("trim") is None else fmt_money(stock["trim"]) + (f" <span style='color:#546E7A;font-size:13px;'>({mech})</span>" if mech else "")
    fund = fund or {}
    fwd_pe = fund.get("forwardPE")
    peg = fund.get("trailingPegRatio") or fund.get("pegRatio")
    egf = egf_proxy(fund)
    score = fundamentals_score(fund)
    egf_color = "#00E676" if (egf or 0) > 0 else ("#FF5252" if egf is not None else "#546E7A")
    if score is None:
        score_txt, score_color = "—", "#546E7A"
    else:
        score_txt = f"{score}/10"
        score_color = "#00E676" if score >= 7 else ("#FFD54F" if score >= 4 else "#FF5252")
    return (
        f"<tr style='border-bottom:1px solid #263238;'>"
        f"<td style='padding:5px 8px; color:#B0BEC5; font-family:monospace;'>{stock['t']}</td>"
        f"<td style='padding:5px 8px; color:{color}; font-weight:600;'>{name}</td>"
        f"<td style='padding:5px 8px; color:{color};'>{fmt_money(price) if price is not None else '—'}</td>"
        f"<td style='padding:5px 8px; color:#B0BEC5;'>{day}</td>"
        f"<td style='padding:5px 8px; color:{color}; font-weight:600;'>{status}</td>"
        f"<td style='padding:5px 8px; color:#B0BEC5;'>{fmt_range(stock)}</td>"
        f"<td style='padding:5px 8px; color:#B0BEC5;'>{trim_txt}</td>"
        f"<td style='padding:5px 8px; color:#B0BEC5;'>{stock.get('target') or '—'}</td>"
        f"<td style='padding:5px 8px; color:#B0BEC5;'>{fmt_n(fwd_pe)}</td>"
        f"<td style='padding:5px 8px; color:#B0BEC5;'>{fmt_n(peg)}</td>"
        f"<td style='padding:5px 8px; color:{egf_color};'>{fmt_g(egf / 100.0) if egf is not None else '—'}</td>"
        f"<td style='padding:5px 8px; color:{score_color}; font-weight:600;'>{score_txt}</td>"
        f"<td style='padding:5px 8px; color:#B0BEC5;'>{fmt_ath(price, ath)}</td>"
        f"</tr>"
    )


def build_static_row(stock) -> str:
    """Non-US / unmonitored row (e.g. BESI) — static info, gray."""
    return (
        f"<tr style='border-bottom:1px solid #263238;'>"
        f"<td style='padding:5px 8px; color:#8D6E63; font-family:monospace;'>{stock['t']}</td>"
        f"<td style='padding:5px 8px; color:#8D6E63; font-weight:600;'>{stock['name']}</td>"
        f"<td style='padding:5px 8px; color:#8D6E63;'>—</td>"
        f"<td style='padding:5px 8px; color:#8D6E63;'>—</td>"
        f"<td style='padding:5px 8px; color:#616161;'>NOT MONITORED (non-US)</td>"
        f"<td style='padding:5px 8px; color:#8D6E63;'>{fmt_range(stock)}</td>"
        f"<td style='padding:5px 8px; color:#8D6E63;'>—</td>"
        f"<td style='padding:5px 8px; color:#8D6E63;'>{stock.get('target') or '—'}</td>"
        f"<td style='padding:5px 8px; color:#8D6E63;'>—</td>"
        f"<td style='padding:5px 8px; color:#8D6E63;'>—</td>"
        f"<td style='padding:5px 8px; color:#8D6E63;'>—</td>"
        f"<td style='padding:5px 8px; color:#8D6E63;'>—</td>"
        f"<td style='padding:5px 8px; color:#8D6E63;'>—</td>"
        f"</tr>"
    )


def table_html(rows, target_label="12m Target") -> str:
    return (
        "<table style='width:100%; border-collapse:collapse; font-size:15.5px;'>"
        "<tr style='color:#78909C; text-align:left; border-bottom:1px solid #37474F;'>"
        "<th style='padding:4px 8px;'>Ticker</th><th>Company</th><th>Price</th>"
        "<th>Day %</th><th>Status</th><th>Buying Range</th><th>Trim ≥</th>"
        f"<th>{target_label}</th><th>Fwd P/E</th><th>PEG</th><th>EGF proxy</th>"
        "<th>FScore</th><th>% from ATH</th></tr>"
        + "".join(rows) + "</table>"
    )


def render_zone_strip(zone_map, quotes, near_pct) -> None:
    """Compact header: the SYMBOLS in each zone (small font) instead of big
    number metrics — keeps the top of the page light."""
    groups = [
        ("🟢 IN BUY RANGE", ZONE_BUY, ZONE_COLORS[ZONE_BUY]),
        (f"⚪ WITHIN {near_pct:.0f}% OF BUY TOP", ZONE_NEAR, ZONE_COLORS[ZONE_NEAR]),
        ("🔴 TRIM / SELL ZONE", ZONE_TRIM, ZONE_COLORS[ZONE_TRIM]),
        ("NO DATA", ZONE_NODATA, ZONE_COLORS[ZONE_NODATA]),
    ]
    lines = []
    for label, zone, color in groups:
        tickers = sorted(t for t, z in zone_map.items() if z == zone)
        if not tickers:
            continue
        lines.append(
            f"<div style='font-size:14px; line-height:1.55; margin:1px 0;'>"
            f"<span style='font-weight:600;'>{label} <span style='color:#78909C;'>({len(tickers)})</span>:</span> "
            f"<span style='color:{color}; font-family:monospace;'>{' '.join(tickers)}</span></div>"
        )
    latest = max((q[2] for q in quotes.values() if q), default=None)
    if latest is not None:
        lines.append(f"<div style='font-size:13px; color:#546E7A; margin-top:2px;'>"
                     f"quotes as of {latest.strftime('%d %b %H:%M')} ET</div>")
    st.markdown("<div style='border:1px solid #263238; border-radius:8px; padding:8px 12px; "
                "margin-bottom:12px;'>" + "".join(lines) + "</div>", unsafe_allow_html=True)


def render_fundamentals_tab(stocks_list, quotes, zones, funds,
                             wacc: float = 0.095, terminal_g: float = 0.025):
    """Full metric set from yfinance (one row per monitored ticker) + the
    computed DCF estimate and Yahoo analyst recommendation / mean target.
    Column explanations and the FScore methodology sit at the bottom."""
    headers = ["Ticker", "Company", "Price", "Fwd P/E", "Trail P/E", "PEG",
               "EGF proxy", "EPS YoY", "Rev growth", "Gross M", "Oper M",
               "Profit M", "ROE", "P/B", "P/S", "D/E", "Cash", "Debt", "FCF",
               "Mkt Cap", "FScore", "DCF/sh", "DCF upside", "Analyst Rec",
               "Analyst Tgt", "Tgt upside"]
    head = "".join(f"<th style='padding:4px 6px;'>{h}</th>" for h in headers)
    rows = []
    for s in stocks_list:
        q = quotes.get(s["t"])
        price = q[0] if q else None
        zone = zones.get(s["t"], ZONE_WAIT)
        f = funds.get(s["t"]) or {}
        color = ZONE_COLORS[zone]
        score = fundamentals_score(f)
        score_html = "—" if score is None else (
            f"<span style='color:{'#00E676' if score >= 7 else ('#FFD54F' if score >= 4 else '#FF5252')};'>"
            f"{score}/10</span>")
        egf = egf_proxy(f)
        egf_html = "—" if egf is None else (
            f"<span style='color:{'#00E676' if egf > 0 else '#FF5252'};'>{egf:+.1f}%</span>")
        de = "—" if not (f.get("totalDebt") and f.get("totalCash")) else \
            (f"{float(f['totalDebt']) / max(float(f['totalCash']), 1.0):.1f}" if f.get("totalCash") else "—")
        dcf = dcf_fair_value(f, wacc, terminal_g)
        if dcf is None or price is None:
            dcf_html, dcf_up_html = "—", "—"
        else:
            up = dcf / price - 1.0
            col = "#00E676" if up > 0.05 else ("#FF5252" if up < -0.05 else "#FFD54F")
            dcf_html = fmt_money(dcf)
            dcf_up_html = f"<span style='color:{col};'>{up * 100:+.1f}%</span>"
        tgt = f.get("targetMeanPrice")
        rec_html, _rc = fmt_recommendation(f.get("recommendationKey"))
        rec_html = f"<span style='color:{_rc};'>{rec_html}</span>"
        if tgt and price:
            tgt_html = fmt_money(float(tgt))
        else:
            tgt_html = "—"
        cells = [
            s["t"], display_name(s, zone), fmt_money(price) if price is not None else "—",
            fmt_n(f.get("forwardPE")), fmt_n(f.get("trailingPE")),
            fmt_n(f.get("trailingPegRatio") or f.get("pegRatio")),
            egf_html, fmt_g(f.get("earningsGrowth")), fmt_g(f.get("revenueGrowth")),
            fmt_g(f.get("grossMargins")), fmt_g(f.get("operatingMargins")),
            fmt_g(f.get("profitMargins")), fmt_g(f.get("returnOnEquity")),
            fmt_n(f.get("priceToBook")), fmt_n(f.get("priceToSalesTrailing12Months")),
            de, fmt_usd_b(f.get("totalCash")), fmt_usd_b(f.get("totalDebt")),
            fmt_usd_b(f.get("freeCashflow")), fmt_usd_b(f.get("marketCap")),
            score_html, dcf_html, dcf_up_html, rec_html, tgt_html,
            (f"{(float(tgt) / price - 1.0) * 100:+.1f}%" if (tgt and price) else "—"),
        ]
        tds = "".join(
            f"<td style='padding:4px 6px; font-size:14px; color:{'#B0BEC5'};'>{c}</td>"
            if i not in (1,) else
            f"<td style='padding:4px 6px; font-size:14px; color:{color}; font-weight:600;'>{c}</td>"
            for i, c in enumerate(cells))
        rows.append(f"<tr style='border-bottom:1px solid #263238;'>{tds}</tr>")
    st.markdown(
        "<table style='width:100%; border-collapse:collapse; font-size:14px;'>"
        f"<tr style='color:#78909C; text-align:left; border-bottom:1px solid #37474F;'>{head}</tr>"
        + "".join(rows) + "</table>", unsafe_allow_html=True)

    # ── BOTTOM: column explanations + FScore methodology (per user request) ──
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
        ("D/E", "total debt ÷ total cash (1.0 = debt equals cash). Cash / Debt / FCF in $B."),
        ("Mkt Cap", "market value of the whole company."),
        ("DCF/sh", "COMPUTED fair-value estimate: FCF × (1+g) ÷ (WACC − g) + cash − debt, per share. "
                    "g = growth capped at 6%; WACC & terminal growth are the sidebar sliders. "
                    "n/a when FCF ≤ 0. A rough anchor, NOT a price target."),
        ("Analyst Rec / Tgt", "Yahoo's aggregated analyst rating and mean 12-month price target "
                               "with implied upside vs the current price."),
    ]
    st.markdown(
        "<table style='width:100%; border-collapse:collapse; font-size:13px;'>"
        + "".join(
            f"<tr style='border-bottom:1px solid #263238;'>"
            f"<td style='padding:4px 8px; color:#90CAF9; font-weight:600; width:170px;'>{k}</td>"
            f"<td style='padding:4px 8px; color:#B0BEC5;'>{v}</td></tr>" for k, v in expl)
        + "</table>", unsafe_allow_html=True)

    st.subheader("⭐ FScore — the 0-10 fundamentals score, explained")
    st.markdown(
        "Walayat maintains a time-consuming hand-built 0-10 Fundamentals column. FScore is a "
        "transparent, automatable stand-in built from the same ideas (PE, EPS, revenue, cash "
        "flow, ROE). **+1 point for each check passed; missing data counts as a FAIL, so the "
        "score skews conservative:**"
    )
    checks = [
        "1. Forward EPS growth > 0 — earnings are growing into the multiple (the EGF idea).",
        "2. EPS YoY > 0 — the last quarter actually earned more than a year ago.",
        "3. Revenue growth > 0 — the top line is still expanding.",
        "4. Profit margin > 10% — the business keeps a healthy slice of revenue.",
        "5. ROE > 15% — capital is being put to work efficiently.",
        "6. Forward P/E < Trailing P/E — the multiple is SHRINKING as earnings grow (his 'gets cheaper over time').",
        "7. Total cash > total debt — balance-sheet safety.",
        "8. Free cash flow > 0 — the business actually generates cash.",
        "9. Gross margin > 30% — pricing power / quality of the business.",
        "10. PEG between 0 and 2 — valuation not detached from growth.",
    ]
    for c in checks:
        st.markdown(f"- {c}")
    st.caption(
        "Reading it: 8-10 green = strong fundamentals (his 'epic' territory); 4-7 amber = mixed, "
        "check the individual columns; 0-3 red = weak — be extra demanding on the buying range. "
        "Fundamentals refresh once per day."
    )


def inject_theme() -> None:
    """[THEME] main window = deep blue, sidebar = deep plum (user preference).
    Injected as CSS so the app stays a single file with no config.toml."""
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
    st.markdown(
        f"<div style='border:2px solid #FFB300; border-radius:10px; padding:12px 16px; "
        f"background:rgba(255,179,0,0.07); font-size:16px; color:#FFD54F; font-weight:600;'>"
        f"🔑 {MANTRA}</div><br>", unsafe_allow_html=True)
    st.markdown("**The 6 Real Secrets for Successful Trading**")
    for s in REAL_SECRETS:
        st.markdown(f"- {s}")
    st.divider()
    col1, col2 = st.columns(2)
    for i, (title, bullets) in enumerate(GUIDE_GROUPS):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"**{title}**")
            for b in bullets:
                st.markdown(f"- {b}")
            st.markdown("")
    st.caption("Distilled from the 'Real Secret' (Jan 2019) and 'Investing Guide' tabs of the "
               "AI Tech Stocks Portfolio spreadsheet — originals there for the full text.")


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

    # Wake exactly at the next snapshot (09:30 / 12:00 / 16:00 ET, skipping
    # weekends). Between snapshots nothing refetches.
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=ms_until_next_snapshot(time.time() * 1000.0),
                       key=f"snapshot_{slot.strftime('%Y%m%d_%H%M')}")
    except ImportError:
        st.sidebar.caption("⚠️ streamlit-autorefresh not installed — data refreshes when the app re-opens.")

    st.title("📊 AI Portfolio")
    st.caption(
        "Accumulate the dumps, distribute the pumps  •  GREEN/ALL CAPS = in buying range (or below)  •  "
        "WHITE = within 10% of buy top  •  RED/ALL CAPS = trim zone  •  Levels: 26 Aug article › author "
        "comments › Trade Wind › portfolio sheet 25 Aug › briefs"
    )

    # per-browser-session nonce: first render of a new session forces ONE
    # fresh download; reruns reuse the snapshot cache.
    if "_session_nonce" not in st.session_state:
        st.session_state["_session_nonce"] = time.time()
    quotes, fetched_at = _load_slot(slot.strftime("%Y-%m-%d_%H:%M"),
                                    st.session_state["_session_nonce"])
    aths = _load_aths(now.strftime("%Y-%m-%d"))
    funds = _load_fundamentals(now.strftime("%Y-%m-%d"))

    market_open = now.weekday() < 5 and 9 * 60 + 30 <= now.hour * 60 + now.minute < 16 * 60
    if market_open:
        st.success(f"🔴 Snapshot {slot.strftime('%H:%M')} ET — prices update 09:30 / 12:00 / 16:00 ET; "
                   f"next update {nxt.strftime('%a %H:%M')} ET.")
    else:
        st.info(f"🌙 MARKET CLOSED — latest snapshot {slot.strftime('%a %d %b %H:%M')} ET. Prices "
                f"update only at 09:30 / 12:00 / 16:00 ET on trading days; re-open the app anytime "
                f"to refresh.")

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
        st.caption(
            "Data: Yahoo Finance (yfinance), one batched download per snapshot. Levels are Nadeem "
            "Walayat's published numbers. Monitor only — no orders. Not investment advice."
        )

    monitored = [s for s in BRIGADE if not s.get("static")] + STOCKS
    quotes = {s["t"]: quotes.get(s["t"]) for s in monitored}
    zones = {s["t"]: zone_of(quotes[s["t"]][0] if quotes[s["t"]] else None, s, near_pct)
             for s in monitored}

    tab_monitor, tab_fund, tab_rules = st.tabs(
        ["📈 Monitor", "🔬 Fundamentals", "📖 Rules to Remember"])

    with tab_monitor:
        # compact symbol strip (no big number metrics)
        render_zone_strip(zones, quotes, near_pct)

        # ── ⭐ 10X BRIGADE — hard-coded, always top & center ───────────────────
        rows = []
        for s in BRIGADE:
            if s.get("static"):
                rows.append(build_static_row(s))
                continue
            q = quotes[s["t"]]
            price, prev = (q[0], q[1]) if q else (None, None)
            rows.append(build_row(s, price, prev, zones[s["t"]], near_pct,
                                  aths.get(s["t"]), funds.get(s["t"])))
        box = (
            "<div style='border:2px solid #FFB300; border-radius:10px; "
            "padding:10px 14px 12px; background:rgba(255,179,0,0.06); margin-bottom:16px;'>"
            "<h3 style='color:#FFB300; margin:4px 0 2px;'>⭐ 10X BRIGADE</h3>"
            f"<div style='color:#9E9E9E; font-size:12px; margin-bottom:6px;'>{BRIGADE_NOTE}</div>"
            + table_html(rows, target_label="10Yr Target") + "</div>"
        )
        st.markdown(box, unsafe_allow_html=True)

        # ── main list: one table, latest-article stocks first ─────────────────
        stocks = STOCKS
        if only_actionable:
            stocks = [s for s in stocks if zones.get(s["t"]) in (ZONE_BUY, ZONE_NEAR, ZONE_TRIM)]
        st.subheader("Portfolio")
        rows = []
        for s in stocks:
            q = quotes[s["t"]]
            price, prev = (q[0], q[1]) if q else (None, None)
            rows.append(build_row(s, price, prev, zones[s["t"]], near_pct,
                                  aths.get(s["t"]), funds.get(s["t"])))
        st.markdown(table_html(rows), unsafe_allow_html=True)

        with st.expander("ℹ️ Sources, exclusions & crypto reference"):
            st.markdown(
                "**Article (26 Aug 2026)** — the 8 primaries (levels win where newer than the sheet) "
                "+ author comments (NVDA stacked orders, BIDU / MRNA / CRM).  \n"
                "**10X Brigade (17 Jul 2026)** — 14 ten-year candidates, always boxed at the top.  \n"
                "**Trade Wind (11–25 Aug)** — FCX trim zone; brigade entries carry TW notes.  \n"
                "**Portfolio CSV (25 Aug)** — buying ranges + trim mechanisms.  \n"
                "**Stocks Briefs** — OXY ($40–74) / SLB ($32–60) ranges, FSLR sub-$200.  \n"
                "**Excluded** — non-US listings (BESI shown static, SMSN.L, SMT.L, WTAI.L/INTL.L, "
                "RBTX.L, UKW.L, BDEV.L, PRX.NV) and dead rows (MED, APM, CRSR, U, BPMC, BDSI)."
            )
            st.markdown(CRYPTO_REFERENCE)
            failures = {t: "no data" for t, q in quotes.items() if not q}
            if failures:
                st.caption("No data (check ticker on Yahoo Finance): " + ", ".join(sorted(failures)))

    with tab_fund:
        st.subheader("🔬 Fundamentals — full yfinance metric set")
        render_fundamentals_tab(monitored, quotes, zones, funds,
                                wacc=wacc_pct / 100.0, terminal_g=tg_pct / 100.0)

    with tab_rules:
        render_rules_tab()

    st.caption(
        f"Feed: yfinance  •  prices: snapshot {slot.strftime('%a %d %b %H:%M')} ET, fetched "
        f"{fetched_at.strftime('%H:%M:%S') if fetched_at is not None else '—'}  •  next update "
        f"{nxt.strftime('%a %H:%M')} ET  •  {len(monitored)} live tickers + "
        f"{sum(1 for s in BRIGADE if s.get('static'))} static  •  fundamentals refresh daily  •  "
        f"Monitor only — not investment advice."
    )


if __name__ == "__main__":
    main()
