"""
backtest_core.py — DAILY-ONLY mean-reversion backtest utilities for S&P 500
- Daily returns, trailing N-day total return, next-day execution
- No shorting; benchmark = Buy & Hold
- All tunables surfaced via Config dataclass (see below)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =============== CONFIG (visible at the top for easy reading) ===============
@dataclass
class Config:
    # Data
    source: str = "yahoo_spy"     # yahoo_spy | yahoo_gspc | stooq_spx | fred_sp500 | csv
    csv_path: str = ""            # used when source="csv"
    price_col: str = "Adj Close"
    date_col: str = "Date"
    start_date: str = "1993-02-01"

    # Strategy (DAILY-ONLY)
    lookback_days: int = 252      # trailing window in trading days
    buy_th: float = -0.017        # e.g., -1.7% (30th percentile from 97y histogram)
    sell_th: float = 0.111        # e.g., +11.1% (median from 97y histogram)
    hysteresis: float = 0.0       # extra buffer to reduce whipsaws (e.g., 0.005 = 0.5%)
    signal_lag_days: int = 1      # execute next day
    min_hold_days: int = 0        # optional minimum holding period

    # Costs / realism
    tcost_bps: float = 2.0        # per position change
    start_invested: bool = True   # initial position

# ========================== Data acquisition ==========================
def fetch_yahoo(symbol: str, start="1990-01-01", auto_adjust=True) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, start=start, progress=False, auto_adjust=auto_adjust)
    if df.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol}.")
    df = df.rename_axis("Date").reset_index()
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    return df

def fetch_stooq(symbol="^spx") -> pd.DataFrame:
    try:
        from pandas_datareader.stooq import StooqDailyReader
        rdr = StooqDailyReader(symbol)
        df = rdr.read().sort_index().reset_index().rename(columns={"Close":"Adj Close"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception:
        import requests, io, urllib.parse
        base = "https://stooq.com/q/d/l/"
        params = {"s": symbol, "i": "d"}
        url = base + "?" + urllib.parse.urlencode(params)
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or len(r.text) < 50:
            raise RuntimeError(f"Stooq download failed: HTTP {r.status_code}")
        df = pd.read_csv(io.StringIO(r.text))
        if "Close" in df.columns:
            df = df.rename(columns={"Close":"Adj Close"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df

def fetch_fred_sp500() -> pd.DataFrame:
    from pandas_datareader import data as pdr
    df = pdr.DataReader("SP500", "fred")
    df = df.dropna().rename(columns={"SP500":"Adj Close"}).reset_index()
    df["Date"] = pd.to_datetime(df["DATE"] if "DATE" in df.columns else df["Date"])
    if "DATE" in df.columns: df = df.drop(columns=["DATE"])
    return df

def load_data(cfg: Config) -> pd.DataFrame:
    if cfg.source == "csv":
        if not cfg.csv_path:
            raise ValueError("csv_path must be provided when source='csv'")
        df = pd.read_csv(cfg.csv_path)
    elif cfg.source == "yahoo_spy":
        df = fetch_yahoo("SPY", start=cfg.start_date, auto_adjust=True)
    elif cfg.source == "yahoo_gspc":
        df = fetch_yahoo("^GSPC", start=cfg.start_date, auto_adjust=False)
    elif cfg.source == "stooq_spx":
        df = fetch_stooq("^spx")
    elif cfg.source == "fred_sp500":
        df = fetch_fred_sp500()
    else:
        raise ValueError(f"Unknown source: {cfg.source}")

    # Normalize column names to strings (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    # If index is DateTimeIndex and no date column, bring index into a column
    if isinstance(df.index, pd.DatetimeIndex) and "Date" not in df.columns and "DATE" not in df.columns:
        df = df.reset_index()

    # Helper to find a column case-insensitively
    def find_col_by_names(names):
        for name in names:
            for col in df.columns:
                if col.strip().lower() == str(name).strip().lower():
                    return col
        return None

    # Find date column
    date_col = find_col_by_names([cfg.date_col, "Date", "DATE", "date"])
    if date_col is None:
        # try to detect any column that can be parsed as datetime
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col].dropna().iloc[:5], errors="coerce")
                if parsed.notna().any():
                    date_col = col
                    break
            except Exception:
                continue
    if date_col is None:
        raise ValueError(f"Date column '{cfg.date_col}' not found. Available: {df.columns.tolist()}")
    cfg.date_col = date_col

    # Find price column
    price_col = find_col_by_names([cfg.price_col, "Adj Close", "Adj_Close", "Close", "close"])
    if price_col is None:
        # fallback: choose first numeric column that's not the date
        for col in df.columns:
            if col == cfg.date_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                price_col = col
                break
    if price_col is None:
        raise ValueError(f"Price column '{cfg.price_col}' not found. Available: {df.columns.tolist()}")
    cfg.price_col = price_col

    # Parse dates, drop invalid, sort, deduplicate
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col]).sort_values(cfg.date_col).reset_index(drop=True)
    df = df.drop_duplicates(subset=[cfg.date_col])

    if cfg.start_date:
        df = df[df[cfg.date_col] >= pd.to_datetime(cfg.start_date)].copy()
    return df

# ====================== Returns, signals, backtest ======================
def daily_returns_from_prices(price: pd.Series) -> pd.Series:
    return price.pct_change().dropna()

def total_return_rolling(ret: pd.Series, window_days: int) -> pd.Series:
    return (1 + ret).rolling(window_days).apply(np.prod, raw=True) - 1.0

def compute_signal(index: pd.DatetimeIndex, trN: pd.Series, cfg: Config) -> pd.Series:
    sig = pd.Series(index=index, dtype=float)
    state = 1.0 if cfg.start_invested else 0.0
    last_change_day = None
    for dt in index:
        val = trN.get(dt, np.nan)
        if np.isfinite(val):
            can_change = True
            if cfg.min_hold_days and last_change_day is not None:
                can_change = (dt - last_change_day).days >= cfg.min_hold_days
            if can_change:
                if val > cfg.sell_th + cfg.hysteresis:
                    if state != 0.0:
                        state = 0.0
                        last_change_day = dt
                elif val <= cfg.buy_th - cfg.hysteresis:
                    if state != 1.0:
                        state = 1.0
                        last_change_day = dt
        sig.loc[dt] = state
    return sig

def apply_tcost(signal: pd.Series, tcost_bps=2.0) -> pd.Series:
    delta = signal.diff().abs().fillna(0.0)
    return -(tcost_bps / 10000.0) * delta

def fetch_tbill_daily(series="TB3MS", start="1980-01-01"):
    from pandas_datareader import data as pdr
    tb = pdr.DataReader(series, "fred", start=start)  # monthly, % p.a.
    tb = tb.rename(columns={series: "ann_pct"}).dropna()
    # dzienny (kalendarzowy) r_f ~ (1+ann)^(1/365)-1
    tb["ann_dec"] = tb["ann_pct"] / 100.0
    tb["rf_daily"] = (1.0 + tb["ann_dec"])**(1/365.0) - 1.0
    # dobowo z forward-fill
    daily = tb["rf_daily"].resample("D").ffill()
    return daily


def backtest_df(df: pd.DataFrame, cfg: Config):
    px = df[cfg.price_col].astype(float)
    idx = pd.DatetimeIndex(df[cfg.date_col])
    rets = daily_returns_from_prices(px)      # długość N-1 po dropna
    rets.index = idx[1:]                      # dopasuj indeks do zwrotów


    trN = total_return_rolling(rets, cfg.lookback_days)
    sig_raw = compute_signal(rets.index, trN, cfg)
    sig = sig_raw.shift(cfg.signal_lag_days).reindex(rets.index).ffill()
    if np.isnan(sig.iloc[0]):
        sig.iloc[0] = 1.0 if cfg.start_invested else 0.0

    tcost = apply_tcost(sig, cfg.tcost_bps)  # ujemne koszty na zmianę pozycji

    # gotówka: 0% lub T-Bill
    if getattr(cfg, "cash_when_flat", "ZERO").upper() == "TBILL":
        rf_daily = fetch_tbill_daily(getattr(cfg, "tbill_series", "TB3MS"),
                                    start=str(rets.index[0].date()))
        rf_daily = rf_daily.reindex(rets.index, method="ffill").fillna(0.0)
    else:
        rf_daily = rets*0.0

    # zwrot strategii: 1*SPX gdy sig=1, inaczej r_f; minus koszty
    strat_ret = (sig * rets) + ((1.0 - sig) * rf_daily) + tcost
    bench_ret = rets  # Buy&Hold


    strat_eq = (1 + strat_ret).cumprod()
    bench_eq = (1 + bench_ret).cumprod()

    return {
        "rets_asset": bench_ret,
        "rets_strategy": strat_ret,
        "position": sig,
        "trailing_return": trN,
        "equity_strategy": strat_eq,
        "equity_bench": bench_eq,
    }

def perf_stats(daily: pd.Series, periods_per_year=252):
    r = daily.dropna()
    n = len(r)
    if n == 0:
        return {"error": "no returns"}
    cum = (1 + r).prod()
    cagr = cum**(periods_per_year / n) - 1
    vol = r.std() * (periods_per_year**0.5)
    sharpe = np.nan if vol == 0 else (r.mean() * periods_per_year) / vol
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax()) - 1
    calmar = np.nan if dd.min() == 0 else cagr / abs(dd.min())
    return {
        "Days": n,
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe(0%)": sharpe,
        "MaxDD": dd.min(),
        "Calmar": calmar,
        "FinalMultiple": eq.iloc[-1],
    }

def save_plots(eq: pd.Series, bench: pd.Series, out_dir: str = "output"):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    eq.plot(title="Strategy Equity (Growth of $1)")
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "equity_strategy.png"))
    plt.close()

    plt.figure()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    dd.plot(title="Strategy Drawdown")
    plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drawdown_strategy.png"))
    plt.close()

    plt.figure()
    bench.plot(title="Buy & Hold Equity (Growth of $1)")
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "equity_bench.png"))
    plt.close()

    plt.figure()
    peakb = bench.cummax()
    ddb = (bench / peakb) - 1.0
    ddb.plot(title="Buy & Hold Drawdown")
    plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drawdown_bench.png"))
    plt.close()
