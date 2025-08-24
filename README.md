# S&P 500 Mean-Reversion — Daily-Only Backtest & Optimizer (v3)

> **Note to recruiters (e.g., Goldman Sachs):** this repository is an **educational exercise** showcasing data handling, backtesting design, and research workflow. It is **not** production trading code and **not** investment advice.

## What this project does
- Runs **daily** backtests of a **long-only** mean-reversion strategy on the **S&P 500**.
- Signals are evaluated **every trading day**, orders execute **next day** (no look-ahead).
- Strategy is compared against a fixed **Buy & Hold** benchmark.
- Configuration is **JSON-first** (`config/config.json`), with a minimal launcher (`main.py`).
- Optional **cash yield when flat**: earn **T-Bill** return instead of 0% (configurable).
- Includes a simple **optimizer** (grid / walk-forward) for parameter sweeps.

## Strategy (short version)
- Compute the **trailing N-day** total return: ∏(1 + r_d) − 1 over the last `lookback_days`.
- **BUY → 100%** when trailing return ≤ `buy_th` (e.g., 30th percentile ≈ −1.7% → `-0.017`).
- **SELL → 0%** when trailing return > `sell_th` (e.g., median ≈ +11.1% → `0.111`).
- Optional **hysteresis** (buffer) and **min_hold_days** to reduce whipsaw.
- **No leverage**, no shorting. When flat, cash can earn **T-Bill** (configurable).

> Thresholds in the config are **decimals**, not percent. Example: −2% → `-0.02`, 11.1% → `0.111`.

## Project layout
```
sp500-meanreversion-backtest-v3/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── config/
│   └── config.json               # primary place to customize the strategy
├── src/
│   ├── backtest_core.py          # daily-only backtest library (no leverage)
│   ├── run_backtest.py           # backtest runner
│   └── optimize_thresholds.py    # parameter optimizer (grid / walk-forward)
├── main.py                       # simple launcher so you don't cd into /src
└── data/
    └── .gitkeep
```

## Quick start
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

Run a backtest using your JSON config:
```bash
python main.py backtest
```

(Advanced) You can also pass overrides to the runner; prefer editing JSON, but if needed:
```bash
# Everything after `--` is forwarded to the underlying runner
python main.py backtest -- --buy_th -0.02 --sell_th 0.111 --lookback_days 252
```

Run the optimizer:
```bash
# Sharpe on an in-sample grid
python main.py optimize -- --metric sharpe --mode grid --step 0.01 --lookbacks 126,189,252,378,504

# Walk-forward (OOS) under Calmar
python main.py optimize -- --metric calmar --mode walk --n_splits 4 --test_years 3 --min_train_years 5
```

## Configuration (JSON)
All key variables live in `config/config.json`. Example:
```json
{
  "source": "yahoo_spy",
  "csv_path": "",
  "price_col": "Adj Close",
  "date_col": "Date",
  "start_date": "1993-02-01",

  "lookback_days": 252,
  "buy_th": -0.017,
  "sell_th": 0.111,
  "hysteresis": 0.005,
  "signal_lag_days": 1,
  "min_hold_days": 3,

  "tcost_bps": 2.0,
  "start_invested": true,

  "cash_when_flat": "TBILL",
  "tbill_series": "TB3MS"
}
```

**Fields**
- `source`: `yahoo_spy` (Adj Close ≈ total return), `yahoo_gspc`, `stooq_spx`, `fred_sp500`, or `csv`.
- `start_date`: earliest date to include. The strategy needs `lookback_days` of history to produce the first signal; if you want signals from Day 1, set `start_date` at least `lookback_days` earlier than your analysis window.
- `lookback_days`: trailing window (e.g., 252 ≈ 12 months of trading days).
- `buy_th` / `sell_th`: **decimals** (e.g., −1% = `-0.01`).
- `hysteresis`: buffer around thresholds (e.g., `0.005` = 0.5%).
- `signal_lag_days`: execution delay (default 1 trading day).
- `min_hold_days`: minimum holding time to reduce churn.
- `tcost_bps`: transaction cost per position change (basis points).
- `cash_when_flat`: earn **T-Bill** when out of the market ("TBILL") or "ZERO" for 0%.
- `tbill_series`: FRED code for the T-Bill series (monthly annualized, converted to daily internally).

## Data
- Default: **Yahoo SPY** (since 1993).  
- Alternatives: **Yahoo ^GSPC** (index price), **Stooq ^SPX**, **FRED SP500**.  
- For your own CSVs, set `source: "csv"`, then provide `csv_path`, `date_col`, and `price_col`.

## Outputs
After a run you’ll find:
- `data/backtest_results.csv` — daily asset return, strategy return, position, trailing return.
- `data/equity_strategy.png`, `data/drawdown_strategy.png`
- `data/equity_bench.png`, `data/drawdown_bench.png`

The console prints summary stats (CAGR, Vol, Sharpe(0%), MaxDD, Calmar, FinalMultiple).

## Notes & caveats
- Daily frequency means more decisions and potential whipsaw; tune `hysteresis` and `min_hold_days`.
- No shorting, no leverage. Cash can optionally earn T-Bill to better reflect reality.
- Avoid overfitting: prefer **walk-forward** for out-of-sample validation.

## License & disclaimer
- Licensed under **MIT** (see `LICENSE`).
- **For educational and recruiting purposes only**; **not** investment advice, **no** warranty of fitness for trading.
