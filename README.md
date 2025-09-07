# S&P 500 Mean-Reversion — Daily Backtest

This repository contains a simple, daily-frequency, long-only mean-reversion backtest for the S&P 500 (default uses SPY). The project focuses on reproducible research: configurable JSON input, clear backtest logic, and output CSV + charts.

Note: the optimizer has been removed from this codebase. This version only provides a clean, documented backtest runner.

## Features
- Daily mean-reversion rule using trailing N-day total return.
- Next-day execution (no look-ahead).
- Optional T‑Bill cash yield when flat.
- Configurable via `config/config.json`.
- Outputs: daily results CSV and PNG charts in `data/`.

## Quick start
1. Create and activate a virtual environment, then install requirements:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run a backtest (from repo root):
```bash
python main.py backtest
```

3. Pass JSON or CLI overrides:
```bash
python main.py backtest -- --config config/config.json --buy_th -0.02 --sell_th 0.11
```
Everything after `--` is forwarded to the underlying runner.

## Configuration
Edit `config/config.json` to change data source and strategy parameters. Key fields:
- source: `yahoo_spy` | `yahoo_gspc` | `stooq_spx` | `fred_sp500` | `csv`
- csv_path: path when `source: "csv"`
- start_date, date_col, price_col
- lookback_days, buy_th, sell_th, hysteresis
- signal_lag_days, min_hold_days
- tcost_bps, start_invested
- cash_when_flat (e.g. "TBILL") and tbill_series (FRED code) — optional

Thresholds are decimals (e.g., −2% → `-0.02`).

## Outputs
- data/backtest_results.csv — daily asset & strategy returns, position, trailing return
- data/equity_strategy.png, data/drawdown_strategy.png
- data/equity_bench.png, data/drawdown_bench.png

## Notes
- This project is educational — not production trading code or investment advice.
- If you want to restore or reintroduce an optimizer, keep it in a separate tool/script and ensure careful limits on grid sizes.

## License
MIT — see LICENSE.
