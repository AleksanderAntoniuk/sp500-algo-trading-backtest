
"""
run_backtest.py — DAILY-ONLY backtest runner
- Central config: see DEFAULT_CONFIG below or edit config/config.json
- Downloads data (Yahoo SPY by default) or reads CSV
- Saves CSV results and PNG charts into ./data/
"""
import os, json, argparse, pprint
import pandas as pd
from backtest_core import Config, load_data, backtest_df, perf_stats, save_plots

# ======= ALL KEY VARIABLES HERE (edit or use JSON) =======
DEFAULT_CONFIG = Config(
    source="yahoo_spy",     # yahoo_spy | yahoo_gspc | stooq_spx | fred_sp500 | csv
    csv_path="",            # path to your CSV if source="csv"
    price_col="Adj Close",
    date_col="Date",
    start_date="1993-02-01",

    lookback_days=252,
    buy_th=-0.017,          # 30th percentile (≈ −1.7%)
    sell_th=0.111,          # median (≈ +11.1%)
    hysteresis=0.0,
    signal_lag_days=1,
    min_hold_days=0,

    tcost_bps=2.0,
    start_invested=True,
)

def load_json_config(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    return {}

def merge_config(cfg: Config, overrides: dict) -> Config:
    for k,v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.json", help="Path to JSON config (optional)")
    # Minimal CLI overrides kept for convenience
    ap.add_argument("--buy_th", type=float)
    ap.add_argument("--sell_th", type=float)
    ap.add_argument("--lookback_days", type=int)
    ap.add_argument("--hysteresis", type=float)
    ap.add_argument("--tcost_bps", type=float)
    ap.add_argument("--min_hold_days", type=int)
    ap.add_argument("--signal_lag_days", type=int)
    ap.add_argument("--source")
    ap.add_argument("--csv_path")
    ap.add_argument("--start_date")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    j = load_json_config(args.config)
    cfg = merge_config(cfg, j)
    cli = {k:v for k,v in vars(args).items() if (k!="config" and v is not None)}
    cfg = merge_config(cfg, cli)

    print("=== ACTIVE CONFIG (DAILY) ===")
    pprint.pprint(cfg.__dict__)
    print("=============================\n")

    # Data & backtest
    df = load_data(cfg)
    out = backtest_df(df, cfg)

    # Stats (daily → 252)
    s_stats = perf_stats(out["rets_strategy"], periods_per_year=252)
    b_stats = perf_stats(out["rets_asset"], periods_per_year=252)

    print("Strategy stats:")
    for k,v in s_stats.items():
        print(f"{k:>14}: {v:.4%}" if isinstance(v,float) else f"{k:>14}: {v}")
    print("\nBuy&Hold stats:")
    for k,v in b_stats.items():
        print(f"{k:>14}: {v:.4%}" if isinstance(v,float) else f"{k:>14}: {v}")

    # Save outputs
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data","backtest_results.csv")
    pd.DataFrame({
        "ret_asset": out["rets_asset"],
        "ret_strategy": out["rets_strategy"],
        "position": out["position"],
        "trailing_return": out["trailing_return"],
    }).to_csv(csv_path)
    print(f"\nSaved results CSV to: {csv_path}")

    save_plots(out["equity_strategy"], out["equity_bench"], out_dir="data")
    print("Saved charts to: data/equity_strategy.png, data/drawdown_strategy.png, data/equity_bench.png, data/drawdown_bench.png")

if __name__ == "__main__":
    main()
