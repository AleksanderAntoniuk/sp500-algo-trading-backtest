
"""
optimize_thresholds.py — DAILY-ONLY optimizer for buy/sell thresholds and lookback
- Grid search (in-sample) or walk-forward (OOS)
- Objective: Sharpe (default) / CAGR / Calmar
- Saves best parameters and summary to data/optimizer_summary.json
"""
import os, json, argparse, numpy as np, pandas as pd
from backtest_core import Config, load_data, backtest_df, perf_stats

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

def objective(stats: dict, metric: str) -> float:
    if "error" in stats: return -1e9
    m = metric.lower()
    if m == "sharpe": return stats.get("Sharpe(0%)", float("-inf"))
    if m == "cagr":   return stats.get("CAGR",       float("-inf"))
    if m == "calmar": return stats.get("Calmar",     float("-inf"))
    if m == "final":  return stats.get("FinalMultiple", float("-inf"))
    if m == "negmaxdd": return -abs(stats.get("MaxDD", 0.0))  # im większy (mniej ujemny), tym lepiej
    raise ValueError("metric must be sharpe|cagr|calmar|final|negmaxdd")


def time_splits(index: pd.DatetimeIndex, n_splits=4, min_train_years=5, test_years=3):
    dates = pd.Series(index).sort_values().unique()
    start = dates[0]
    out = []
    for i in range(n_splits):
        train_end = start + pd.DateOffset(years=min_train_years + i*test_years) - pd.DateOffset(days=1)
        test_end  = train_end + pd.DateOffset(years=test_years)
        if test_end > dates[-1]: break
        out.append((start, train_end, test_end))
    return out

def run_once(df, cfg: Config):
    o = backtest_df(df, cfg)
    return perf_stats(o["rets_strategy"], periods_per_year=252)

def grid_search(df, base: Config, buy_grid, sell_grid, lookbacks, metric: str):
    best = None
    for lb in lookbacks:
        for b in buy_grid:
            for s in sell_grid:
                if b >= s: continue
                cfg = Config(**base.__dict__)
                cfg.lookback_days = lb; cfg.buy_th = b; cfg.sell_th = s
                stats = run_once(df, cfg)
                score = objective(stats, metric)
                if (best is None) or (score > best[0]):
                    best = (score, (b,s,lb), stats)
    return best

def walk_forward(df, base: Config, buy_grid, sell_grid, lookbacks, metric: str,
                 n_splits=4, min_train_years=5, test_years=3):
    idx = pd.DatetimeIndex(df[base.date_col])
    splits = time_splits(idx, n_splits, min_train_years, test_years)
    results = []
    for (train_start, train_end, test_end) in splits:
        train = df[(df[base.date_col] >= train_start) & (df[base.date_col] <= train_end)]
        test  = df[(df[base.date_col] >  train_end) & (df[base.date_col] <= test_end)]
        if len(train) < 50 or len(test) < 20: continue
        score, (b,s,lb), _ = grid_search(train, base, buy_grid, sell_grid, lookbacks, metric)
        # evaluate on test
        cfg = Config(**base.__dict__); cfg.buy_th, cfg.sell_th, cfg.lookback_days = b, s, lb
        stats = run_once(test, cfg)
        results.append((b,s,lb,stats))
    if not results: return None
    # aggregate by mean of objective
    by_key = {}
    for b,s,lb,st in results:
        key = (b,s,lb); val = objective(st, metric)
        by_key.setdefault(key, []).append(val)
    best_key = max(by_key.items(), key=lambda kv: np.mean(kv[1]))[0]
    # compute mean stats for display
    sel = [st for b,s,lb,st in results if (b,s,lb)==best_key]
    keys = sel[0].keys()
    avg_stats = {k: float(np.nanmean([d.get(k, np.nan) for d in sel])) for k in keys}
    return best_key, avg_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--metric", choices=["sharpe","cagr","calmar"], default="sharpe")
    ap.add_argument("--mode", choices=["grid","walk"], default="grid")
    ap.add_argument("--buy_min", type=float, default=-0.60)
    ap.add_argument("--buy_max", type=float, default=0.10)
    ap.add_argument("--sell_min", type=float, default=-0.10)
    ap.add_argument("--sell_max", type=float, default=0.60)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--lookbacks", type=str, default="126,189,252,378,504")
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--min_train_years", type=int, default=5)
    ap.add_argument("--test_years", type=int, default=3)
    args = ap.parse_args()

    base = Config()
    if os.path.exists(args.config):
        with open(args.config,"r",encoding="utf-8") as f:
            base = merge_config(base, json.load(f))

    import numpy as np
    df = load_data(base)

    buy_grid  = np.arange(args.buy_min,  args.buy_max + 1e-9, args.step)
    sell_grid = np.arange(args.sell_min, args.sell_max + 1e-9, args.step)
    lookbacks = [int(x) for x in args.lookbacks.split(",") if x.strip()]

    os.makedirs("data", exist_ok=True)
    summary_path = os.path.join("data","optimizer_summary.json")

    if args.mode == "grid":
        score, (b,s,lb), stats = grid_search(df, base, buy_grid, sell_grid, lookbacks, args.metric)
        print("Best (in-sample):")
        print(f"  buy_th={b:.3%}, sell_th={s:.3%}, lookback_days={lb}")
        for k,v in stats.items():
            print(f"{k:>14}: {v:.4%}" if isinstance(v,float) else f"{k:>14}: {v}")
        with open(summary_path,"w",encoding="utf-8") as f:
            json.dump({"mode":"grid","metric":args.metric,"buy_th":b,"sell_th":s,"lookback_days":lb,"stats":stats}, f, indent=2)
        print(f"\nSaved summary to: {summary_path}")
    else:
        result = walk_forward(df, base, buy_grid, sell_grid, lookbacks, args.metric,
                              n_splits=args.n_splits, min_train_years=args.min_train_years, test_years=args.test_years)
        if result is None:
            print("Walk-forward: not enough data.")
            return
        (b,s,lb), avg_stats = result
        print("Best by walk-forward (avg OOS):")
        print(f"  buy_th={b:.3%}, sell_th={s:.3%}, lookback_days={lb}")
        for k,v in avg_stats.items():
            print(f"{k:>14}: {v:.4%}" if isinstance(v,float) else f"{k:>14}: {v}")
        with open(summary_path,"w",encoding="utf-8") as f:
            json.dump({"mode":"walk","metric":args.metric,"buy_th":b,"sell_th":s,"lookback_days":lb,"avg_stats":avg_stats}, f, indent=2)
        print(f"\nSaved summary to: {summary_path}")

if __name__ == "__main__":
    main()
