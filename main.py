
"""
Main launcher for the S&P 500 mean‑reversion project (daily‑only).
Place this file in the **repository root** (same level as /src and /config).

Usage:
    python main.py backtest [--config config/config.json] [other run_backtest flags...]
    python main.py optimize [--config config/config.json] [other optimize flags...]

This script forwards all extra flags to the underlying modules, so you don't have to cd into /src.
"""

import sys, argparse, importlib, os
from pathlib import Path

def _add_src_to_path():
    root = Path(__file__).resolve().parent
    src = root / "src"
    sys.path.insert(0, str(src))

def _call_module_main(module_name: str, forwarded_args: list[str]):
    """
    Import <module_name> from /src and call its main(), forwarding CLI args.
    """
    import importlib
    mod = importlib.import_module(module_name)
    # Emulate CLI call for that module
    old_argv = sys.argv[:]
    try:
        sys.argv = [mod.__file__] + forwarded_args
        mod.main()
    finally:
        sys.argv = old_argv

def main():
    _add_src_to_path()
    parser = argparse.ArgumentParser(prog="main.py", description="Project launcher (daily-only)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # backtest
    p_back = sub.add_parser("backtest", help="Run daily backtest (wrapper for src/run_backtest.py)")
    p_back.add_argument("--config", default="config/config.json")
    p_back.add_argument("extra", nargs=argparse.REMAINDER, help="extra flags forwarded to run_backtest")

    # optimize
    p_opt = sub.add_parser("optimize", help="Run optimizer (wrapper for src/optimize_thresholds.py)")
    p_opt.add_argument("--config", default="config/config.json")
    p_opt.add_argument("extra", nargs=argparse.REMAINDER, help="extra flags forwarded to optimize_thresholds")

    args = parser.parse_args()

    if args.cmd == "backtest":
        forwarded = ["--config", args.config] + args.extra
        _call_module_main("run_backtest", forwarded)
    elif args.cmd == "optimize":
        forwarded = ["--config", args.config] + args.extra
        _call_module_main("optimize_thresholds", forwarded)
    else:
        parser.error("Unknown command. Use 'backtest' or 'optimize'.")

if __name__ == "__main__":
    main()
