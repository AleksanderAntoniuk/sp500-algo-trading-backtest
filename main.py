"""
Main launcher for the S&P 500 mean‑reversion project (daily‑only).

This launcher adds the `src/` directory to sys.path and forwards the `backtest`
command to `src/run_backtest.py`. All detailed backtest logic lives in src/.
"""

import sys, argparse
from pathlib import Path
import importlib

def _add_src_to_path():
    # ensure `src` is importable regardless of current working directory
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

def _call_module_main(module_name: str, forwarded_args: list[str]):
    """
    Import <module_name> from /src and call its main(), forwarding CLI args.

    This function temporarily replaces sys.argv so the module sees CLI args as
    if called directly (useful for modules that use argparse).
    """
    mod = importlib.import_module(module_name)
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

    # backtest only (optimizer removed)
    p_back = sub.add_parser("backtest", help="Run daily backtest (wrapper for src/run_backtest.py)")
    p_back.add_argument("--config", default="config/config.json")
    # everything after `--` is forwarded directly to the underlying runner
    p_back.add_argument("extra", nargs=argparse.REMAINDER, help="extra flags forwarded to run_backtest")

    args = parser.parse_args()

    if args.cmd == "backtest":
        forwarded = ["--config", args.config] + args.extra
        _call_module_main("run_backtest", forwarded)
    else:
        parser.error("Unknown command. Use 'backtest'.")

if __name__ == "__main__":
    main()
