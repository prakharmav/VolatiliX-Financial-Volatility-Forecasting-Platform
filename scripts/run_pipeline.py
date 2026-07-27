from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.pipeline import run_pipeline


def main():
    p = argparse.ArgumentParser(description="Stock volatility forecasting pipeline")
    p.add_argument("--ticker", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--no-lstm", action="store_true")
    p.add_argument("--no-prophet", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    if args.ticker:
        cfg["data"]["ticker"] = args.ticker
    if args.start:
        cfg["data"]["start_date"] = args.start
    if args.end:
        cfg["data"]["end_date"] = args.end

    run_pipeline(cfg, include_lstm=not args.no_lstm, include_prophet=not args.no_prophet)


if __name__ == "__main__":
    main()
