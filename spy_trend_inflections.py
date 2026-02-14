"""
SPY Trend Inflection Point Detector

Identifies dates when SPY transitions between uptrends and downtrends
based on swing (pivot) highs and lows on the daily chart.

Uptrend:   Higher highs AND higher lows (on pivot points)
Downtrend: Lower highs AND lower lows (on pivot points)
Inflection: The date when the regime flips.

Usage:
    # Auto-download from Yahoo Finance (requires yfinance):
    python spy_trend_inflections.py

    # From a local CSV file (must have Date, High, Low columns):
    python spy_trend_inflections.py --csv SPY_daily.csv

    # Adjust pivot lookback window (default k=5):
    python spy_trend_inflections.py --k 3

    # Custom date range:
    python spy_trend_inflections.py --start 2000-01-01 --end 2025-01-01
"""

import argparse
import pandas as pd
import numpy as np


def find_pivots(df: pd.DataFrame, k: int = 5):
    """
    Pivot high: High is the strict maximum in a (2k+1) window centered on the bar.
    Pivot low:  Low is the strict minimum in a (2k+1) window centered on the bar.
    """
    highs = df["High"].values
    lows = df["Low"].values

    pivot_high = np.zeros(len(df), dtype=bool)
    pivot_low = np.zeros(len(df), dtype=bool)

    for i in range(k, len(df) - k):
        window_high = highs[i - k : i + k + 1]
        window_low = lows[i - k : i + k + 1]
        if highs[i] == window_high.max() and window_high.argmax() == k:
            pivot_high[i] = True
        if lows[i] == window_low.min() and window_low.argmin() == k:
            pivot_low[i] = True

    pivots = df.loc[pivot_high | pivot_low, ["High", "Low"]].copy()
    pivots["pivot_high"] = pivot_high[pivot_high | pivot_low]
    pivots["pivot_low"] = pivot_low[pivot_high | pivot_low]
    return pivots


def trend_inflections_from_pivots(pivots: pd.DataFrame):
    """
    Walk the pivot sequence and declare:
      Uptrend   when the last two pivot highs are rising AND last two pivot lows are rising.
      Downtrend when the last two pivot highs are falling AND last two pivot lows are falling.

    Returns a DataFrame of inflection points (date, regime).
    """
    last_two_highs = []
    last_two_lows = []

    regime = None  # "uptrend", "downtrend", or None
    flips = []

    for dt, row in pivots.iterrows():
        if row["pivot_high"]:
            last_two_highs.append((dt, row["High"]))
            last_two_highs = last_two_highs[-2:]
        if row["pivot_low"]:
            last_two_lows.append((dt, row["Low"]))
            last_two_lows = last_two_lows[-2:]

        if len(last_two_highs) == 2 and len(last_two_lows) == 2:
            h1, h2 = last_two_highs[0][1], last_two_highs[1][1]
            l1, l2 = last_two_lows[0][1], last_two_lows[1][1]

            is_up = (h2 > h1) and (l2 > l1)
            is_down = (h2 < h1) and (l2 < l1)

            new_regime = regime
            if is_up:
                new_regime = "uptrend"
            elif is_down:
                new_regime = "downtrend"

            if new_regime != regime and new_regime is not None:
                flips.append({"date": dt, "regime": new_regime})
                regime = new_regime

    return pd.DataFrame(flips)


def load_from_csv(path: str) -> pd.DataFrame:
    """Load OHLC data from a CSV file. Expects Date, High, Low columns at minimum."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    for col in ("High", "Low"):
        if col not in df.columns:
            raise ValueError(f"CSV must have a '{col}' column. Found: {list(df.columns)}")
    df = df.dropna(subset=["High", "Low"]).sort_index()
    return df


def load_from_yfinance(ticker: str, start: str, end=None) -> pd.DataFrame:
    """Download daily OHLC from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit(
            "yfinance not installed. Either:\n"
            "  pip install yfinance\n"
            "or provide a CSV with --csv (needs Date, High, Low columns)"
        )

    print(f"Downloading {ticker} daily data from {start} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def main():
    parser = argparse.ArgumentParser(description="SPY trend inflection detector")
    parser.add_argument("--csv", type=str, help="Path to CSV with Date,High,Low columns")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker to download (default: SPY)")
    parser.add_argument("--start", type=str, default="1993-01-01", help="Start date (default: 1993-01-01)")
    parser.add_argument("--end", type=str, default=None, help="End date (default: today)")
    parser.add_argument("--k", type=int, default=5, help="Pivot lookback window (default: 5)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    if args.csv:
        print(f"Loading data from {args.csv} ...")
        df = load_from_csv(args.csv)
    else:
        df = load_from_yfinance(args.ticker, args.start, args.end)

    print(f"Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")

    pivots = find_pivots(df, k=args.k)
    print(f"Found {pivots['pivot_high'].sum()} pivot highs and {pivots['pivot_low'].sum()} pivot lows (k={args.k})")

    flips = trend_inflections_from_pivots(pivots)

    print(f"\n{'='*60}")
    print(f"Found {len(flips)} trend regime inflection points (k={args.k})")
    print(f"{'='*60}\n")

    if not flips.empty:
        flips_display = flips.copy()
        flips_display["date"] = pd.to_datetime(flips_display["date"]).dt.strftime("%Y-%m-%d")
        print(flips_display.to_string(index=False))
    else:
        print("No inflection points found.")

    out = args.output or f"{args.ticker}_trend_inflections_k{args.k}.csv"
    flips.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
