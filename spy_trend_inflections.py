"""
SPY Trend Inflection Point Detector

Identifies dates when SPY transitions between uptrends and downtrends
based on swing (pivot) highs and lows on the daily or weekly chart.

Uptrend:   Higher highs AND higher lows (on pivot points)
Downtrend: Lower highs AND lower lows (on pivot points)
Inflection: The date when the regime flips.

Usage:
    # Daily chart (default):
    python spy_trend_inflections.py

    # Weekly chart (less noise):
    python spy_trend_inflections.py --timeframe weekly

    # Weekly with custom pivot window:
    python spy_trend_inflections.py --timeframe weekly --k 2

    # From a local CSV file (must have Date, High, Low, Close columns):
    python spy_trend_inflections.py --csv SPY_daily.csv

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


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLC bars into weekly bars (Mon-Fri weeks)."""
    weekly = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna()
    if "Volume" in df.columns:
        weekly["Volume"] = df["Volume"].resample("W-FRI").sum()
    return weekly


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


def backtest(df: pd.DataFrame, flips: pd.DataFrame, initial_capital: float = 10000.0):
    """
    Backtest the trend-following strategy over the price data.

    Rules:
        - Start in cash.
        - On an "uptrend" signal: buy SPY at the next trading day's Close.
        - On a "downtrend" signal: sell SPY at the next trading day's Close, go to cash.

    Returns a DataFrame with daily portfolio value for both the strategy and buy-and-hold.
    """
    closes = df[["Close"]].copy()
    closes = closes.sort_index()

    signal_dates = {}
    for _, row in flips.iterrows():
        sig_date = pd.Timestamp(row["date"])
        signal_dates[sig_date] = row["regime"]

    dates = closes.index.tolist()
    position = 0.0  # shares held
    cash = initial_capital
    strategy_values = []
    pending_signal = None

    for i, dt in enumerate(dates):
        price = closes.loc[dt, "Close"]
        if isinstance(price, pd.Series):
            price = price.iloc[0]

        # Execute pending signal from yesterday at today's close
        if pending_signal == "uptrend" and position == 0:
            position = cash / price
            cash = 0.0
        elif pending_signal == "downtrend" and position > 0:
            cash = position * price
            position = 0.0
        pending_signal = None

        # Check for new signal on this date
        if dt in signal_dates:
            pending_signal = signal_dates[dt]

        portfolio_value = cash + position * price
        strategy_values.append(portfolio_value)

    # Buy-and-hold: invest full capital on day 1
    first_price = closes.iloc[0]["Close"]
    if isinstance(first_price, pd.Series):
        first_price = first_price.iloc[0]
    bh_shares = initial_capital / first_price
    bh_values = [bh_shares * (c.iloc[0] if isinstance(c, pd.Series) else c)
                 for c in closes["Close"]]

    result = pd.DataFrame({
        "date": dates,
        "strategy": strategy_values,
        "buy_and_hold": bh_values,
    })
    result.set_index("date", inplace=True)
    return result


def compute_stats(equity: pd.Series, label: str, periods_per_year: int = 252):
    """Compute key performance stats for an equity curve."""
    total_days = (equity.index[-1] - equity.index[0]).days
    years = total_days / 365.25

    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min() * 100

    ret = equity.pct_change().dropna()
    sharpe = (ret.mean() / ret.std() * np.sqrt(periods_per_year)) if ret.std() > 0 else 0

    return {
        "label": label,
        "start_value": f"${equity.iloc[0]:,.2f}",
        "end_value": f"${equity.iloc[-1]:,.2f}",
        "total_return": f"{total_return:+.1f}%",
        "cagr": f"{cagr:.1f}%",
        "max_drawdown": f"{max_dd:.1f}%",
        "sharpe_ratio": f"{sharpe:.2f}",
    }


def count_trades(flips: pd.DataFrame):
    """Count round-trip trades and win/loss info."""
    entries, exits, trades = [], [], []
    for _, row in flips.iterrows():
        if row["regime"] == "uptrend":
            entries.append(row["date"])
        elif row["regime"] == "downtrend" and entries:
            exits.append(row["date"])

    return len(entries), len(exits)


def print_backtest_report(bt: pd.DataFrame, flips: pd.DataFrame):
    """Print a human-readable backtest summary."""
    strat_stats = compute_stats(bt["strategy"], "Trend Strategy")
    bh_stats = compute_stats(bt["buy_and_hold"], "Buy & Hold SPY")

    n_entries, n_exits = count_trades(flips)

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS  ({bt.index[0].date()} to {bt.index[-1].date()})")
    print(f"{'='*60}")
    print(f"  Starting capital: $10,000")
    print()

    for stats in [strat_stats, bh_stats]:
        print(f"  --- {stats['label']} ---")
        print(f"  Final value:    {stats['end_value']}")
        print(f"  Total return:   {stats['total_return']}")
        print(f"  Annual return:  {stats['cagr']}")
        print(f"  Max drawdown:   {stats['max_drawdown']}")
        print(f"  Sharpe ratio:   {stats['sharpe_ratio']}")
        print()

    print(f"  --- Trade Summary ---")
    print(f"  Buy signals:    {n_entries}")
    print(f"  Sell signals:   {n_exits}")
    print(f"  Round trips:    {min(n_entries, n_exits)}")

    # Time in market
    in_market_days = (bt["strategy"].diff().ne(0) | (bt["strategy"] != bt["strategy"].iloc[0])).sum()
    total_days = len(bt)
    time_in = sum(1 for i in range(len(bt)) if bt["strategy"].iloc[i] != bt["strategy"].iloc[max(0,i-1)] or i == 0)

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="SPY trend inflection detector")
    parser.add_argument("--csv", type=str, help="Path to CSV with Date,High,Low,Close columns")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker to download (default: SPY)")
    parser.add_argument("--start", type=str, default="1993-01-01", help="Start date (default: 1993-01-01)")
    parser.add_argument("--end", type=str, default=None, help="End date (default: today)")
    parser.add_argument("--timeframe", type=str, default="daily", choices=["daily", "weekly"],
                        help="Timeframe for pivot detection (default: daily)")
    parser.add_argument("--k", type=int, default=None,
                        help="Pivot lookback window (default: 5 for daily, 3 for weekly)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--backtest-years", type=int, default=12, help="Number of years to backtest (default: 12)")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital for backtest (default: 10000)")
    args = parser.parse_args()

    # Default k depends on timeframe
    k = args.k if args.k is not None else (3 if args.timeframe == "weekly" else 5)

    if args.csv:
        print(f"Loading data from {args.csv} ...")
        df_daily = load_from_csv(args.csv)
    else:
        df_daily = load_from_yfinance(args.ticker, args.start, args.end)

    print(f"Loaded {len(df_daily)} daily bars ({df_daily.index[0].date()} to {df_daily.index[-1].date()})")

    # Resample to weekly if requested
    if args.timeframe == "weekly":
        df = resample_to_weekly(df_daily)
        bar_label = "weekly"
        periods_per_year = 52
        print(f"Resampled to {len(df)} weekly bars")
    else:
        df = df_daily
        bar_label = "daily"
        periods_per_year = 252

    pivots = find_pivots(df, k=k)
    print(f"Found {pivots['pivot_high'].sum()} pivot highs and {pivots['pivot_low'].sum()} pivot lows (k={k}, {bar_label})")

    flips = trend_inflections_from_pivots(pivots)

    print(f"\n{'='*60}")
    print(f"Found {len(flips)} trend inflection points (k={k}, {bar_label})")
    print(f"{'='*60}\n")

    if not flips.empty:
        flips_display = flips.copy()
        flips_display["date"] = pd.to_datetime(flips_display["date"]).dt.strftime("%Y-%m-%d")
        print(flips_display.to_string(index=False))
    else:
        print("No inflection points found.")

    out = args.output or f"{args.ticker}_trend_inflections_{bar_label}_k{k}.csv"
    flips.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    # --- Backtest (always uses daily bars for realistic execution) ---
    if args.backtest and not flips.empty:
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=args.backtest_years * 365)
        bt_df = df_daily.loc[df_daily.index >= pd.Timestamp(cutoff)]
        bt_flips = flips[pd.to_datetime(flips["date"]) >= pd.Timestamp(cutoff)]

        if bt_df.empty or bt_flips.empty:
            print("Not enough data in the backtest window.")
        else:
            bt = backtest(bt_df, bt_flips, initial_capital=args.capital)
            print_backtest_report(bt, bt_flips)

            bt_out = f"{args.ticker}_backtest_{bar_label}_{args.backtest_years}yr.csv"
            bt.to_csv(bt_out)
            print(f"Backtest equity curve saved to {bt_out}")


if __name__ == "__main__":
    main()
