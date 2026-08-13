"""
Runs the Step 4 recommendation engine across a user-selected portfolio (as
opposed to composite_score.compute_composite_signal, which scores ONE
stock at a time, and dashboard_routes.refresh_dashboard, which scores the
full 50-stock universe). This is what /optimize calls so its returned
signals are the fused ML+leading+lagging+risk output, not just the old
lagging-only trading_signals.generate_signals() result.
"""

from typing import Optional

import pandas as pd

from data.fetch_data import fetch_ohlcv_data, fetch_india_vix
from signals.composite_score import compute_composite_signal, SIGNAL_TO_NUM


def get_composite_signals_for_portfolio(
    valid_stocks: list,
    valid_tickers: list,
    benchmark_returns: pd.Series,
    model,
    vix_series: Optional[pd.Series] = None,
) -> list:
    """
    valid_stocks  : e.g. ["TCS", "INFY"]           (plain symbols)
    valid_tickers : e.g. ["TCS.NS", "INFY.NS"]      (same order, .NS suffixed)

    Returns a list of composite_score result dicts, one per stock that had
    enough history to score (stocks without enough history are silently
    skipped, matching the existing generate_signals() failure behaviour).
    """
    if vix_series is None:
        vix_series = fetch_india_vix()

    ohlcv_by_ticker, valid_ohlcv_tickers, failed = fetch_ohlcv_data(valid_tickers, period="1y")

    results = []
    for stock, ticker in zip(valid_stocks, valid_tickers):
        if ticker not in ohlcv_by_ticker:
            continue  # not enough OHLCV history -- same stocks fetch_data() would also have skipped
        ohlcv_df = ohlcv_by_ticker[ticker]
        stock_returns = ohlcv_df["Close"].pct_change().dropna()

        result = compute_composite_signal(
            ticker=stock,
            ohlcv_df=ohlcv_df,
            stock_returns=stock_returns,
            benchmark_returns=benchmark_returns,
            model=model,
            vix_series=vix_series,
        )
        if result is not None:
            results.append(result)
    return results


def composite_portfolio_signal(composite_results: list) -> str:
    """
    Portfolio-level rollup, same 40%-threshold majority-share rule as the
    existing trading_signals.generate_portfolio_signal(), but applied to
    the FUSED per-stock signals instead of lagging-only ones.
    """
    if not composite_results:
        return "HOLD"
    total = len(composite_results)
    buy_count = sum(1 for r in composite_results if r["signal"] == "BUY")
    sell_count = sum(1 for r in composite_results if r["signal"] == "SELL")

    if buy_count / total >= 0.4:
        return "BUY"
    if sell_count / total >= 0.4:
        return "SELL"
    return "HOLD"
