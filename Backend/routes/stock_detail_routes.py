"""
GET /stocks/<ticker>/details

Feeds the frontend's stock-details page (Step 6): the candlestick chart data,
SMA20/SMA50 overlay lines, a separate RSI series, and the full composite
breakdown (ML probabilities + SHAP top factors, leading indicator detail,
lagging RSI) for the "why" panel next to the chart.

Reuses the dashboard cache (Step 5) when available instead of recomputing --
falls back to a fresh single-stock composite calculation for tickers outside
the 50-stock universe or before the first dashboard refresh has happened.
"""

from flask import Blueprint, jsonify

from data.fetch_data import fetch_ohlcv_data, fetch_benchmark_data, fetch_india_vix
from signals.lagging_indicators import _calculate_rsi
from signals.composite_score import compute_composite_signal
from optimization.model_loader import get_ensemble_model
import routes.dashboard_routes as dashboard_routes

stock_detail_bp = Blueprint("stock_detail", __name__)


def _to_chart_points(series, ohlcv_df=None):
    """lightweight-charts wants [{time: 'YYYY-MM-DD', value: n}, ...] for line
    series, or {time, open, high, low, close} for candlesticks."""
    points = []
    for date, value in series.dropna().items():
        points.append({"time": date.strftime("%Y-%m-%d"), "value": round(float(value), 2)})
    return points


def _to_candles(ohlcv_df):
    candles = []
    for date, row in ohlcv_df.iterrows():
        candles.append({
            "time": date.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
        })
    return candles


@stock_detail_bp.route("/stocks/<ticker>/details", methods=["GET"])
def get_stock_details(ticker):
    base_ticker = ticker.upper().replace(".NS", "")
    full_ticker = f"{base_ticker}.NS"

    ohlcv_by_ticker, valid_tickers, failed = fetch_ohlcv_data([full_ticker], period="1y")
    if full_ticker not in ohlcv_by_ticker:
        return jsonify({"error": f"No market data available for {base_ticker}"}), 404

    ohlcv_df = ohlcv_by_ticker[full_ticker]
    close = ohlcv_df["Close"]

    sma20 = close.rolling(window=20, min_periods=20).mean()
    sma50 = close.rolling(window=50, min_periods=50).mean()
    rsi = _calculate_rsi(close, period=14)

    # Prefer the cached composite result from the Step 5 dashboard (already
    # scored, no extra network/model calls) -- fall back to computing it
    # fresh for tickers outside the cached universe.
    composite = dashboard_routes.DASHBOARD_CACHE.get(base_ticker)
    if composite is None:
        try:
            stock_returns = close.pct_change().dropna()
            benchmark_returns = fetch_benchmark_data().pct_change().dropna()
            vix_series = fetch_india_vix()
            model = get_ensemble_model()
            composite = compute_composite_signal(
                ticker=base_ticker, ohlcv_df=ohlcv_df, stock_returns=stock_returns,
                benchmark_returns=benchmark_returns, model=model, vix_series=vix_series,
            )
        except Exception as exc:
            composite = None  # chart still renders even if scoring fails

    # A single marker on the most recent candle, colored by the final signal --
    # the chart-level equivalent of the dashboard block's color.
    markers = []
    if composite is not None and not ohlcv_df.empty:
        last_date = ohlcv_df.index[-1].strftime("%Y-%m-%d")
        markers.append({"time": last_date, "signal": composite["signal"]})

    return jsonify({
        "ticker": base_ticker,
        "candles": _to_candles(ohlcv_df.tail(180)),   # ~6 months of candles is plenty on screen
        "overlays": {
            "sma20": _to_chart_points(sma20.tail(180)),
            "sma50": _to_chart_points(sma50.tail(180)),
        },
        "rsi": _to_chart_points(rsi.tail(180)),
        "markers": markers,
        "composite": composite,   # full breakdown: signal, score, confidence, ML/leading/lagging detail
    })
