"""
GET  /dashboard/stocks           -> all 50 stocks, color-coded, sorted by score
GET  /dashboard/stocks/<ticker>  -> full breakdown for one stock (for a dashboard block's detail view)
POST /dashboard/refresh          -> manually re-score everything now

This is what makes the platform show information on login instead of only
after a user manually searches for a ticker. Scoring is synchronous on
first request (there's no background scheduler yet -- that's Step 8), so
the very first call after a server restart will be slow (fetches + scores
all 50 stocks); every call after that serves the cache.
"""

from datetime import datetime, timezone

from flask import Blueprint, jsonify

from data.fetch_data import fetch_ohlcv_data, fetch_benchmark_data, fetch_india_vix
from data.stock_sectors import STOCK_SECTORS
from signals.composite_score import compute_composite_signal
from optimization.model_loader import get_ensemble_model

dashboard_bp = Blueprint("dashboard", __name__)

UNIVERSE = list(STOCK_SECTORS.keys())  # NOTE: this list and app.py's NIFTY_50_STOCKS are two
                                        # separately-maintained lists of (almost) the same 50
                                        # tickers -- worth unifying into one source of truth
                                        # in a follow-up cleanup, they're not guaranteed identical.

DASHBOARD_CACHE: dict = {}
LAST_REFRESHED_AT = None
MIN_HISTORY_DAYS = 200  # ~1 trading year, needed for the 252-day beta window in live_features


def refresh_dashboard():
    """Re-scores every stock in UNIVERSE and replaces DASHBOARD_CACHE."""
    global DASHBOARD_CACHE, LAST_REFRESHED_AT

    tickers = [f"{t}.NS" for t in UNIVERSE]
    ohlcv_by_ticker, valid_tickers, failed_tickers = fetch_ohlcv_data(tickers, period="1y")

    if not valid_tickers:
        raise RuntimeError("No market data available for any dashboard stock -- check network/tickers.")

    benchmark_prices = fetch_benchmark_data()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    vix_series = fetch_india_vix()

    model = get_ensemble_model()

    new_cache = {}
    for ticker in valid_tickers:
        base_ticker = ticker.replace(".NS", "")
        ohlcv_df = ohlcv_by_ticker[ticker]
        if len(ohlcv_df) < MIN_HISTORY_DAYS:
            continue  # not enough history for a reliable beta/volatility calc yet

        stock_returns = ohlcv_df["Close"].pct_change().dropna()

        try:
            result = compute_composite_signal(
                ticker=base_ticker,
                ohlcv_df=ohlcv_df,
                stock_returns=stock_returns,
                benchmark_returns=benchmark_returns,
                model=model,
                vix_series=vix_series,
            )
        except Exception as exc:
            print(f"Warning: failed to score {base_ticker}: {exc}")
            continue

        if result is not None:
            new_cache[base_ticker] = result

    DASHBOARD_CACHE = new_cache
    LAST_REFRESHED_AT = datetime.now(timezone.utc).isoformat()
    return DASHBOARD_CACHE


@dashboard_bp.route("/dashboard/stocks", methods=["GET"])
def get_dashboard():
    if not DASHBOARD_CACHE:
        try:
            refresh_dashboard()
        except Exception as exc:
            return jsonify({"error": f"Could not build dashboard: {exc}"}), 503

    stocks = list(DASHBOARD_CACHE.values())
    stocks.sort(key=lambda s: s["score"], reverse=True)  # BUY-leaning stocks surface first

    return jsonify({
        "last_refreshed_at": LAST_REFRESHED_AT,
        "count": len(stocks),
        "stocks": stocks,
    })


@dashboard_bp.route("/dashboard/stocks/<ticker>", methods=["GET"])
def get_dashboard_stock_detail(ticker):
    entry = DASHBOARD_CACHE.get(ticker.upper())
    if entry is None:
        return jsonify({"error": f"No cached data for {ticker}. Try GET /dashboard/stocks first."}), 404
    return jsonify(entry)


@dashboard_bp.route("/dashboard/refresh", methods=["POST"])
def post_dashboard_refresh():
    try:
        refresh_dashboard()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"status": "ok", "last_refreshed_at": LAST_REFRESHED_AT, "count": len(DASHBOARD_CACHE)})
