"""
Leading indicators -- anticipate a price move rather than confirm one
already underway (unlike RSI / SMA-crossover / momentum in
trading_signals.py, which are all lagging).

Each price-based function takes an OHLCV DataFrame for ONE stock (as
produced by data.fetch_data.fetch_ohlcv_data) with columns
["Open", "High", "Low", "Close", "Volume"] and returns a pandas Series
aligned to the same index.
"""

import numpy as np
import pandas as pd

from optimization.ensemble import ENSEMBLE_FEATURE_ORDER, extract_ensemble_features_from_price_series
from data.stock_sectors import STOCK_SECTORS, get_sector

# Column order the trained ensemble model expects. Kept as its own name here
# (rather than importing ENSEMBLE_FEATURE_ORDER directly everywhere) so
# composite_score.py has one obvious place to import "the feature schema" from.
FEATURE_COLUMNS = ENSEMBLE_FEATURE_ORDER


def _universe_sector_exposure(ticker: str) -> float:
    """What fraction of the whole NIFTY 50 universe shares this stock's sector.

    Unlike calculate_dynamic_sector_exposure() in app.py (which measures
    exposure within a user's *selected* portfolio), the dashboard scores
    all 50 stocks independently with no portfolio context -- so exposure
    is measured against the full tracked universe instead.
    """
    if not STOCK_SECTORS:
        return 1.0
    current_sector = get_sector(ticker)
    matching = sum(1 for s in STOCK_SECTORS if get_sector(s) == current_sector)
    return float(matching / len(STOCK_SECTORS))


def build_feature_row(ticker: str, ohlcv_df: pd.DataFrame, stock_returns: pd.Series,
                       benchmark_returns: pd.Series) -> dict:
    """Turn one stock's price history into the 5-feature row the ensemble
    model was trained on. Reuses extract_ensemble_features_from_price_series
    (optimization/ensemble.py) rather than recomputing recent_return /
    volatility / momentum a second time.

    Returns None if there isn't enough price history yet (mirrors the
    ValueError extract_ensemble_features_from_price_series raises for
    too-short series).
    """
    if ohlcv_df is None or "Close" not in ohlcv_df or ohlcv_df["Close"].dropna().empty:
        return None

    sector_exposure = _universe_sector_exposure(ticker)

    try:
        feature_vector = extract_ensemble_features_from_price_series(
            ohlcv_df["Close"],
            sector_exposure=sector_exposure,
            risk_score=None,
        )
    except ValueError:
        return None

    row = dict(zip(FEATURE_COLUMNS, feature_vector.flatten().tolist()))
    return row


def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """%K and %D lines. %K crossing above %D from oversold (<20) territory is
    a leading buy cue -- it typically turns before RSI confirms."""
    low_min = df["Low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["High"].rolling(window=k_period, min_periods=k_period).max()
    percent_k = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-12)
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()
    return percent_k, percent_d


def williams_percent_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Range: -100 (oversold) to 0 (overbought). Reacts faster than RSI to
    reversals because it weights the most recent close more heavily."""
    high_max = df["High"].rolling(window=period, min_periods=period).max()
    low_min = df["Low"].rolling(window=period, min_periods=period).min()
    return -100 * (high_max - df["Close"]) / (high_max - low_min + 1e-12)


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """Cumulative volume flow. Volume often shifts before price does (smart
    money accumulating/distributing), so a rising OBV while price is flat is
    a classic leading bullish divergence, and vice versa."""
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


def rate_of_change(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """% change in price over `period` days. Measures acceleration, not just
    direction -- a rising ROC flags momentum building before a breakout."""
    return (df["Close"] - df["Close"].shift(period)) / df["Close"].shift(period) * 100


def vix_regime(vix_series: pd.Series, lookback: int = 20) -> dict:
    """
    India VIX is market-wide (not per-stock) and forward-looking -- it prices
    in *expected* volatility from options, ahead of it showing up in realized
    price moves. Rather than a per-stock vote, it returns a market regime
    used to scale conviction across all stocks at once:
      - "calm":      current VIX well below its recent average -> full weight
      - "normal":    within a normal band
      - "elevated":  VIX rising / above recent average -> dampen conviction
      - "unknown":   not enough VIX history
    """
    if vix_series is None or vix_series.empty or len(vix_series) < lookback:
        return {"regime": "unknown", "current": None, "avg": None}

    current = float(vix_series.iloc[-1])
    avg = float(vix_series.tail(lookback).mean())

    if current < avg * 0.85:
        regime = "calm"
    elif current > avg * 1.15:
        regime = "elevated"
    else:
        regime = "normal"

    return {"regime": regime, "current": round(current, 2), "avg": round(avg, 2)}


def leading_signal(df: pd.DataFrame, vix_series: pd.Series = None) -> dict:
    """
    Combine the four price/volume-based leading indicators into one vote,
    mirroring the scoring pattern used for lagging indicators in
    trading_signals.py so the two can be merged consistently in
    composite_score.py (Phase 8, next step). India VIX is folded in as a
    market-wide dampener rather than a per-stock vote, since it doesn't
    belong to any single stock.

    Returns: {"signal": "BUY"|"SELL"|"HOLD", "score": float, "detail": {...}}
    """
    if len(df) < 30:
        return {"signal": "HOLD", "score": 0, "detail": {}}

    percent_k, percent_d = stochastic_oscillator(df)
    williams_r = williams_percent_r(df)
    obv = on_balance_volume(df)
    roc = rate_of_change(df)

    k_now, d_now = percent_k.iloc[-1], percent_d.iloc[-1]
    k_prev, d_prev = percent_k.iloc[-2], percent_d.iloc[-2]
    wr_now = williams_r.iloc[-1]
    roc_now = roc.iloc[-1]

    obv_slope = obv.diff().tail(5).mean()
    price_slope = df["Close"].diff().tail(5).mean()

    score = 0
    detail = {}

    # Stochastic: bullish/bearish %K-%D crossover in oversold/overbought zone
    if k_prev < d_prev and k_now > d_now and k_now < 30:
        score += 2; detail["stochastic"] = "bullish_crossover_oversold"
    elif k_prev > d_prev and k_now < d_now and k_now > 70:
        score -= 2; detail["stochastic"] = "bearish_crossover_overbought"
    else:
        detail["stochastic"] = "neutral"

    # Williams %R: extreme zones
    if wr_now <= -80:
        score += 1; detail["williams_r"] = "oversold"
    elif wr_now >= -20:
        score -= 1; detail["williams_r"] = "overbought"
    else:
        detail["williams_r"] = "neutral"

    # OBV divergence: volume accumulating while price isn't moving up yet
    if obv_slope > 0 and price_slope <= 0:
        score += 1; detail["obv"] = "bullish_divergence"
    elif obv_slope < 0 and price_slope >= 0:
        score -= 1; detail["obv"] = "bearish_divergence"
    else:
        detail["obv"] = "confirming"

    # ROC: momentum acceleration
    if roc_now > 5:
        score += 1; detail["roc"] = "accelerating_up"
    elif roc_now < -5:
        score -= 1; detail["roc"] = "accelerating_down"
    else:
        detail["roc"] = "flat"

    # India VIX: market-wide dampener, not a per-stock vote.
    # Elevated VIX shrinks the score toward HOLD -- it doesn't flip direction,
    # it lowers conviction, since a spike affects every stock at once.
    vix_info = vix_regime(vix_series) if vix_series is not None else {"regime": "unknown"}
    detail["india_vix"] = vix_info
    if vix_info["regime"] == "elevated":
        score *= 0.5

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {"signal": signal, "score": round(score, 2), "detail": detail}