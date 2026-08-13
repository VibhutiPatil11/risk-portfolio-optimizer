"""
The recommendation engine -- the piece that was missing from the original
system. Previously ML predictions (ensemble.py) and technical indicators
(trading_signals.py) were computed completely independently and never
combined; the portfolio-level result used neither. This module is the
single place where ML + leading indicators + lagging indicators + risk are
fused into ONE final BUY/HOLD/SELL per stock, with a confidence score and a
"why" breakdown suitable for showing directly on a dashboard block.

Weighting:
  ML classifier (Step 3):        40%  -- forward-looking, learned from data
  Leading indicators (Step 1):   25%  -- anticipates moves before price confirms
  Lagging indicators (existing): 20%  -- confirms trend, still a useful sanity check
  Risk penalty (existing CVaR):  15%  -- a risky BUY should not look identical
                                          to a safe BUY
"""

from typing import Optional

import numpy as np
import pandas as pd

from preprocessing.live_features import build_feature_row, FEATURE_COLUMNS
from signals.leading_indicators import leading_signal
from signals.lagging_indicators import _calculate_rsi, _combined_signal, _confidence_from_rsi
from risk.risk_metrics import calculate_tail_risk
from data.stock_sectors import get_sector

SIGNAL_TO_NUM = {"SELL": -1, "HOLD": 0, "BUY": 1}
COLOR_FOR_SIGNAL = {"BUY": "green", "HOLD": "yellow", "SELL": "red"}
WEIGHTS = {"ml": 0.40, "leading": 0.25, "lagging": 0.20, "risk": 0.15}


def _normalize(value: float, scale: float) -> float:
    """Squash an arbitrary-range score into [-1, 1]."""
    if scale == 0:
        return 0.0
    return float(np.clip(value / scale, -1.0, 1.0))


def _lagging_component(ohlcv_df: pd.DataFrame) -> dict:
    """Reuses the EXISTING lagging-indicator logic (trading_signals.py)
    rather than duplicating it -- RSI + SMA20/50 crossover + 10-day momentum."""
    close = ohlcv_df["Close"]
    if len(close) < 51:
        return {"signal": "HOLD", "confidence": 0.0, "rsi": None}

    rsi_series = _calculate_rsi(close, period=14)
    rsi_value = float(rsi_series.iloc[-1])
    sma20 = float(close.rolling(window=20, min_periods=20).mean().iloc[-1])
    sma50 = float(close.rolling(window=50, min_periods=50).mean().iloc[-1])

    signal = _combined_signal(rsi_value, sma20, sma50, close)
    confidence = _confidence_from_rsi(rsi_value)

    return {"signal": signal, "confidence": round(confidence, 2), "rsi": round(rsi_value, 2)}


def compute_composite_signal(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    model,
    vix_series: Optional[pd.Series] = None,
) -> Optional[dict]:
    """
    Parameters
    ----------
    ticker            : e.g. "TCS" (without .NS suffix)
    ohlcv_df          : DataFrame with Open/High/Low/Close/Volume for this stock
                         (from data.fetch_data.fetch_ohlcv_data)
    stock_returns     : daily returns for this stock (from preprocessing.preprocess.calculate_returns)
    benchmark_returns : daily returns for the NIFTY 50 index (from fetch_benchmark_data)
    model             : a fitted optimization.ensemble.SimpleEnsembleModel
                         (must have been trained with labels, i.e. classifier is not None)
    vix_series        : optional India VIX series, dampens the leading-indicator
                         component market-wide when volatility is elevated

    Returns
    -------
    dict with the fused signal, color, score, confidence, and a full
    per-component breakdown -- or None if there isn't enough history yet.
    """
    # --- ML component (Step 3) ---
    feature_row = build_feature_row(ticker, ohlcv_df, stock_returns, benchmark_returns)
    if feature_row is None:
        return None
    X = np.array([[feature_row[col] for col in FEATURE_COLUMNS]])

    predicted_return, ml_signal, ml_confidence, ml_proba = model.predict_with_confidence(X)
    ml_component = SIGNAL_TO_NUM[ml_signal] * ml_confidence

    shap_explanation = None
    try:
        shap_explanation = model.explain(X)
    except Exception:
        pass  # explainability is a bonus, never block the recommendation on it

    # --- Leading component (Step 1) ---
    leading = leading_signal(ohlcv_df, vix_series=vix_series)
    leading_component = _normalize(leading["score"], scale=4)

    # --- Lagging component (existing trading_signals.py, reused not duplicated) ---
    lagging = _lagging_component(ohlcv_df)
    lagging_component = SIGNAL_TO_NUM[lagging["signal"]]

    # --- Risk penalty (existing risk_metrics.py, reused not duplicated) ---
    var_95, cvar_95 = calculate_tail_risk(stock_returns)
    risk_penalty = -_normalize(cvar_95, scale=0.08)  # 8% CVaR treated as high risk

    weighted_score = (
        WEIGHTS["ml"] * ml_component
        + WEIGHTS["leading"] * leading_component
        + WEIGHTS["lagging"] * lagging_component
        + WEIGHTS["risk"] * risk_penalty
    )

    if weighted_score >= 0.25:
        final_signal = "BUY"
    elif weighted_score <= -0.25:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    # Confidence blends the ML classifier's own confidence with how much the
    # four components actually agree with each other.
    components = [ml_component, leading_component, lagging_component]
    agreement = 1 - (max(components) - min(components)) / 2  # 1 = all agree, 0 = fully split
    final_confidence = round(float(np.clip(0.6 * ml_confidence + 0.4 * agreement, 0, 1)), 2)

    return {
        "ticker": ticker,
        "sector": get_sector(ticker),
        "signal": final_signal,
        "color": COLOR_FOR_SIGNAL[final_signal],
        "score": round(float(weighted_score), 3),
        "confidence": final_confidence,
        "expected_return": round(float(predicted_return), 4),
        "risk_level": "high" if cvar_95 > 0.06 else ("medium" if cvar_95 > 0.03 else "low"),
        "var_95": round(var_95, 4),
        "cvar_95": round(cvar_95, 4),
        "components": {
            "ml": {
                "signal": ml_signal,
                "confidence": round(ml_confidence, 2),
                "probabilities": {k: round(v, 3) for k, v in ml_proba.items()},
                "top_factors": (
                    list(shap_explanation["contributions"].items())[:3]
                    if shap_explanation else None
                ),
            },
            "leading": {"signal": leading["signal"], "detail": leading["detail"]},
            "lagging": {"signal": lagging["signal"], "rsi": lagging["rsi"]},
        },
    }
