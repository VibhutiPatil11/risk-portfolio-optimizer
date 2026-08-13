"""
ADD this function to Backend/optimization/portfolio_optimizer.py (append it --
nothing existing needs to change, optimize_portfolio() itself is untouched).

Makes the optimizer forward-looking instead of purely historical: previously
`mean_returns` passed into optimize_portfolio() was always the historical
daily mean, so MPT ran without ever consulting the ML model (Step 3) or the
recommendation engine (Step 4). This blends in the ML model's predicted
return per stock.

IMPORTANT SCALE MISMATCH THIS FUNCTION HANDLES:
  `mean_returns` (existing, from risk_metrics.calculate_risk) is a DAILY
  return -- e.g. 0.0005/day for a stock returning ~12%/year.

  The ML model's `predicted_return` (Step 3) is NOT a daily return -- it was
  trained against a target built from `recent_return` (10-day window) and
  `momentum` (20-day window) style features (see preprocessing/live_features.py),
  so its scale is closer to a ~10-trading-day forward return -- e.g. 0.08
  for a stock expected to move 8% over the next two weeks.

  Blending these directly (0.6 * 0.08 + 0.4 * 0.0005) would make the ML
  side ~150x too large relative to the historical side, and the optimizer
  would essentially ignore the covariance/risk term and dump everything
  into whichever stock had the highest raw ML score -- silently wrong, not
  an error. ML_RETURN_HORIZON_DAYS converts the ML prediction back to an
  implied daily rate before blending.
"""

import numpy as np
import pandas as pd

ML_RETURN_HORIZON_DAYS = 10  # matches live_features.py's recent_return window


def build_expected_returns(
    historical_mean_returns: pd.Series,
    ml_predicted_returns: dict,
    ml_weight: float = 0.6,
    ml_horizon_days: int = ML_RETURN_HORIZON_DAYS,
) -> pd.Series:
    """
    historical_mean_returns : pandas Series, index = ticker, DAILY mean return
                               (from risk_metrics.calculate_risk)
    ml_predicted_returns    : dict {ticker: predicted_return}, from the
                               ensemble model / composite engine -- a
                               ~10-day-horizon return, NOT daily
    ml_weight                : float in [0,1], how much to trust the ML view
                                vs. the historical daily average
    ml_horizon_days           : the forward window the ML prediction
                                 represents, used to convert it to a daily
                                 equivalent rate (predicted_return / horizon)

    Returns a pandas Series in the SAME shape/order as historical_mean_returns,
    ready to pass into optimize_portfolio() as `mean_returns`.
    """
    blended = historical_mean_returns.copy().astype(float)

    for ticker in blended.index:
        ml_val = ml_predicted_returns.get(ticker)
        if ml_val is None:
            continue  # no ML prediction for this ticker -- keep the historical value as-is
        ml_daily_equivalent = ml_val / ml_horizon_days
        blended[ticker] = ml_weight * ml_daily_equivalent + (1 - ml_weight) * blended[ticker]

    return blended
