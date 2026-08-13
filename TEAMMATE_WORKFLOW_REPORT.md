# AI-Powered Portfolio Optimizer — Workflow and Implementation Report

## Purpose

This project helps an investor choose NIFTY 50 stocks and allocate an investment amount according to a selected risk profile. It combines live market data, technical indicators, machine-learning signals, portfolio-risk metrics, and Markowitz optimization into one decision-support workflow.

## Intended user workflow

1. The user signs up, completes KYC, and logs in.
2. The system fetches live OHLCV price data for the NIFTY 50 universe, the NIFTY 50 index (`^NSEI`), and India VIX (`^INDIAVIX`) through Yahoo Finance.
3. For every stock, the backend calculates:
   - Lagging indicators: RSI, SMA-20, SMA-50, and momentum.
   - Leading indicators: Stochastic Oscillator, Williams %R, OBV, and ROC.
   - Risk measures: volatility, VaR, CVaR, covariance, beta, and drawdown.
4. The ensemble ML model produces an expected-return estimate and BUY/HOLD/SELL probabilities. The Random Forest component can also provide SHAP feature contributions.
5. The recommendation engine combines ML, leading, lagging, and risk components into one final BUY/HOLD/SELL result, with confidence and an explanation per stock.
6. Immediately after login, the user sees the complete NIFTY stock universe colour-coded by that final signal:
   - Green: BUY
   - Yellow: HOLD
   - Red: SELL
7. The user can open one or more stock detail cards to compare candlestick charts, SMA overlays, RSI, signal markers, and recommendation explanations.
8. The user then selects a portfolio basket, enters an investment amount, and chooses a risk level:
   - Conservative / Low: minimise variance.
   - Moderate / Medium: maximise Sharpe ratio.
   - Aggressive / High: maximise expected return.
9. Markowitz optimization calculates stock weights subject to full-investment and maximum-concentration constraints.
10. The results screen shows allocation, sector exposure, expected return, volatility, Sharpe ratio, VaR, CVaR, beta, maximum drawdown, diversification score, NIFTY benchmark comparison, and portfolio signal.
11. The system should periodically refresh signals and identify signal changes or portfolio-rebalancing needs.

## Current implementation status

| Workflow area | Current status | Notes |
| --- | --- | --- |
| Authentication and KYC | Implemented | Flask JWT endpoints and Postgres-backed users are present. JWT secret validation was added. |
| NIFTY universe dashboard | Implemented | `/dashboard/stocks` fetches, scores, caches, and returns the stock universe. |
| Stock technical detail | Implemented | `/stocks/<ticker>/details` returns candles, SMA-20/SMA-50, RSI, marker, and composite explanation. |
| Multi-stock chart comparison | Implemented | The React universe page now keeps multiple selected stock detail cards open simultaneously. |
| ML classification | Implemented | The ensemble artifact was retrained with BUY/HOLD/SELL labels, resolving the untrained-classifier error. |
| Signal fusion | Implemented for dashboard/detail | `composite_score.py` combines ML (40%), leading (25%), lagging (20%), and risk (15%). |
| Markowitz optimizer | Implemented | `/optimize` supports min-variance, max-Sharpe, and max-return modes with allocation limits. |
| Benchmark comparison | Implemented | `/optimize` obtains NIFTY data and returns beta, benchmark return, and aligned performance data. |
| Optimization UI handoff | Not currently connected | `App.js` renders `UniverseDashboard`, while the amount/risk/optimization interface remains in `src/pages/Dashboard.js` and is not currently reachable. |
| ML-predicted returns in optimizer | Prepared but not connected | `Backend/optimization/build_expected_returns.py` contains a scale-aware blending helper, but `/optimize` currently passes historical mean returns directly to `optimize_portfolio`. |
| Periodic monitoring and alerts | Not implemented | Dashboard polling only rereads cached results. There is no backend scheduler, persistent signal history, rebalance rule, or notification channel yet. |

## Changes completed in this update cycle

1. Rebuilt `Backend/data/ensemble_model.joblib` with a trained `RandomForestClassifier`.
   - Fixes repeated `Classifier was not trained` dashboard errors.
   - Allows real BUY/HOLD/SELL probabilities and confidence values.

2. Updated the retired `TATAMOTORS` Yahoo Finance symbol to `TMPV`.
   - Prevents the former `TATAMOTORS.NS` quote-not-found failure after the Tata Motors demerger.

3. Strengthened JWT setup.
   - The application now rejects missing or short JWT secrets at startup.
   - The development `.env` key was upgraded to a SHA-256-safe length.

4. Fixed the candlestick chart crash.
   - Lightweight Charts v5 uses `createSeriesMarkers(...)`; `series.setMarkers(...)` is no longer available.

5. Added multi-stock detail comparison.
   - Selecting additional stocks no longer replaces the current chart.
   - Each selected stock gets its own chart and explanation card and can be removed independently.

## Recommended next implementation steps

1. Connect the stock-selection view to the existing optimizer form.
   - Add a “Build portfolio” action to pass selected tickers from `UniverseDashboard` into `Dashboard.js`, or merge the optimizer form/results into the universe page.
   - Keep chart selection and portfolio selection as clearly labelled states if they serve different purposes.

2. Use ML-informed returns in `/optimize`.
   - Compute the selected stocks’ composite/ML predictions.
   - Call `build_expected_returns(historical_mean_returns, ml_predicted_returns)`.
   - Pass the blended daily expected-return series to `optimize_portfolio` while retaining historical covariance for risk estimation.

3. Use fused composite signals for portfolio results.
   - `signals/portfolio_composite.py` already contains helper functions for a selected portfolio.
   - Replace the older lagging-only signal call in `/optimize` with the composite portfolio signal path.

4. Add monitoring and rebalancing.
   - Schedule dashboard refreshes in the backend.
   - Persist per-stock and per-portfolio signals with timestamps.
   - Define a rebalance rule, for example: signal reversal, allocation drift above a threshold, or volatility exceeding the user’s risk limit.
   - Add in-app/email notification delivery only after the trigger rules are agreed.

## Key API endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /signup` | Creates the user and returns a JWT for KYC. |
| `POST /login` | Authenticates an approved user. |
| `GET /dashboard/stocks` | Returns cached, colour-coded composite signals for the stock universe. |
| `POST /dashboard/refresh` | Recomputes dashboard signals on demand. |
| `GET /stocks/<ticker>/details` | Returns chart data and the stock’s recommendation explanation. |
| `POST /optimize` | Builds optimized allocations and portfolio-level risk/benchmark outputs. |

## Architecture summary

```text
Yahoo Finance (stocks + NIFTY + VIX)
        |
        v
Feature engineering and risk calculations
        |
        +--> Ensemble ML prediction + SHAP
        |
        +--> Leading and lagging indicators
        |
        v
Composite BUY/HOLD/SELL recommendation
        |
        +--> React NIFTY universe and stock-detail charts
        |
        +--> Selected basket + risk preference
                 |
                 v
          Markowitz portfolio optimizer
                 |
                 v
   Allocation, risk metrics, NIFTY comparison, rebalance monitoring
```

## Team handoff note

The dashboard and charting work is operational, including multi-stock inspection. The highest-priority product gap is connecting that inspection/selection flow to the investment amount, risk-level, and `/optimize` result interface. Once that UI handoff and ML-return blending are wired, the app will match the core workflow described above; periodic monitoring is the remaining phase after that.
