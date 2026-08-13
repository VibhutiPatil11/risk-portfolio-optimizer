from flask import Flask, request, jsonify   # Flask for API, request for input, jsonify for response
from flask_cors import CORS                 # Enable cross-origin requests (React ↔ Flask)
import numpy as np                          # Numerical operations
import os

# Core Config + Database
from config import Config                   # App configuration
from database import db                     # Database connection

# JWT Authentication
from flask_jwt_extended import JWTManager  # For handling authentication tokens

# Auth Routes
from routes.auth_routes import auth_bp     # Blueprint for login/signup routes

# Existing Portfolio Modules
from data.fetch_data import fetch_data, fetch_benchmark_data   # Fetch stock + benchmark data
from data.stock_sectors import get_sector                      # Get sector of stock
from preprocessing.preprocess import calculate_returns         # Calculate stock returns
from risk.risk_metrics import calculate_risk, calculate_tail_risk  # Risk calculations
from optimization.portfolio_optimizer import (
    optimize_portfolio,
    calculate_risk_contribution
)
from optimization.ensemble import (
    ENSEMBLE_FEATURE_METADATA,
    ENSEMBLE_FEATURE_ORDER,
    SimpleEnsembleModel,
    extract_ensemble_features_from_price_series,
)
from signals.lagging_indicators import generate_signals, generate_portfolio_signal
# Generate buy/sell signals


# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)                      # Create Flask app
app.config.from_object(Config)             # Load configuration

db.init_app(app)                           # Initialize database

jwt = JWTManager(app)                      # Initialize JWT auth

CORS(app)                                  # Enable CORS

app.register_blueprint(auth_bp)            # Register authentication routes


# -----------------------------
# Constants
# -----------------------------
TRADING_DAYS = 252                         # Number of trading days in a year
MARKET_VOLATILITY_THRESHOLD = 0.25         # Threshold to detect high volatility market
BENCHMARK_TICKER = "^NSEI"                 # NIFTY 50 index as benchmark


def calculate_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()  
    # Track highest portfolio value over time

    drawdowns = (cumulative_returns / running_max) - 1  
    # Calculate percentage drop from peak

    return abs(float(drawdowns.min())) if not drawdowns.empty else 0.0  
    # Return maximum loss


def calculate_portfolio_beta(portfolio_returns, benchmark_returns):
    aligned = portfolio_returns.to_frame("portfolio").join(
        benchmark_returns.rename("benchmark"),
        how="inner"
    ).dropna()
    # Align portfolio and benchmark data

    if aligned.empty or aligned["benchmark"].var() == 0:
        return 0.0  
        # Avoid division error

    covariance = aligned["portfolio"].cov(aligned["benchmark"])  
    # Covariance between portfolio and market

    benchmark_variance = aligned["benchmark"].var()  
    # Market variance

    return float(covariance / benchmark_variance)  
    # Beta = covariance / variance


# -----------------------------
# NIFTY 50 List
# -----------------------------
NIFTY_50_STOCKS = [
    # Predefined list of stocks shown in frontend
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "INDUSINDBK", "INFY", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND",
    "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT"
]


PORTFOLIO_SIGNAL_REASONS = {
    "BUY": "Strong momentum across selected assets.",
    "SELL": "Weak momentum dominates selected assets.",
    "HOLD": "Signals are mixed, so waiting is prudent."
}

OPTIMIZATION_MODE_LABELS = {
    "min_variance": "Minimize Risk",
    "max_sharpe": "Maximize Return Score",
    "max_return": "Maximize Expected Return"
}

RISK_LEVEL_OPTIMIZATION_MODES = {
    "low": "min_variance",
    "medium": "max_sharpe",
    "high": "max_return"
}


# -----------------------------
# Health Check API
# -----------------------------
@app.route("/")
def home():
    return "Backend Running Successfully 🚀"  
    # Simple API to check server status


# -----------------------------
# Stock List API
# -----------------------------
@app.route("/stocks", methods=["GET"])
def get_stocks():
    return jsonify(NIFTY_50_STOCKS)  
    # Send stock list to frontend


# -----------------------------
# Ensemble Prediction API
# -----------------------------
ENSEMBLE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "ensemble_model.joblib")
ENSEMBLE_MODEL = None
def get_ensemble_model():
    global ENSEMBLE_MODEL
    if ENSEMBLE_MODEL is None:
        if not os.path.exists(ENSEMBLE_MODEL_PATH):
            raise FileNotFoundError("Ensemble model not found. Generate and train the model first.")
        model = SimpleEnsembleModel(model_dir=os.path.dirname(ENSEMBLE_MODEL_PATH))
        model.load(os.path.basename(ENSEMBLE_MODEL_PATH))
        ENSEMBLE_MODEL = model
    return ENSEMBLE_MODEL


def normalize_nse_ticker(symbol):
    ticker = str(symbol or "").strip().upper()
    if not ticker:
        raise ValueError("ticker must be a non-empty string.")
    return ticker if ticker.endswith(".NS") else f"{ticker}.NS"


def ticker_to_stock_symbol(ticker):
    return str(ticker or "").replace(".NS", "").upper()


def calculate_dynamic_sector_exposure(ticker, data):
    portfolio_stocks = data.get("portfolio_stocks") or data.get("stocks") or []
    if not isinstance(portfolio_stocks, list):
        portfolio_stocks = []

    cleaned_stocks = [ticker_to_stock_symbol(stock) for stock in portfolio_stocks if str(stock or "").strip()]
    current_stock = ticker_to_stock_symbol(ticker)
    if current_stock and current_stock not in cleaned_stocks:
        cleaned_stocks.append(current_stock)

    if not cleaned_stocks:
        return 1.0

    current_sector = get_sector(current_stock)
    sector_count = sum(1 for stock in cleaned_stocks if get_sector(stock) == current_sector)
    return float(sector_count / len(cleaned_stocks))


def feature_support_label(key, value):
    value = float(value)
    if key == "recent_return":
        if value > 0.05:
            return "BUY"
        if value < -0.05:
            return "SELL"
        return "HOLD"
    if key == "volatility":
        if value < 0.02:
            return "BUY"
        if value > 0.04:
            return "SELL"
        return "HOLD"
    if key == "momentum":
        if value > 0.05:
            return "BUY"
        if value < -0.05:
            return "SELL"
        return "HOLD"
    if key == "sector_exposure":
        if value > 0.6:
            return "SELL"
        if value < 0.35:
            return "BUY"
        return "HOLD"
    if key == "risk_score":
        if value > 0.98:
            return "BUY"
        if value < 0.96:
            return "SELL"
        return "HOLD"
    return "HOLD"


def feature_plain_explanation(key, value):
    value = float(value)
    if key == "recent_return":
        direction = "gained" if value >= 0 else "fell"
        return f"The stock {direction} {abs(value) * 100:.2f}% in the last 7 trading days."
    if key == "volatility":
        return f"Daily price swings are around {value * 100:.2f}%; lower swings usually mean lower risk."
    if key == "momentum":
        if value > 0:
            return f"The 30-day trend is upward by {value * 100:.2f}%."
        if value < 0:
            return f"The 30-day trend is downward by {abs(value) * 100:.2f}%."
        return "The 30-day trend is flat."
    if key == "sector_exposure":
        return f"About {value * 100:.2f}% of the selected stocks are in this stock's sector."
    if key == "risk_score":
        return f"The risk score is {value:.2f}; higher means the stock looks more stable."
    return "Calculated from recent market data."


def format_dynamic_feature_details(features):
    first_row = np.asarray(features, dtype=float)[0]
    details = []
    for meta, value in zip(ENSEMBLE_FEATURE_METADATA, first_row):
        details.append({
            "key": meta["key"],
            "label": meta.get("label", meta["key"]),
            "value": float(value),
            "support": feature_support_label(meta["key"], value),
            "description": feature_plain_explanation(meta["key"], value),
        })
    return details


def calculate_dynamic_ensemble_features(data):
    ticker = normalize_nse_ticker(data.get("ticker"))
    sector_exposure = calculate_dynamic_sector_exposure(ticker, data)

    price_data, valid_tickers, failed_tickers = fetch_data([ticker], period="6mo", interval="1d")
    if not valid_tickers:
        raise ValueError(f"Unable to fetch price data for ticker {ticker}")

    features = extract_ensemble_features_from_price_series(
        price_data[valid_tickers[0]],
        sector_exposure=sector_exposure,
        risk_score=None,
    )
    return features, format_dynamic_feature_details(features), ticker_to_stock_symbol(ticker)


def parse_features(data):
    if "features" in data:
        features = data["features"]
    elif "feature_vector" in data:
        features = data["feature_vector"]
    elif "feature_matrix" in data:
        features = data["feature_matrix"]
    elif "ticker" in data:
        features, _, _ = calculate_dynamic_ensemble_features(data)
        return features
    else:
        raise ValueError("Request JSON must include 'features', 'feature_vector', 'feature_matrix', or 'ticker'.")

    if isinstance(features, dict):
        missing = [name for name in ENSEMBLE_FEATURE_ORDER if name not in features]
        if missing:
            raise ValueError(f"Missing feature keys: {', '.join(missing)}")
        features = [features[name] for name in ENSEMBLE_FEATURE_ORDER]

    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    return features


@app.route("/ensemble/predict", methods=["POST"])
def ensemble_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    try:
        features = parse_features(data)
        model = get_ensemble_model()
        predictions = model.predict(features)
        return jsonify({
            "predictions": [float(x) for x in predictions],
            "model": "simple_ensemble"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/ensemble/explain", methods=["POST"])
def ensemble_explain():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    try:
        feature_details = None
        analyzed_stock = None
        if "ticker" in data and not any(key in data for key in ("features", "feature_vector", "feature_matrix")):
            features, feature_details, analyzed_stock = calculate_dynamic_ensemble_features(data)
        else:
            features = parse_features(data)

        model = get_ensemble_model()

        # Primary prediction
        predictions = model.predict(features)

        # Baseline: use scaler mean if available, otherwise zeros
        if hasattr(model.scaler, "mean_"):
            baseline = np.asarray(model.scaler.mean_, dtype=float)
        else:
            baseline = np.zeros(features.shape[1], dtype=float)

        baseline_mat = np.tile(baseline, (features.shape[0], 1))

        # Per-feature contribution: difference between full prediction and prediction
        # with the feature set to baseline (simple ablation approximation).
        contributions = []
        for i, key in enumerate(ENSEMBLE_FEATURE_ORDER):
            X_ablated = np.array(features, copy=True)
            X_ablated[:, i] = baseline_mat[:, i]
            pred_ablated = model.predict(X_ablated)
            # For now return contribution for first input row (UI uses single vector)
            contributions.append(float(predictions[0] - pred_ablated[0]))

        # Base model predictions (for transparency)
        base_preds = model.get_base_predictions(features)
        base_names = [
            "ridge",
            "random_forest",
            "hist_gradient_boosting"
        ]
        base_predictions = {
            name: [float(x) for x in preds]
            for name, preds in zip(base_names, base_preds)
        }

        explanations = [
            {
                "key": meta["key"],
                "label": meta.get("label", meta.get("key")),
                "contribution": contrib
            }
            for meta, contrib in zip(ENSEMBLE_FEATURE_METADATA, contributions)
        ]

        return jsonify({
            "predictions": [float(x) for x in predictions],
            "features": feature_details or format_dynamic_feature_details(features),
            "explanations": explanations,
            "base_predictions": base_predictions,
            "analyzed_stock": analyzed_stock,
            "model": "simple_ensemble"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------
# Portfolio Optimization API
# -----------------------------
@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        data = request.json  
        # Get user input from frontend

        if not data or "stocks" not in data:
            return jsonify({"error": "Stocks list missing"}), 400

        stocks = data["stocks"]  
        investment_amount = data.get("investment", 100000)  
        risk_level = str(data.get("risk", "medium")).lower()

        # Validate inputs
        if risk_level not in {"low", "medium", "high"}:
            risk_level = "medium"

        try:
            investment_amount = float(investment_amount)
        except:
            return jsonify({"error": "Investment must be a number"}), 400

        if investment_amount <= 0:
            return jsonify({"error": "Investment must be greater than 0"}), 400

        if len(stocks) < 2:
            return jsonify({"error": "Select at least 2 stocks"}), 400


        # Add .NS for NSE stocks
        tickers = [
            s if s.upper().endswith(".NS") else f"{s}.NS"
            for s in stocks
        ]

        stock_by_ticker = {ticker: stock for stock, ticker in zip(stocks, tickers)}

        price_data, valid_tickers, failed_tickers = fetch_data(tickers)  
        # Fetch stock data

        # Filter valid stocks
        valid_stocks = [stock_by_ticker[ticker] for ticker in valid_tickers if ticker in stock_by_ticker]
        removed_stocks = [stock_by_ticker[ticker] for ticker in failed_tickers if ticker in stock_by_ticker]

        if len(valid_stocks) < 2:
            return jsonify({"error": "Not enough valid stocks"}), 400


        # -----------------------------
        # Core Processing Pipeline
        # -----------------------------
        returns = calculate_returns(price_data)  
        # Calculate returns

        returns = returns.rename(columns={ticker: stock for stock, ticker in zip(stocks, tickers)})

        mean_returns, cov_matrix = calculate_risk(returns)  
        # Risk metrics

        market_proxy_returns = returns.mean(axis=1)  
        market_volatility_annual = float(market_proxy_returns.std() * np.sqrt(TRADING_DAYS))

        if market_volatility_annual > MARKET_VOLATILITY_THRESHOLD:
            optimize_mode = "min_variance"
        else:
            optimize_mode = RISK_LEVEL_OPTIMIZATION_MODES.get(risk_level, "max_sharpe")
        # Adjust strategy based on market condition


        # Run optimization
        weights, portfolio_return_daily, volatility_daily, portfolio_return_annual, volatility_annual, sharpe = optimize_portfolio(
            mean_returns,
            cov_matrix,
            risk_level=risk_level,
            optimize_mode=optimize_mode
        )

        signals = generate_signals(price_data, valid_stocks, valid_tickers)
        portfolio_signal = generate_portfolio_signal(signals)

        # -----------------------------
        # Allocation & Sector Analysis
        # -----------------------------
        allocation_chart = []
        sector_weights = {}

        for stock, weight in zip(valid_stocks, weights):
            sector = get_sector(stock)  
            # Get stock sector

            sector_weights[sector] = sector_weights.get(sector, 0) + float(weight)

            allocation_chart.append({
                "name": stock,
                "sector": sector,
                "value": round(float(weight * 100), 2)
            })


        # Diversification score
        diversification_score = (1 - float(np.sum(np.square(list(sector_weights.values()))))) * 100


        # -----------------------------
        # Performance + Benchmark
        # -----------------------------
        portfolio_returns = returns.dot(weights)

        benchmark_prices = fetch_benchmark_data(BENCHMARK_TICKER)
        benchmark_returns = benchmark_prices.pct_change().dropna()

        portfolio_beta = calculate_portfolio_beta(portfolio_returns, benchmark_returns)
        # Measure market sensitivity


        # -----------------------------
        # Risk Metrics
        # -----------------------------
        var_95_daily, cvar_95_daily = calculate_tail_risk(portfolio_returns)
        # Tail risk

        risk_contribution = calculate_risk_contribution(weights, cov_matrix.values)
        # Contribution of each stock to total risk

        cumulative_returns = (1 + portfolio_returns).cumprod()
        aligned_curve = portfolio_returns.to_frame("portfolio_return").join(
            benchmark_returns.rename("benchmark_return"),
            how="inner"
        ).dropna()
        portfolio_curve = (1 + aligned_curve["portfolio_return"]).cumprod() * float(investment_amount)
        benchmark_curve = (1 + aligned_curve["benchmark_return"]).cumprod() * float(investment_amount)
        max_drawdown = calculate_max_drawdown(cumulative_returns)


        portfolio_signal_reason = PORTFOLIO_SIGNAL_REASONS[portfolio_signal]


        # -----------------------------
        # Final Response
        # -----------------------------
        benchmark_return_annual = (
            float(benchmark_returns.mean() * TRADING_DAYS)
            if not benchmark_returns.empty else 0.0
        )

        expected_return_percent = float(portfolio_return_annual * 100)
        volatility_percent = float(volatility_annual * 100)
        var_95_percent = float(abs(var_95_daily) * 100)
        cvar_95_percent = float(abs(cvar_95_daily) * 100)
        max_drawdown_percent = float(max_drawdown * 100)

        portfolio_metrics = {
            "portfolio_value": round(float(portfolio_curve.iloc[-1]), 2) if not portfolio_curve.empty else round(float(investment_amount), 2),
            "investment_amount": round(float(investment_amount), 2),
            "expected_return": round(expected_return_percent, 2),
            "returns": round(expected_return_percent, 2),
            "sharpe": round(float(sharpe), 4),
            "volatility": round(volatility_percent, 2),
            "var_95": round(var_95_percent, 2),
            "cvar_95": round(cvar_95_percent, 2),
            "portfolio_beta": round(float(portfolio_beta), 4),
            "benchmark_return": round(float(benchmark_return_annual * 100), 2),
            "expected_return_daily": round(float(portfolio_return_daily), 6),
            "expected_return_annual": round(float(portfolio_return_annual), 6),
            "volatility_daily": round(float(volatility_daily), 6),
            "volatility_annual": round(float(volatility_annual), 6),
            "sharpe_ratio": round(float(sharpe), 4),
            "beta": round(float(portfolio_beta), 4),
            "var_95_daily": round(float(var_95_daily), 6),
            "cvar_95_daily": round(float(cvar_95_daily), 6),
            "max_drawdown": round(max_drawdown_percent, 2),
            "diversification_score": round(float(diversification_score), 2),
            "optimize_mode": optimize_mode,
            "optimization_method": OPTIMIZATION_MODE_LABELS.get(optimize_mode, "Maximize Return Score"),
            "market_volatility_annual": round(float(market_volatility_annual), 6),
            "removed_stocks": removed_stocks
        }

        invested_sector_count = len(sector_weights)

        portfolio_summary = {
            "investment_amount": round(float(investment_amount), 2),
            "stocks_selected": len(valid_stocks),
            "sectors_covered": invested_sector_count,
            "risk_level": risk_level,
            "optimize_mode": optimize_mode,
            "optimization_method": OPTIMIZATION_MODE_LABELS.get(optimize_mode, "Maximize Return Score"),
            "portfolio_signal": portfolio_signal,
            "expected_return": round(expected_return_percent, 2)
        }

        sector_allocation = [
            {
                "name": sector,
                "sector": sector,
                "value": round(float(weight * 100), 2)
            }
            for sector, weight in sector_weights.items()
        ]

        performance_curve = [
            {
                "date": str(index.date()) if hasattr(index, "date") else str(index),
                "portfolio": round(float(portfolio_curve.loc[index]), 2),
                "benchmark": round(float(benchmark_curve.loc[index]), 2)
            }
            for index in portfolio_curve.index
        ]

        risk_contribution_chart = [
            {
                "stock": stock,
                "name": stock,
                "value": round(float(contribution * 100), 2)
            }
            for stock, contribution in zip(valid_stocks, risk_contribution)
        ]

        response_payload = {
            "portfolio_metrics": portfolio_metrics,
            "portfolio_summary": portfolio_summary,
            "allocation": allocation_chart,
            "sector_allocation": sector_allocation,
            "performance_curve": performance_curve,
            "risk_contribution": risk_contribution_chart,
            "signals": signals,
            "portfolio_signal": portfolio_signal,
            "portfolio_signal_reason": portfolio_signal_reason
        }

        return jsonify(response_payload)  
        # Send final result to frontend

    except Exception as e:
        return jsonify({"error": str(e)}), 500  
        # Handle errors safely


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  
        # Create database tables

    app.run(debug=True)  
    # Start backend server
