import os
import logging
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ENSEMBLE_FEATURE_METADATA = [
    {"key": "recent_return", "label": "Recent Return", "description": "Latest price return change."},
    {"key": "volatility", "label": "Volatility", "description": "Short-term price movement size."},
    {"key": "momentum", "label": "Momentum", "description": "Direction and speed of trend."},
    {"key": "sector_exposure", "label": "Sector Exposure", "description": "Industry weight signal."},
    {"key": "risk_score", "label": "Risk Score", "description": "Risk signal for the asset mix."},
]
ENSEMBLE_FEATURE_ORDER = [feature["key"] for feature in ENSEMBLE_FEATURE_METADATA]
VOLATILITY_FEATURE_INDEX = ENSEMBLE_FEATURE_ORDER.index("volatility")
CLASS_LABELS = ["SELL", "HOLD", "BUY"]  # fixed, alphabetical-independent order used everywhere
logger = logging.getLogger(__name__)


def extract_ensemble_features_from_price_series(
    price_series: pd.Series,
    sector_exposure: float = 0.0,
    risk_score: Optional[float] = None,
) -> np.ndarray:
    prices = price_series.dropna().astype(float)
    if prices.empty:
        raise ValueError("Price series is empty.")

    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough price history to calculate features.")

    recent_return = (
        float((prices.iloc[-1] - prices.iloc[-7]) / prices.iloc[-7])
        if len(prices) >= 7
        else float(returns.iloc[-1])
    )
    volatility = float(returns.std())
    momentum = (
        float((prices.iloc[-1] - prices.iloc[-30]) / prices.iloc[-30])
        if len(prices) >= 30
        else float(returns.tail(5).sum())
    )

    if np.isnan(recent_return):
        recent_return = 0.0
    if np.isnan(volatility) or volatility == 0.0:
        volatility = float(returns.std()) if not returns.empty else 0.0
    if np.isnan(momentum):
        momentum = 0.0

    if risk_score is None:
        risk_score = 1 - volatility

    feature_vector = [
        recent_return,
        volatility,
        momentum,
        float(sector_exposure),
        float(risk_score),
    ]
    return np.asarray(feature_vector, dtype=float).reshape(1, -1)


def build_dummy_ensemble_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    recent_return = rng.uniform(-0.15, 0.20, size=n_samples)
    volatility = np.abs(rng.normal(0.18, 0.07, size=n_samples))
    momentum = rng.uniform(-0.20, 0.25, size=n_samples)
    sector_exposure = rng.uniform(0.1, 1.5, size=n_samples)
    risk_score = -0.6 * volatility + 0.4 * momentum + rng.normal(0.0, 0.05, size=n_samples)

    target = (
        0.6 * recent_return
        - 0.8 * volatility
        + 1.2 * momentum
        + 0.3 * sector_exposure
        + 0.5 * risk_score
        + rng.normal(0.0, 0.03, size=n_samples)
    )
    target = np.clip(target, -0.3, 0.4)

    df = pd.DataFrame({
        "recent_return": recent_return,
        "volatility": volatility,
        "momentum": momentum,
        "sector_exposure": sector_exposure,
        "risk_score": risk_score,
        "target": target,
    })
    df["label"] = np.select(
        [df["target"] > 0.05, df["target"] < -0.05],
        ["BUY", "SELL"],
        default="HOLD",
    )
    return df


class SimpleEnsembleModel:
    """Ensemble model for regression predictions, PLUS classification
    probabilities and SHAP explainability.

    - Regression side (unchanged): 3 base regressors (Ridge, RandomForest,
      HistGradientBoosting), averaged, predicting expected return.
    - Classification side (new): a RandomForestClassifier trained on the
      same features against the BUY/HOLD/SELL label, giving real class
      probabilities instead of just thresholding the regression output.
    - Explainability (new): SHAP TreeExplainer on the RandomForestRegressor,
      giving per-feature contribution for any single prediction.
    """

    def __init__(self, model_dir: Optional[str] = None, weights: Optional[Sequence[float]] = None):
        self.model_dir = model_dir or os.getcwd()
        self.scaler = StandardScaler()
        self.base_models = [
            Ridge(alpha=1.0, random_state=42),
            RandomForestRegressor(n_estimators=100, random_state=42),
            HistGradientBoostingRegressor(random_state=42),
        ]
        self.weights = np.array(weights if weights is not None else [1.0, 1.0, 1.0], dtype=float)
        self.classifier: Optional[RandomForestClassifier] = None
        self._explainer = None  # built lazily on first explain() call, after fit

    def _prepare_inference_features(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] > VOLATILITY_FEATURE_INDEX:
            negative_volatility = X[:, VOLATILITY_FEATURE_INDEX] < 0
            if np.any(negative_volatility):
                logger.warning("Negative volatility feature received during inference; clipping to 0.0.")
                X = X.copy()
                X[negative_volatility, VOLATILITY_FEATURE_INDEX] = 0.0

        return X

    def fit(self, X, y, labels: Optional[Sequence[str]] = None):
        """
        X      : feature matrix
        y      : continuous target (expected return) for the regressors
        labels : optional BUY/HOLD/SELL string labels for the classifier.
                 If omitted, predict_proba()/predict_with_confidence() will
                 raise -- the regression side still works fine without it.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        for model in self.base_models:
            model.fit(X_scaled, y)

        self.weights = self.weights / np.sum(self.weights)

        if labels is not None:
            self.classifier = RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced"
            )
            self.classifier.fit(X_scaled, np.asarray(labels))

        self._explainer = None  # invalidate any stale explainer from a previous fit
        return self

    def predict(self, X):
        X = self._prepare_inference_features(X)
        X_scaled = self.scaler.transform(X)

        predictions = np.column_stack([model.predict(X_scaled) for model in self.base_models])
        return np.average(predictions, axis=1, weights=self.weights)

    def predict_proba(self, X) -> np.ndarray:
        """Returns an (n_samples, 3) array of [SELL, HOLD, BUY] probabilities."""
        if self.classifier is None:
            raise RuntimeError(
                "Classifier was not trained -- call fit(X, y, labels=...) with labels first."
            )
        X = self._prepare_inference_features(X)
        X_scaled = self.scaler.transform(X)
        # sklearn orders columns by self.classifier.classes_, which may not match
        # CLASS_LABELS order -- realign so callers always get [SELL, HOLD, BUY].
        raw_proba = self.classifier.predict_proba(X_scaled)
        class_index = {c: i for i, c in enumerate(self.classifier.classes_)}
        return np.column_stack([raw_proba[:, class_index[label]] for label in CLASS_LABELS])

    def predict_with_confidence(self, X):
        """
        Returns (predicted_return: float, signal: str, confidence: float, proba: dict)
        for a SINGLE row of features. confidence is the classifier's top
        class probability -- a principled measure of how sure the model is,
        rather than an ad-hoc agreement score across the regressors.
        """
        predicted_return = float(self.predict(X)[0])
        proba_row = self.predict_proba(X)[0]
        proba = dict(zip(CLASS_LABELS, proba_row.tolist()))
        signal = CLASS_LABELS[int(np.argmax(proba_row))]
        confidence = float(np.max(proba_row))
        return predicted_return, signal, confidence, proba

    def get_base_predictions(self, X):
        X = self._prepare_inference_features(X)
        X_scaled = self.scaler.transform(X)
        return [model.predict(X_scaled) for model in self.base_models]

    def explain(self, X, feature_names: Optional[Sequence[str]] = None) -> dict:
        """
        SHAP explanation for a SINGLE row of features, using the
        RandomForestRegressor (index 1 of base_models) as the explained
        model -- TreeExplainer is exact and fast for tree models, unlike
        KernelExplainer which would be needed for the Ridge/blended output
        and is far slower. The RandomForest is a reasonable stand-in since
        it's one-third of the blended prediction and captures non-linear
        feature interactions the linear Ridge can't.

        Returns:
          {
            "base_value": float,          # average prediction over training data
            "prediction": float,          # this row's RandomForest prediction
            "contributions": {feature_name: shap_value, ...}  # sorted by |impact|
          }
        """
        X = self._prepare_inference_features(X)
        X_scaled = self.scaler.transform(X)
        names = list(feature_names) if feature_names is not None else ENSEMBLE_FEATURE_ORDER

        rf_model = self.base_models[1]  # RandomForestRegressor
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(rf_model)

        shap_values = self._explainer.shap_values(X_scaled)
        row_shap = shap_values[0] if shap_values.ndim > 1 else shap_values

        contributions = dict(zip(names, row_shap.tolist()))
        contributions = dict(
            sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )

        # expected_value is a scalar for single-output regression in most SHAP
        # versions, but some return a length-1 array -- normalize either way.
        expected_value = self._explainer.expected_value
        if isinstance(expected_value, (np.ndarray, list)):
            expected_value = np.asarray(expected_value).reshape(-1)[0]

        return {
            "base_value": float(expected_value),
            "prediction": float(rf_model.predict(X_scaled)[0]),
            "contributions": contributions,
        }

    def save(self, filename: str = "ensemble_model.joblib"):
        path = os.path.join(self.model_dir, filename)
        joblib.dump({
            "scaler": self.scaler,
            "models": self.base_models,
            "weights": self.weights,
            "classifier": self.classifier,
        }, path)
        return path

    def load(self, filename: str = "ensemble_model.joblib"):
        path = os.path.join(self.model_dir, filename)
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.base_models = data["models"]
        self.weights = np.asarray(data["weights"], dtype=float)
        self.classifier = data.get("classifier")  # .get() so old saved files without a classifier still load
        self._explainer = None
        return self
