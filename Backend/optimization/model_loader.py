"""
Shared, cached loader for the trained ensemble model.

This was previously defined inline in app.py. It's extracted here because
Step 5 adds a second consumer (routes/dashboard_routes.py) that also needs
the model -- importing it directly from app.py would create a circular
import (app.py registers the dashboard blueprint, the blueprint would need
to import from app.py). Both app.py and dashboard_routes.py now import
get_ensemble_model() from here instead.
"""

import os
from optimization.ensemble import SimpleEnsembleModel

ENSEMBLE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ensemble_model.joblib")
ENSEMBLE_MODEL_PATH = os.path.normpath(ENSEMBLE_MODEL_PATH)
_ENSEMBLE_MODEL = None


def get_ensemble_model():
    global _ENSEMBLE_MODEL
    if _ENSEMBLE_MODEL is None:
        if not os.path.exists(ENSEMBLE_MODEL_PATH):
            raise FileNotFoundError("Ensemble model not found. Generate and train the model first.")
        model = SimpleEnsembleModel(model_dir=os.path.dirname(ENSEMBLE_MODEL_PATH))
        model.load(os.path.basename(ENSEMBLE_MODEL_PATH))
        _ENSEMBLE_MODEL = model
    return _ENSEMBLE_MODEL
