import os
import joblib
from django.conf import settings

_model = None  # cache

def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(settings.BASE_DIR, 'ml_model', 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        _model = joblib.load(model_path)
    return _model


def predict_command(features):
    model = get_model()
    return model.predict([features])[0]
