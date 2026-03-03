import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model


class SurrogateModel:

    def __init__(self, model_path, scaler_path):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_error(self, feature_vector):
        x = np.array(feature_vector, dtype=float).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        pred = self.model.predict(x_scaled, verbose=0)[0][0]
        return float(pred)