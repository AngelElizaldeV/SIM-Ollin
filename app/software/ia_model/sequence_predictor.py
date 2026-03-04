import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sequence_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "sequence_scaler.pkl")

SEQ_LEN = 64


class SequencePredictor:

    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def _build_tensor_from_csv(self, csv_path):

        if not os.path.exists(csv_path):
            return None

        if os.path.getsize(csv_path) == 0:
            return None

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return None

        if df.empty:
            return None

        df = df.sort_values("step")

        features = [
            "j_cmd_0",
            "j_cmd_1",
            "j_cmd_2",
            "j_cmd_3",
            "j_cmd_4",
            "elapsed_time",
            "wait_time",
            "latency_ms"
        ]

        for col in features:
            if col not in df.columns:
                return None

        sequence = df[features].values.astype(np.float32)

        # 🔥 Escalado correcto
        try:
            seq_scaled = self.scaler.transform(sequence)
        except Exception:
            return None

        n_feat = seq_scaled.shape[1]
        current_len = seq_scaled.shape[0]

        # -----------------------------
        # Ajustar longitud a 64
        # -----------------------------
        if current_len > SEQ_LEN:
            seq_scaled = seq_scaled[:SEQ_LEN]

        elif current_len < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - current_len, n_feat))
            seq_scaled = np.vstack([seq_scaled, pad])

        # 🔥 reshape FINAL
        seq_scaled = seq_scaled.reshape(1, SEQ_LEN, n_feat)

        return seq_scaled

    def predict_from_csv(self, csv_path):

        tensor = self._build_tensor_from_csv(csv_path)

        if tensor is None:
            return None

        try:
            preds = self.model.predict(tensor, verbose=0)
        except Exception:
            return None

        dynamic_index = float(preds[0][0] * 100)
        ineff_risk = float(preds[1][0] * 100)

        return {
            "dynamic_index": dynamic_index,
            "ineff_risk": ineff_risk
        }