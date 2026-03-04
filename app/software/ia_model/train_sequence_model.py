import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "sequence_dataset.pkl")

def create_labels(times):

    p70 = np.percentile(times, 70)

    # Score continuo normalizado (0 = peor, 1 = mejor)
    score = 1 - (times - times.min()) / (times.max() - times.min())

    # Ineficiencia binaria
    ineff = (times > p70).astype(int)

    print("\nPercentil 70 usado para ineficiencia:", p70)
    print("Distribución ineficiencia:", np.bincount(ineff))

    return score, ineff


def main():

    X, times = joblib.load(DATA_PATH)

    n_samples, seq_len, n_feat = X.shape
    X_reshaped = X.reshape(-1, n_feat)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, seq_len, n_feat)

    score, ineff = create_labels(times)

    X_train, X_test, y_score_train, y_score_test, y_ineff_train, y_ineff_test = train_test_split(
        X_scaled, score, ineff,
        test_size=0.2,
        random_state=42
    )

    inputs = Input(shape=(seq_len, n_feat))

    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dense(32, activation="relu")(x)

    output_score = Dense(1, activation="sigmoid", name="score_output")(x)
    output_ineff = Dense(1, activation="sigmoid", name="ineff_output")(x)

    model = Model(inputs=inputs,
                  outputs=[output_score, output_ineff])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            "score_output": "mse",
            "ineff_output": "binary_crossentropy"
        },
        metrics={
            "score_output": "mae",
            "ineff_output": "accuracy"
        }
    )

    model.fit(
        X_train,
        {
            "score_output": y_score_train,
            "ineff_output": y_ineff_train
        },
        validation_data=(
            X_test,
            {
                "score_output": y_score_test,
                "ineff_output": y_ineff_test
            }
        ),
        epochs=40,
        batch_size=32,
        verbose=2
    )

    model.save(os.path.join(BASE_DIR, "sequence_model.keras"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "sequence_scaler.pkl"))

    print("\nModelo final secuencial entrenado y guardado.")


if __name__ == "__main__":
    main()