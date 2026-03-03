# train_surrogate.py
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



# -----------------------
# Config
# -----------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset_rn.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "surrogate_model.keras")
SCALER_OUT = os.path.join(os.path.dirname(__file__), "scaler.pkl")
HISTORY_OUT = os.path.join(os.path.dirname(__file__), "train_history.json")
RANDOM_SEED = 42

# -----------------------
# Column lists (features y target)
# -----------------------
FEATURE_COLUMNS = [
    "shape_mode",
    "n_movable",
    "n_fixed",
    "fitness",
    "baseline_time",
    "improvement_percent",
    "move_r1",
    "proc_r1",
    "total_r1",
    "move_r2",
    "proc_r2",
    "total_r2",
    "total_system_est",
    "balance_teorico",
    "symmetry_index",
    "idle_time_estimated",
    "avg_dist_r1",
    "avg_dist_r2",
    "max_dist_r1",
    "max_dist_r2",
    "std_dist_r1",
    "std_dist_r2",
    "pct_closer_r1",
    "pct_closer_r2",
    "geom_balance",
    "centroid_x",
    "centroid_y",
    "centroid_dist_r1",
    "centroid_dist_r2",
    "var_x",
    "var_y",
    "radial_dispersion"
]

TARGET_COLUMN = "error_dinamico"  # preferible
# Si no existe, crearemos: total_time_real - total_system_est

# -----------------------
# Helpers
# -----------------------
def load_data(path):
    df = pd.read_csv(path)
    return df

def prepare_df(df):
    # 1) Crear target si no existe
    if TARGET_COLUMN not in df.columns:
        if ("total_time_real" in df.columns) and ("total_system_est" in df.columns):
            df[TARGET_COLUMN] = df["total_time_real"] - df["total_system_est"]
        else:
            raise ValueError("No se encuentra target ni columnas para calcularlo (total_time_real / total_system_est).")

    # 2) Handle shape_mode: puede ser numérico o string
    if "shape_mode" in df.columns:
        if df["shape_mode"].dtype == object:
            # convertir a código (asegúrate que coincide con encoding en builder)
            mapping = {"BASE": 0, "S": 1, "U": 2, "L": 3}
            df["shape_mode"] = df["shape_mode"].map(mapping).fillna(0).astype(int)
    else:
        # si no existe, poner 0
        df["shape_mode"] = 0

    # 3) Select features (si faltan columnas, lanzar excepción clara)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el dataset necesarias para features: {missing}")

    # 4) Drop rows con NaNs en features o target (o imputar si prefieres)
    df_sel = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    before = len(df_sel)
    df_sel = df_sel.dropna()
    after = len(df_sel)
    if after < before:
        print(f"[prepare_df] Se eliminaron {before-after} filas por NaNs.")
    return df_sel

def build_model(input_dim):
    # Arquitectura recomendada para ~20-40 features y 3k samples
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # regresión
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# -----------------------
# Entrenamiento
# -----------------------
def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("[train] Cargando dataset:", DATA_PATH)
    df = load_data(DATA_PATH)
    df = prepare_df(df)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values.reshape(-1, 1)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # scaler (muy importante: guardar para inferencia)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # guardar scaler
    joblib.dump(scaler, SCALER_OUT)
    print(f"[train] Scaler guardado en: {SCALER_OUT}")

    # model
    model = build_model(input_dim=X_train_s.shape[1])
    model.summary()

    # callbacks
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    ckpt = ModelCheckpoint(MODEL_OUT, monitor='val_loss', save_best_only=True, verbose=1)

    # fit
    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_test_s, y_test),
        epochs=500,
        batch_size=32,
        callbacks=[es, ckpt],
        verbose=2
    )

    # guardar history
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(HISTORY_OUT, "w") as f:
        json.dump({"created_at": datetime.utcnow().isoformat(), "history": hist}, f)
    print(f"[train] History guardado en: {HISTORY_OUT}")

    # evaluar
    y_pred = model.predict(X_test_s).ravel()
    y_true = y_test.ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"[train] Test MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    # guardar modelo final (ModelCheckpoint ya lo guardó en el mejor epoch)
    if not os.path.exists(MODEL_OUT):
        model.save(MODEL_OUT)
    print(f"[train] Modelo guardado en: {MODEL_OUT}")

if __name__ == "__main__":
    main()