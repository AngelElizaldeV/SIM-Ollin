import os
import pandas as pd
import numpy as np
import joblib

SEQ_LEN = 64

def process_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
    except:
        return None

    if len(df) != SEQ_LEN:
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

    total_time = df["total_elapsed_time"].max()

    return sequence, total_time


def build_dataset(csv_folder, output_file):

    sequences = []
    times = []

    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            path = os.path.join(csv_folder, file)
            result = process_csv(path)
            if result:
                seq, total_time = result
                if total_time > 90:  # eliminar outliers irreales
                    sequences.append(seq)
                    times.append(total_time)

    X = np.array(sequences)
    y = np.array(times)

    print("Total secuencias:", X.shape)
    print("Total tiempos:", y.shape)

    joblib.dump((X, y), output_file)
    print("Dataset secuencial guardado:", output_file)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_folder = "app/public/executions"
    output_file = os.path.join(BASE_DIR, "sequence_dataset.pkl")

    build_dataset(csv_folder, output_file)