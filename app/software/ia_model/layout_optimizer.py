# layout_optimizer.py
import math
import random
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import uuid
import app.public.globals as globals
import copy
import os

# carga segura de artefactos de preprocessing
import joblib
import pandas as pd
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))

# intenta cargar scaler y feature_columns (si existen)
try:
    scaler = joblib.load(os.path.join(base_path, "scaler_v1.pkl"))
except Exception as e:
    print("Warning: no se pudo cargar scaler_v1.pkl:", e)
    scaler = None

try:
    feature_columns = joblib.load(os.path.join(base_path, "feature_columns.pkl"))
except Exception as e:
    print("Warning: no se pudo cargar feature_columns.pkl:", e)
    feature_columns = []

# Surrogate (modelo TF) — NO importamos tensorflow aquí para evitar errores de runtime en la UI.
_SURROGATE = None
_SURROGATE_LOADED = False

def get_surrogate():
    global _SURROGATE, _SURROGATE_LOADED

    if _SURROGATE_LOADED:
        return _SURROGATE

    _SURROGATE_LOADED = True

    try:
        import os
        import tensorflow as tf

        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "surrogate_v1.keras")

        _SURROGATE = tf.keras.models.load_model(model_path)
        print("Surrogate cargado correctamente desde:", model_path)

    except Exception as e:
        print("No se pudo cargar surrogate:", e)
        _SURROGATE = None

    return _SURROGATE

# ===============================
# Utilidades
# ===============================

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def separation_penalty(stations, d_min, PENALTY):
    penalty = 0.0
    for i, j in itertools.combinations(range(len(stations)), 2):
        d = dist(stations[i], stations[j])
        if d < d_min:
            penalty += PENALTY * (d_min - d)
    return penalty


def radial_penalty(stations, R1, R2, r_min, r_max, y_split):
    penalty = 0.0
    r_target = 0.5 * (r_min + r_max)

    for x, y in stations:
        if y <= y_split:
            r = dist((x, y), R1)
        else:
            r = dist((x, y), R2)

        penalty += (r - r_target) ** 2

    return penalty


def shape_penalty_S(stations, R1, R2, y_split):
    penalty = 0.0

    for x, y in stations:
        if y <= y_split:
            penalty += max(0, x - R1[0]) ** 2
        else:
            penalty += max(0, R2[0] - x) ** 2

    return penalty


def shape_penalty_L(stations, params):
    """
    Forma L tipo A:
    - Robot 1: estaciones alineadas verticalmente (x ≈ R1.x)
    - Robot 2: estaciones alineadas horizontalmente (y ≈ R2.y)
    """

    penalty = 0.0

    R1 = params["R1"]
    R2 = params["R2"]

    # Robot 1 → vertical
    for idx in params["robot1_stations"]:
        x, y = stations[idx]
        penalty += (x - R1[0]) ** 2

    # Robot 2 → horizontal
    for idx in params["robot2_stations"]:
        x, y = stations[idx]
        penalty += (y - R2[1]) ** 2

    return penalty


# ===============================
# Evaluación
# ===============================

def evaluar_layout(layout_vars, params):
    """Evaluación original (sin surrogate) — devuelve fitness clásico."""
    stations = build_full_layout(layout_vars, params)

    R1 = params["R1"]
    R2 = params["R2"]

    reach = params["reach"]
    r_min = params["r_min"]

    tiempos = params["tiempos_estacion"]
    baseline = params["baseline_time"]

    d_min = params["d_min"]
    PENALTY = params["PENALTY"]

    total_time = 0.0
    penalty = 0.0

    # Penalización separación
    penalty += separation_penalty(stations, d_min, PENALTY)

    # ROBOT 1
    prev = params["A"]
    for idx in params["robot1_stations"]:
        d = dist(R1, stations[idx])

        if d < r_min:
            penalty += PENALTY * (r_min - d)
        elif d > reach:
            penalty += PENALTY * (d - reach)

        total_time += dist(prev, stations[idx])
        total_time += tiempos[idx]
        prev = stations[idx]

    # ROBOT 2
    prev = stations[params["robot2_stations"][0]]
    for idx in params["robot2_stations"][1:]:
        d = dist(R2, stations[idx])

        if d < r_min:
            penalty += PENALTY * (r_min - d)
        elif d > reach:
            penalty += PENALTY * (d - reach)

        total_time += dist(prev, stations[idx])
        total_time += tiempos[idx]
        prev = stations[idx]

    # SHAPE MODE
    if params["shape_mode"] == "S":
        penalty += params["shape_weight"] * shape_penalty_S(stations, R1, R2, stations[3][1])
    elif params["shape_mode"] == "U":
        penalty += params["shape_weight"] * radial_penalty(stations, R1, R2, r_min, reach, stations[3][1])
    elif params["shape_mode"] == "L":
        penalty += params["shape_weight"] * shape_penalty_L(stations, params)

    fitness = (total_time + penalty) / baseline
    return fitness


def compute_layout_metrics(layout_vars, params):
    """
    Calcula las métricas teóricas que ya usabas: move/proc/total por robot,
    balance, symmetry_index, idle_time_estimated, total_system.
    """
    stations = build_full_layout(layout_vars, params)

    R1 = params["R1"]
    R2 = params["R2"]

    tiempos = params["tiempos_estacion"]

    # ROBOT 1
    prev = params["A"]
    move_r1 = 0.0
    proc_r1 = 0.0
    for idx in params["robot1_stations"]:
        move_r1 += dist(prev, stations[idx])
        proc_r1 += tiempos[idx]
        prev = stations[idx]
    total_r1 = move_r1 + proc_r1

    # ROBOT 2
    prev = stations[params["robot2_stations"][0]]
    move_r2 = 0.0
    proc_r2 = 0.0
    for idx in params["robot2_stations"][1:]:
        move_r2 += dist(prev, stations[idx])
        proc_r2 += tiempos[idx]
        prev = stations[idx]
    total_r2 = move_r2 + proc_r2

    total_system = total_r1 + total_r2
    balance = abs(total_r1 - total_r2)
    symmetry_index = balance / total_system if total_system > 0 else 0.0
    idle_time = balance

    return {
        "move_r1": move_r1,
        "proc_r1": proc_r1,
        "total_r1": total_r1,
        "move_r2": move_r2,
        "proc_r2": proc_r2,
        "total_r2": total_r2,
        "total_system": total_system,
        "balance": balance,
        "symmetry_index": symmetry_index,
        "idle_time_estimated": idle_time
    }


# ===============================
# GA helpers
# ===============================

def random_layout(bounds_R1, bounds_R2, params):
    layout = []
    for i, is_fixed in enumerate(params["fixed_mask"]):
        if not is_fixed:
            if i in params["robot1_stations"]:
                bounds = bounds_R1
            else:
                bounds = bounds_R2
            x = random.uniform(bounds["x"][0], bounds["x"][1])
            y = random.uniform(bounds["y"][0], bounds["y"][1])
            layout.extend([x, y])
    return layout


def crossover(p1, p2):
    point = random.randint(1, len(p1)-2)
    return p1[:point] + p2[point:]


def mutate(layout, sigma=0.02):
    if len(layout) == 0:
        return
    i = random.randint(0, len(layout)-1)
    layout[i] += random.gauss(0, sigma)


# ===============================
# Surrogate feature builder + evaluator
# ===============================

def build_feature_vector(layout_vars, params):
    """
    Construye el vector (1xN) de features exactamente igual a como lo hiciste para entrenar:
    - reconstruye estaciones
    - calcula métricas (compute_layout_metrics)
    - calcula geometría (avg/max/std/distancias, centroid, var, radial_dispersion, pct_closer)
    - arma DataFrame, one-hot de shape_mode y completa/ordena según feature_columns
    - escala con 'scaler' si está disponible
    Devuelve numpy array con shape (1, n_features) listo para model.predict.
    """

    stations = build_full_layout(layout_vars, params)

    # geometría y distancias
    if stations:
        xs = np.array([s[0] for s in stations])
        ys = np.array([s[1] for s in stations])
        # distancias a robots
        R1 = params.get("R1", (0, 0))
        R2 = params.get("R2", (0, 0))
        dist_r1 = np.array([math.dist(s, R1) for s in stations])
        dist_r2 = np.array([math.dist(s, R2) for s in stations])
        avg_dist_r1 = float(np.mean(dist_r1))
        avg_dist_r2 = float(np.mean(dist_r2))
        max_dist_r1 = float(np.max(dist_r1))
        max_dist_r2 = float(np.max(dist_r2))
        std_dist_r1 = float(np.std(dist_r1))
        std_dist_r2 = float(np.std(dist_r2))
        closer_to_r1 = int(np.sum(dist_r1 < dist_r2))
        closer_to_r2 = int(np.sum(dist_r2 <= dist_r1))
        pct_r1 = closer_to_r1 / len(stations)
        pct_r2 = closer_to_r2 / len(stations)
        geom_balance = abs(pct_r1 - pct_r2)
        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))
        centroid_dist_r1 = float(math.dist((centroid_x, centroid_y), R1))
        centroid_dist_r2 = float(math.dist((centroid_x, centroid_y), R2))
        var_x = float(np.var(xs))
        var_y = float(np.var(ys))
        radial_dispersion = float(np.mean([math.dist((x, y), (centroid_x, centroid_y)) for x, y in stations]))
    else:
        avg_dist_r1 = avg_dist_r2 = max_dist_r1 = max_dist_r2 = 0.0
        std_dist_r1 = std_dist_r2 = 0.0
        pct_r1 = pct_r2 = geom_balance = 0.0
        centroid_x = centroid_y = centroid_dist_r1 = centroid_dist_r2 = 0.0
        var_x = var_y = radial_dispersion = 0.0

    # métricas del AG
    metrics = compute_layout_metrics(layout_vars, params)

    # ensamblar diccionario de features (manteniendo nombres usados en tu CSV)
    feature_dict = {
        "shape_mode": params.get("shape_mode", "BASE"),
        "n_movable": len([i for i, f in enumerate(params.get("fixed_mask", [])) if not f]),
        "n_fixed": sum(params.get("fixed_mask", [])) if params.get("fixed_mask") else 0,
        "fitness": float(evaluar_layout(layout_vars, params)),
        "baseline_time": float(params.get("baseline_time", 0.0)),
        # métricas estimadas por AG
        "move_r1": float(metrics.get("move_r1", 0.0)),
        "proc_r1": float(metrics.get("proc_r1", 0.0)),
        "total_r1": float(metrics.get("total_r1", 0.0)),
        "move_r2": float(metrics.get("move_r2", 0.0)),
        "proc_r2": float(metrics.get("proc_r2", 0.0)),
        "total_r2": float(metrics.get("total_r2", 0.0)),
        "total_system_est": float(metrics.get("total_system", 0.0)),
        "balance_teorico": float(metrics.get("balance", 0.0)),
        "symmetry_index": float(metrics.get("symmetry_index", 0.0)),
        "idle_time_estimated": float(metrics.get("idle_time_estimated", 0.0)),
        # geometría calculada
        "avg_dist_r1": avg_dist_r1,
        "avg_dist_r2": avg_dist_r2,
        "max_dist_r1": max_dist_r1,
        "max_dist_r2": max_dist_r2,
        "std_dist_r1": std_dist_r1,
        "std_dist_r2": std_dist_r2,
        "pct_closer_r1": pct_r1,
        "pct_closer_r2": pct_r2,
        "geom_balance": geom_balance,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "centroid_dist_r1": centroid_dist_r1,
        "centroid_dist_r2": centroid_dist_r2,
        "var_x": var_x,
        "var_y": var_y,
        "radial_dispersion": radial_dispersion,
    }

    # DataFrame temporario
    df_temp = pd.DataFrame([feature_dict])

    # One-hot exacto como en entrenamiento
    df_temp = pd.get_dummies(df_temp, columns=["shape_mode"], drop_first=False)

    # Agregar columnas faltantes (con 0) y ordenar según feature_columns
    if feature_columns:
        for col in feature_columns:
            if col not in df_temp.columns:
                df_temp[col] = 0
        # asegúrate de ordenar (si feature_columns no tiene exactamente todas, filtrar)
        cols_present = [c for c in feature_columns if c in df_temp.columns]
        df_temp = df_temp[cols_present]
    else:
        # si no hay feature_columns guardadas, usar columnas actuales del df_temp
        pass

    # Escalar
    if scaler is not None:
        try:
            X_scaled = scaler.transform(df_temp)
        except Exception as e:
            # si falla transform con nombres, convertir a numpy y escalar manualmente si es posible
            try:
                X_scaled = scaler.transform(df_temp.values)
            except Exception as e2:
                print("Warning: scaler.transform falló:", e, e2)
                X_scaled = df_temp.values
    else:
        X_scaled = df_temp.values

    return X_scaled


def evaluar_layout_surrogate(layout_vars, params):
    """
    Evalúa con surrogate si está disponible; si no, vuelve a la evaluación directa.
    Devuelve fitness (predicted_time / baseline).
    """
    model = get_surrogate()
    if model is None:
        # fallback seguro
        return evaluar_layout(layout_vars, params)

    try:
        X = build_feature_vector(layout_vars, params)
        # model.predict espera shape (n_samples, n_features)
        pred = model.predict(X, verbose=0)
        # soporte que retorne array (1,) o (1,1)
        predicted_time = float(np.asarray(pred).reshape(-1)[0])
        # sanity
        if predicted_time <= 0:
            predicted_time = max(predicted_time, 0.0)
    except Exception as e:
        print("Error usando surrogate, fallback a evaluar_layout:", e)
        return evaluar_layout(layout_vars, params)

    baseline = params.get("baseline_time", 1.0)
    fitness = predicted_time / baseline
    return fitness


# ===============================
# GA principal (usa surrogate si está disponible)
# ===============================

def run_ga(params, bounds_R1, bounds_R2,
           POP_SIZE=300, N_GEN=150):
    movable_indices = [i for i, is_fixed in enumerate(params["fixed_mask"]) if not is_fixed]

    # Inicialización población
    population = [random_layout(bounds_R1, bounds_R2, params) for _ in range(POP_SIZE)]

    best_history = []
    avg_history = []

    # determinar si hay surrogate disponible (pero no forzar crash)
    surrogate_model = get_surrogate()
    use_surrogate = surrogate_model is not None
    print("USANDO SURROGATE:", use_surrogate)

    # Evolución
    for gen in range(N_GEN):
        if use_surrogate:
            fitnesses = [evaluar_layout_surrogate(ind, params) for ind in population]
        else:
            fitnesses = [evaluar_layout(ind, params) for ind in population]

        best_fitness = min(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_history.append(best_fitness)
        avg_history.append(avg_fitness)

        # Selección elite (asegurarnos que nunca sea 0)
        elite_size = max(1, int(0.1 * POP_SIZE))
        elite_idx = sorted(range(len(population)), key=lambda i: fitnesses[i])[:elite_size]
        elite = [population[i] for i in elite_idx]

        # Nueva población
        new_population = elite.copy()
        while len(new_population) < POP_SIZE:
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            child = crossover(p1, p2)
            mutate(child)
            new_population.append(child)

        population = new_population

        if gen % 10 == 0:
            print(f"Gen {gen} | Mejor fitness: {best_fitness:.4f}")

    # Resultado final (aquí usamos la evaluación real para el desglose final)
    fitnesses = [evaluar_layout(ind, params) for ind in population]
    best_idx = fitnesses.index(min(fitnesses))
    best_layout = population[best_idx]
    best_fitness = fitnesses[best_idx]

    best_stations = build_full_layout(best_layout, params)
    baseline = params["baseline_time"]
    total_time = best_fitness * baseline
    delta_time = total_time - baseline
    improvement_percent = (1 - best_fitness) * 100
    metrics = compute_layout_metrics(best_layout, params)

    layout_id = uuid.uuid4().hex[:8]
    globals.current_layout_id = layout_id

    return {
        "layout_id": layout_id,
        "layout_vars": best_layout,
        "stations": best_stations,
        "movable_indices": movable_indices,
        "fixed_mask": params["fixed_mask"],
        "params": params,
        "fitness": best_fitness,
        "total_time": total_time,
        "baseline_time": baseline,
        "delta_time": delta_time,
        "improvement_percent": improvement_percent,
        "best_history": best_history,
        "avg_history": avg_history,
        "metrics": metrics
    }


# ===============================
# Plot helpers (sin cambios lógicos)
# ===============================

def plot_layout_on_axis(ax, best_layout, params):
    stations = build_full_layout(best_layout, params)
    R1 = params["R1"]
    R2 = params["R2"]
    reach = params["reach"]
    r_min = params["r_min"]
    from matplotlib.patches import Circle
    ax.clear()
    ax.add_patch(Circle(R1, reach, fill=False, linestyle="--", alpha=0.5))
    ax.add_patch(Circle(R1, r_min, fill=False, linestyle=":", alpha=0.5))
    ax.add_patch(Circle(R2, reach, fill=False, linestyle="--", alpha=0.5))
    ax.add_patch(Circle(R2, r_min, fill=False, linestyle=":", alpha=0.5))
    for i, (x, y) in enumerate(stations):
        if params["fixed_mask"][i]:
            ax.scatter(x, y, c="red", s=120, marker="s")
        else:
            ax.scatter(x, y, c="blue")
        ax.text(x + 0.01, y + 0.01, f"E{i+1}")
    ax.scatter(*R1, c="green", s=150)
    ax.scatter(*R2, c="purple", s=150)
    ax.set_aspect("equal")
    ax.grid(True)


def plot_layout(best_layout, params):
    stations = build_full_layout(best_layout, params)
    R1 = params["R1"]
    R2 = params["R2"]
    reach = params["reach"]
    r_min = params["r_min"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Circle(R1, reach, fill=False, linestyle="--", alpha=0.5, label="Reach R1"))
    ax.add_patch(Circle(R1, r_min, fill=False, linestyle=":", alpha=0.5))
    ax.add_patch(Circle(R2, reach, fill=False, linestyle="--", alpha=0.5, label="Reach R2"))
    ax.add_patch(Circle(R2, r_min, fill=False, linestyle=":", alpha=0.5))
    for i, (x, y) in enumerate(stations):
        if params["fixed_mask"][i]:
            ax.scatter(x, y, c="red", s=120, marker="s")
        else:
            ax.scatter(x, y, c="blue")
        ax.text(x + 0.01, y + 0.01, f"E{i+1}")
    ax.scatter(*R1, c="green", s=150, label="Robot 1")
    ax.scatter(*R2, c="purple", s=150, label="Robot 2")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    ax.set_title("Layout optimizado")
    plt.show()


def plot_convergence(best_history, avg_history):
    plt.figure(figsize=(7,5))
    plt.plot(best_history, label="Mejor fitness")
    plt.plot(avg_history, label="Fitness promedio", linestyle="--")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Convergencia del Algoritmo Genético")
    plt.grid(True)
    plt.legend()
    plt.show()


def build_full_layout(layout_vars, params):
    """
    Reconstruye estaciones combinando fijas + variables del GA (igual que antes).
    """
    stations = []
    var_index = 0
    for i, is_fixed in enumerate(params["fixed_mask"]):
        if is_fixed:
            stations.append(params["stations_base"][i])
        else:
            x = layout_vars[var_index]
            y = layout_vars[var_index+1]
            stations.append((x, y))
            var_index += 2
    return stations


def default_params():
    return {
        "R1": (0.5325, 0.65),
        "R2": (0.5325, 1.2),
        "A": (0.66, 0.44),
        "reach": 0.381,
        "r_min": 0.20,
        "stations_base": [
            (0.61, 0.39),
            (0.8, 0.525),
            (0.8, 0.655),
            (0.8, 0.79),
            (0.7, 0.93),
            (0.8, 1.06),
            (0.8, 1.19),
            (0.8, 1.325),
            (0.61, 1.45)
        ],
        "fixed_mask": [True, False, False, False, True, False, False, False, True],
        "robot1_stations": [0,1,2,3,4],
        "robot2_stations": [4,5,6,7,8],
        "shape_mode": "BASE",
        "shape_weight": 1.0,
        "tiempos_estacion": [0,8,8,5,10,5,4,4,0],
        "baseline_time": 120.0,
        "d_min": 0.15,
        "PENALTY": 1000.0
    }


def default_bounds(params):
    R1 = params["R1"]
    R2 = params["R2"]
    reach = params["reach"]
    bounds_R1 = {"x": (R1[0] - reach, R1[0] + reach), "y": (R1[1] - reach, R1[1] + reach)}
    bounds_R2 = {"x": (R2[0] - reach, R2[0] + reach), "y": (R2[1] - reach, R2[1] + reach)}
    return bounds_R1, bounds_R2


# Si ejecutas este archivo como main, corre un GA de prueba (igual que antes).
if __name__ == "__main__":
    params = default_params()
    bounds_R1, bounds_R2 = default_bounds(params)
    result = run_ga(params, bounds_R1, bounds_R2, POP_SIZE=20, N_GEN=10)
    print("\nMejor fitness:", result["fitness"])
    print("\nMejores Estaciones", result["stations"])
    plot_layout(result["layout_vars"], params)
    plot_convergence(result["best_history"], result["avg_history"])