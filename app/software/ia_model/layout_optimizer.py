# layout_optimizer.py

import math
import random
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import uuid
import app.public.globals as globals
import os
from app.software.ia_model.surrogate import SurrogateModel
from app.software.robot.robot_kinematics import RobotKinematics

IK_MODEL = RobotKinematics()
# ===============================
# Utilidades
# ===============================

# ===============================
# Surrogate global (opcional)
# ===============================

SURROGATE = None

def load_surrogate():
    global SURROGATE
    if SURROGATE is None:
        base_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "ia_model"
        )
        model_path = os.path.abspath(os.path.join(base_path, "surrogate_model.keras"))
        scaler_path = os.path.abspath(os.path.join(base_path, "scaler.pkl"))

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            SURROGATE = SurrogateModel(model_path, scaler_path)
            print("Surrogate cargado correctamente.")
        else:
            print("No se encontró modelo surrogate.")

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
    penalty = 0.0
    R1 = params["R1"]
    R2 = params["R2"]

    for idx in params["robot1_stations"]:
        x, y = stations[idx]
        penalty += (x - R1[0]) ** 2

    for idx in params["robot2_stations"]:
        x, y = stations[idx]
        penalty += (y - R2[1]) ** 2

    return penalty


# ===============================
# Evaluación (ORIGINAL)
# ===============================

def evaluar_layout(layout_vars, params):

    baseline = params["baseline_time"]

    # 1️⃣ Tiempo dinámico calibrado
    metrics = compute_layout_metrics(layout_vars, params)
    total_time = metrics["total_system"]

    # 2️⃣ Penalizaciones geométricas (se mantienen)
    stations = build_full_layout(layout_vars, params)

    penalty = 0.0
    penalty += separation_penalty(
        stations,
        params["d_min"],
        params["PENALTY"]
    )

    # Penalizaciones de alcance
    for idx in params["robot1_stations"]:
        d = dist(params["R1"], stations[idx])
        if d < params["r_min"]:
            penalty += params["PENALTY"] * (params["r_min"] - d)
        elif d > params["reach"]:
            penalty += params["PENALTY"] * (d - params["reach"])

    for idx in params["robot2_stations"]:
        d = dist(params["R2"], stations[idx])
        if d < params["r_min"]:
            penalty += params["PENALTY"] * (params["r_min"] - d)
        elif d > params["reach"]:
            penalty += params["PENALTY"] * (d - params["reach"])

    # Penalización de forma
    if params["shape_mode"] == "S":
        penalty += params["shape_weight"] * shape_penalty_S(
            stations, params["R1"], params["R2"], stations[3][1]
        )
    elif params["shape_mode"] == "U":
        penalty += params["shape_weight"] * radial_penalty(
            stations,
            params["R1"],
            params["R2"],
            params["r_min"],
            params["reach"],
            stations[3][1]
        )
    elif params["shape_mode"] == "L":
        penalty += params["shape_weight"] * shape_penalty_L(
            stations, params
        )

    fitness = (total_time + penalty) / baseline

    return fitness


# ===============================
# Métricas
# ===============================

# ------------------------------
# Calibración de tiempo de movimiento
# ------------------------------
# Coeficientes obtenidos por regresión sobre tus ejecuciones:
# elapsed_s = a * distance_mm + b
# (puedes ajustar estos valores si recalibras con más datos)
MOTION_COEF_DEFAULT = {
    "Robot 1": {"a": 0.007181, "b": 1.446751},
    "Robot 2": {"a": 0.009065, "b": 0.806503}
}

def _meters_to_mm(d_m):
    return d_m * 1000.0

def compute_layout_metrics(layout_vars, params):

    stations = build_full_layout(layout_vars, params)
    tiempos = params["tiempos_estacion"]

    # Coeficientes calibrados
    A_XY = 0.012188
    A_Z  = 0.017244
    C_OFFSET = 0.299785

    # Alturas
    FEEDER_Z_MM = 110
    STATION_Z_MM = 150

    total_move_time = 0.0
    total_proc = 0.0

    # ========================
    # ROBOT 1
    # ========================
    prev_global = params["A"]
    base_x, base_y = params["R1"]

    for idx in params["robot1_stations"]:

        # convertir a coordenadas locales
        xg, yg = stations[idx]
        xr = xg - base_x
        yr = yg - base_y

        pxr = prev_global[0] - base_x
        pyr = prev_global[1] - base_y

        dx = xr - pxr
        dy = yr - pyr

        dist_xy_m = math.sqrt(dx**2 + dy**2)
        dist_xy_mm = dist_xy_m * 1000.0

        total_move_time += A_XY * dist_xy_mm + C_OFFSET

        # vertical fijo por estación
        if idx == 0 or idx == 8:
            total_move_time += A_Z * (2 * FEEDER_Z_MM)
        else:
            total_move_time += A_Z * (2 * STATION_Z_MM)

        total_proc += tiempos[idx]

        prev_global = stations[idx]

    # ========================
    # ROBOT 2
    # ========================
    base_x, base_y = params["R2"]
    prev_global = stations[params["robot2_stations"][0]]

    for idx in params["robot2_stations"][1:]:

        xg, yg = stations[idx]
        xr = xg - base_x
        yr = yg - base_y

        pxr = prev_global[0] - base_x
        pyr = prev_global[1] - base_y

        dx = xr - pxr
        dy = yr - pyr

        dist_xy_m = math.sqrt(dx**2 + dy**2)
        dist_xy_mm = dist_xy_m * 1000.0

        total_move_time += A_XY * dist_xy_mm + C_OFFSET

        if idx == 0 or idx == 8:
            total_move_time += A_Z * (2 * FEEDER_Z_MM)
        else:
            total_move_time += A_Z * (2 * STATION_Z_MM)

        total_proc += tiempos[idx]

        prev_global = stations[idx]

    total_system = total_move_time + total_proc

    return {
        "move_r1": None,
        "proc_r1": None,
        "total_r1": None,
        "move_r2": None,
        "proc_r2": None,
        "total_r2": None,
        "total_system": total_system,
        "balance": 0.0,
        "symmetry_index": 0.0,
        "idle_time_estimated": 0.0
    }



# ===============================
# GA helpers
# ===============================

def random_layout(bounds_R1, bounds_R2, params):
    layout = []
    for i, is_fixed in enumerate(params["fixed_mask"]):
        if not is_fixed:
            bounds = bounds_R1 if i in params["robot1_stations"] else bounds_R2
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


def evaluar_layout_joint_based(layout_vars, params):

    stations = build_full_layout(layout_vars, params)

    baseline = params["baseline_time"]
    tiempos = params["tiempos_estacion"]

    K = 0.038161
    OFFSET = 0.633842

    total_time = 0.0
    penalty = 0.0

    # Penalización geométrica básica (mantenerla)
    penalty += separation_penalty(
        stations,
        params["d_min"],
        params["PENALTY"]
    )

    # ==========================
    # ROBOT 1
    # ==========================
    base_x, base_y = params["R1"]
    prev_joints = None

    for idx in params["robot1_stations"]:

        xg, yg = stations[idx]

        xr = (xg - base_x) * 1000.0
        yr = (yg - base_y) * 1000.0

        # PS fijo
        ps_z = 234.635322

        # orientación como en tu generador
        j0_est = math.degrees(math.atan2(yr, xr))
        is_feeder = (idx == 0 or idx == 8)

        global_orientation = 90 if is_feeder else 0
        beta_orientation = global_orientation - j0_est

        res = IK_MODEL.xyz_to_joint(
            [xr, yr, ps_z, -95, beta_orientation]
        )

        if res is None or res["status"] != 0:
            return 9999.0  # layout inválido

        joints = res["joint"]

        if prev_joints is not None:
            delta = max(abs(joints[i] - prev_joints[i]) for i in range(5))
            total_time += K * delta + OFFSET

        total_time += tiempos[idx]
        prev_joints = joints

    # ==========================
    # ROBOT 2
    # ==========================
    base_x, base_y = params["R2"]
    prev_joints = None

    for idx in params["robot2_stations"]:

        xg, yg = stations[idx]

        xr = (xg - base_x) * 1000.0
        yr = (yg - base_y) * 1000.0

        ps_z = 234.635322

        j0_est = math.degrees(math.atan2(yr, xr))
        is_feeder = (idx == 0 or idx == 8)

        global_orientation = 90 if is_feeder else 0
        beta_orientation = global_orientation - j0_est

        res = IK_MODEL.xyz_to_joint(
            [xr, yr, ps_z, -95, beta_orientation]
        )

        if res is None or res["status"] != 0:
            return 9999.0

        joints = res["joint"]

        if prev_joints is not None:
            delta = max(abs(joints[i] - prev_joints[i]) for i in range(5))
            total_time += K * delta + OFFSET

        total_time += tiempos[idx]
        prev_joints = joints

    fitness = (total_time + penalty) / baseline
    return fitness

# ===============================
# GA PURO
# ===============================

def run_ga(params, bounds_R1, bounds_R2,
           POP_SIZE=100,
           N_GEN=80):

    movable_indices = [i for i, is_fixed in enumerate(params["fixed_mask"]) if not is_fixed]

    population = [random_layout(bounds_R1, bounds_R2, params) for _ in range(POP_SIZE)]

    best_history = []
    avg_history = []

    baseline = params["baseline_time"]

    for gen in range(N_GEN):

        fitnesses = [evaluar_layout_joint_based(ind, params) for ind in population]

        best_fitness = min(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)

        best_history.append(best_fitness)
        avg_history.append(avg_fitness)

        elite_size = max(1, int(0.1 * POP_SIZE))
        elite_idx = sorted(range(len(population)), key=lambda i: fitnesses[i])[:elite_size]
        elite = [population[i] for i in elite_idx]

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

    fitnesses = [evaluar_layout(ind, params) for ind in population]
    best_idx = fitnesses.index(min(fitnesses))
    best_layout = population[best_idx]
    best_fitness = fitnesses[best_idx]

    best_stations = build_full_layout(best_layout, params)

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
# Layout helpers
# ===============================

def build_full_layout(layout_vars, params):
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


# ===============================
# Defaults
# ===============================

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

    bounds_R1 = {"x": (R1[0] - reach, R1[0] + reach),
                 "y": (R1[1] - reach, R1[1] + reach)}

    bounds_R2 = {"x": (R2[0] - reach, R2[0] + reach),
                 "y": (R2[1] - reach, R2[1] + reach)}

    return bounds_R1, bounds_R2


if __name__ == "__main__":
    params = default_params()
    bounds_R1, bounds_R2 = default_bounds(params)
    result = run_ga(params, bounds_R1, bounds_R2, POP_SIZE=200, N_GEN=120)
    print("\nMejor fitness:", result["fitness"])
    print("\nMejores Estaciones", result["stations"])