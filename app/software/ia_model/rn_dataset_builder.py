import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime


class RNDatasetBuilder:
    """
    Construye y agrega filas al dataset de entrenamiento
    para la red neuronal.
    """

    def __init__(self, controller):
        self.controller = controller

    # ==========================================================
    # MAIN
    # ==========================================================
    def build(self, execution_csv_path):

        try:
            df = pd.read_csv(execution_csv_path)
        except Exception as e:
            print("Error leyendo CSV de ejecución para RN:", e)
            return

        layout_id = df["layout_id"].iloc[0] if "layout_id" in df.columns else None

        total_time_real = (
            df["total_elapsed_time"].max()
            if "total_elapsed_time" in df.columns
            else df["elapsed_time"].sum()
        )

        total_wait = df["wait_time"].sum() if "wait_time" in df.columns else 0.0

        total_r1_real = df[df["robot"] == "Robot 1"]["elapsed_time"].sum()
        total_r2_real = df[df["robot"] == "Robot 2"]["elapsed_time"].sum()

        steps_r1 = len(df[df["robot"] == "Robot 1"])
        steps_r2 = len(df[df["robot"] == "Robot 2"])

        balance_real = abs(total_r1_real - total_r2_real)

        # ==========================================================
        # OBTENER AG
        # ==========================================================
        ag = getattr(self.controller, "optimized_layout", None)

        if not ag and layout_id is not None:
            try:
                layout_file = os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "layouts",
                    f"layout_{layout_id}.json"
                )
                layout_file = os.path.abspath(layout_file)

                if os.path.exists(layout_file):
                    with open(layout_file, "r") as f:
                        ag = json.load(f)
            except Exception as e:
                print("No se pudo cargar AG desde disco:", e)

        if not ag:
            ag = {
                "layout_id": layout_id or "",
                "params": {},
                "metrics": {},
                "fitness": None,
                "baseline_time": None,
                "improvement_percent": None,
                "movable_indices": [],
                "fixed_mask": []
            }

        stations = ag.get("stations", [])
        params = ag.get("params", {})

        R1 = params.get("R1", (0, 0))
        R2 = params.get("R2", (0, 0))

        (
            avg_dist_r1,
            avg_dist_r2,
            max_dist_r1,
            max_dist_r2,
            std_dist_r1,
            std_dist_r2,
            pct_r1,
            pct_r2,
            geom_balance,
            centroid_x,
            centroid_y,
            centroid_dist_r1,
            centroid_dist_r2,
            var_x,
            var_y,
            radial_dispersion
        ) = self._compute_geometry_features(stations, R1, R2)

        # ==========================================================
        # MÉTRICAS AG
        # ==========================================================
        m = ag.get("metrics", {})

        m = {
            "move_r1": m.get("move_r1", 0.0),
            "proc_r1": m.get("proc_r1", 0.0),
            "total_r1": m.get("total_r1", 0.0),
            "move_r2": m.get("move_r2", 0.0),
            "proc_r2": m.get("proc_r2", 0.0),
            "total_r2": m.get("total_r2", 0.0),
            "total_system": m.get("total_system", 0.0),
            "balance": m.get("balance", 0.0),
            "symmetry_index": m.get("symmetry_index", 0.0),
            "idle_time_estimated": m.get("idle_time_estimated", 0.0)
        }

        shape_encoding = {"BASE": 0, "S": 1, "U": 2, "L": 3}
        shape_mode = ag.get("params", {}).get("shape_mode", "BASE")
        shape_code = shape_encoding.get(shape_mode, 0)

        fitness = ag.get("fitness", None)
        baseline_time = ag.get("baseline_time", None)
        improvement_percent = ag.get("improvement_percent", None)

        estimated_system_time = m.get("total_system", 0.0)
        error_dinamico = total_time_real - estimated_system_time

        rn_row = {
            "layout_id": ag.get("layout_id", layout_id),
            "shape_mode": shape_code,
            "n_movable": len(ag.get("movable_indices", [])),
            "n_fixed": sum(ag.get("fixed_mask", [])) if ag.get("fixed_mask") else 0,
            "fitness": fitness,
            "baseline_time": baseline_time,
            "improvement_percent": improvement_percent,
            "move_r1": m["move_r1"],
            "proc_r1": m["proc_r1"],
            "total_r1": m["total_r1"],
            "move_r2": m["move_r2"],
            "proc_r2": m["proc_r2"],
            "total_r2": m["total_r2"],
            "total_system_est": m["total_system"],
            "balance_teorico": m["balance"],
            "symmetry_index": m["symmetry_index"],
            "idle_time_estimated": m["idle_time_estimated"],
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
            "total_time_real": total_time_real,
            "total_wait_real": total_wait,
            "total_r1_real": total_r1_real,
            "total_r2_real": total_r2_real,
            "steps_r1": steps_r1,
            "steps_r2": steps_r2,
            "balance_real": balance_real,
            "error_dinamico": error_dinamico,
            "generated_at": datetime.utcnow().isoformat()
        }

        self._append_row(rn_row)

    # ==========================================================
    # GEOMETRY
    # ==========================================================
    def _compute_geometry_features(self, stations, R1, R2):

        if not stations:
            return (0,) * 16

        xs = np.array([s[0] for s in stations])
        ys = np.array([s[1] for s in stations])

        dist_r1 = np.array([math.dist(s, R1) for s in stations])
        dist_r2 = np.array([math.dist(s, R2) for s in stations])

        avg_dist_r1 = dist_r1.mean()
        avg_dist_r2 = dist_r2.mean()

        max_dist_r1 = dist_r1.max()
        max_dist_r2 = dist_r2.max()

        std_dist_r1 = dist_r1.std()
        std_dist_r2 = dist_r2.std()

        closer_to_r1 = np.sum(dist_r1 < dist_r2)
        closer_to_r2 = np.sum(dist_r2 <= dist_r1)

        pct_r1 = closer_to_r1 / len(stations)
        pct_r2 = closer_to_r2 / len(stations)

        geom_balance = abs(pct_r1 - pct_r2)

        centroid_x = xs.mean()
        centroid_y = ys.mean()

        centroid_dist_r1 = math.dist((centroid_x, centroid_y), R1)
        centroid_dist_r2 = math.dist((centroid_x, centroid_y), R2)

        var_x = xs.var()
        var_y = ys.var()

        radial_dispersion = np.mean(
            [math.dist((x, y), (centroid_x, centroid_y)) for x, y in stations]
        )

        return (
            avg_dist_r1, avg_dist_r2,
            max_dist_r1, max_dist_r2,
            std_dist_r1, std_dist_r2,
            pct_r1, pct_r2,
            geom_balance,
            centroid_x, centroid_y,
            centroid_dist_r1, centroid_dist_r2,
            var_x, var_y,
            radial_dispersion
        )

    # ==========================================================
    # SAVE
    # ==========================================================
    def _append_row(self, rn_row):

        rn_csv = os.path.join(
            os.path.dirname(__file__),
            "..",
            "dataset_rn.csv"
        )
        rn_csv = os.path.abspath(rn_csv)

        try:
            df_row = pd.DataFrame([rn_row])

            if not os.path.exists(rn_csv):
                df_row.to_csv(rn_csv, index=False)
            else:
                df_row.to_csv(rn_csv, mode='a', header=False, index=False)

            print("Fila RN añadida a:", rn_csv)

        except Exception as e:
            print("Error guardando dataset RN:", e)