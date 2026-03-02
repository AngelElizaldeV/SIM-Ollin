# batch_orchestrator.py
import os
import time
import copy
import threading
import random
import math

from PyQt6.QtCore import QObject, QThread, pyqtSignal

import layout_optimizer as lo

class BatchWorker(QThread):
    progress = pyqtSignal(int, int)   # (current_run, total_runs)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, controller, tab, n_runs=10, vary_bounds=False, speed_multiplier=10.0, timeout_per_run=600):
        super().__init__()
        self.controller = controller  # instancia RobotController
        self.tab = tab                # instancia LayoutOptimizerDesignerTab
        self.n_runs = int(n_runs)
        self.vary_bounds = bool(vary_bounds)
        self.speed_multiplier = float(speed_multiplier)
        self._stop = False
        self.timeout_per_run = timeout_per_run

    def stop(self):
        self._stop = True

    def _perturb_params(self, params):
        """Perturbación pequeña (compatibilidad con la versión anterior)."""
        p = copy.deepcopy(params)
        jitter = 0.01  # ±1 cm
        for i, fixed in enumerate(p.get("fixed_mask", [])):
            if not fixed:
                x, y = p["stations_base"][i]
                p["stations_base"][i] = (
                    x + random.uniform(-jitter, jitter),
                    y + random.uniform(-jitter, jitter)
                )
        return p

    def _randomize_params(self, params, aggressive=False):
        """
        Genera una copia de params con variaciones amplias:
        - mantiene siempre fixed_mask[4] = True (estación 5 fija).
        - puede mover R1/R2 ocasionalmente.
        - reasignar estaciones entre robots en algunos casos.
        - variar tiempos, reach, r_min, shape_mode, baseline_time, shape_weight.
        - variar qué estaciones son fijas (respetando index 4 fijo).
        - si aggressive=True, las variaciones son mayores (usar cuando vary_bounds True).
        """
        p = copy.deepcopy(params)

        # factores de variación
        if aggressive:
            pos_jitter = 0.05   # ±5 cm
            time_jitter_frac = 0.25  # ±25%
            reach_jitter = 0.15  # ±15%
            robot_move_prob = 0.5
            swap_robot_assign_prob = 0.35
            unfix_prob = 0.35
        else:
            pos_jitter = 0.01   # ±1 cm
            time_jitter_frac = 0.10  # ±10%
            reach_jitter = 0.05  # ±5%
            robot_move_prob = 0.2
            swap_robot_assign_prob = 0.12
            unfix_prob = 0.12

        # --- Mantener backup de estaciones base original (por si se necesita)
        original_stations = params.get("stations_base", [])

        # 1) Variar posiciones de estaciones no fijas
        new_stations = []
        for i, (x, y) in enumerate(p["stations_base"]):
            if p["fixed_mask"][i]:
                # mantener fijas (pero permitir mover robots, no estaciones fijas)
                new_stations.append((x, y))
            else:
                nx = x + random.uniform(-pos_jitter, pos_jitter)
                ny = y + random.uniform(-pos_jitter, pos_jitter)
                new_stations.append((nx, ny))

        p["stations_base"] = new_stations

        # 2) Asegurar que ESTACIÓN 5 (index 4) esté fija siempre
        if len(p.get("fixed_mask", [])) >= 5:
            p["fixed_mask"] = list(p["fixed_mask"])  # asegurar mutabilidad
            p["fixed_mask"][4] = True

        # 3) Posicionar robots aleatoriamente en algunos casos (mover R1/R2)
        #    Mantenerlos lo suficientemente cerca de la mesa (evitar colocaciones imposibles)
        def jitter_robot(coord):
            x, y = coord
            # mover con probabilidad
            if random.random() < robot_move_prob:
                nx = x + random.uniform(-0.08, 0.08) if aggressive else x + random.uniform(-0.03, 0.03)
                ny = y + random.uniform(-0.08, 0.08) if aggressive else y + random.uniform(-0.03, 0.03)
                return (nx, ny)
            return (x, y)

        p["R1"] = jitter_robot(tuple(p.get("R1", (0.5325, 0.65))))
        p["R2"] = jitter_robot(tuple(p.get("R2", (0.5325, 1.2))))

        # 4) Reasignar algunas estaciones entre robots (cambiar robot1_stations/robot2_stations)
        r1 = list(p.get("robot1_stations", []))
        r2 = list(p.get("robot2_stations", []))

        # probabilidad de intercambiar alguna estación (pero no la 0/8 si son feeders)
        if random.random() < swap_robot_assign_prob:
            # elegir una estación movible (no fija y no index4)
            candidates = [i for i, fixed in enumerate(p["fixed_mask"]) if not fixed and i != 4]
            if candidates:
                chosen = random.choice(candidates)
                # mover chosen de r1 a r2 o viceversa
                if chosen in r1 and chosen not in r2:
                    r1 = [i for i in r1 if i != chosen]
                    r2.append(chosen)
                elif chosen in r2 and chosen not in r1:
                    r2 = [i for i in r2 if i != chosen]
                    r1.append(chosen)

        # asegurar que ambos robots tienen al menos una estación
        if not r1:
            r1 = [0]
        if not r2:
            r2 = [len(p["stations_base"]) - 1]

        p["robot1_stations"] = sorted(list(set(r1)))
        p["robot2_stations"] = sorted(list(set(r2)))

        # 5) Variar tiempos de estación
        new_times = []
        for t in p.get("tiempos_estacion", []):
            frac = random.uniform(-time_jitter_frac, time_jitter_frac)
            nt = max(0.0, t * (1 + frac))
            # discretizar a 0.1s para evitar ruido muy pequeño
            new_times.append(round(nt, 2))
        p["tiempos_estacion"] = new_times

        # 6) Vary reach / r_min slightly
        base_reach = p.get("reach", 0.381)
        base_rmin = p.get("r_min", 0.20)
        p["reach"] = max(0.1, base_reach * (1 + random.uniform(-reach_jitter, reach_jitter)))
        # ensure r_min < reach
        p["r_min"] = max(0.05, min(base_rmin * (1 + random.uniform(-0.2, 0.2)), p["reach"] * 0.9))

        # 7) Cambiar shape_mode ocasionalmente (pero permitir "BASE" a veces)
        shape_choices = ["BASE", "S", "U", "L"]
        if random.random() < 0.6:  # 60% de chance de variar
            p["shape_mode"] = random.choice(shape_choices)
        # shape_weight variation
        p["shape_weight"] = max(0.0, p.get("shape_weight", 1.0) * (1 + random.uniform(-0.5, 0.5)))

        # 8) Baseline_time variation modest
        p["baseline_time"] = max(10.0, p.get("baseline_time", 120.0) * (1 + random.uniform(-0.2, 0.2)))

        # 9) Variar fixed_mask: permitir "desfijar" algunas (pero nunca index 4)
        fixed = list(p.get("fixed_mask", []))
        for i in range(len(fixed)):
            if i == 4:
                fixed[i] = True
                continue
            if random.random() < unfix_prob:
                fixed[i] = False if not p["fixed_mask"][i] else False if random.random() < 0.4 else p["fixed_mask"][i]
            # con baja probabilidad, fijar otras
            if random.random() < 0.08:
                fixed[i] = True
        p["fixed_mask"] = fixed

        # 10) Finalmente, recomputar bounds si quieres usando default_bounds en el caller.

        return p

    def run(self):
        orig_speed = getattr(self.controller, "speed_multiplier", 1.0)

        try:
            # Acelerar simulación
            self.controller.speed_multiplier = self.speed_multiplier

            for run_idx in range(1, self.n_runs + 1):

                if self._stop:
                    self.status.emit("Batch cancelado por usuario")
                    break

                self.status.emit(f"Generando layout {run_idx}/{self.n_runs} (AG)")

                # Obtener parámetros base desde la pestaña
                params_source = self.tab.current_params

                # Generar params variantes: si self.vary_bounds True -> agresivo
                if self.vary_bounds:
                    params = self._randomize_params(params_source, aggressive=True)
                else:
                    # pequeño jitter para exploración ligera
                    params = self._randomize_params(params_source, aggressive=False)

                # Asegurar que est5 (index 4) quede siempre fija y en su posición base original
                try:
                    # si la pestaña tiene default params we preserve its original coord
                    original_base = self.tab.current_params["stations_base"][4]
                    params["stations_base"][4] = original_base
                    params["fixed_mask"][4] = True
                except Exception:
                    pass

                bounds_R1, bounds_R2 = lo.default_bounds(params)

                # Ejecutar AG (bloqueante, pero estamos en thread)
                result = lo.run_ga(params, bounds_R1, bounds_R2)

                if not result:
                    self.error.emit(f"AG devolvió None en la corrida {run_idx}")
                    return

                # Guardar resultado donde GUI y controller lo esperan
                self.tab.current_layout_id = result.get("layout_id")
                self.tab.current_layout_result = result
                setattr(self.controller, "optimized_layout", result)

                # Actualizar plot (opcional)
                try:
                    self.status.emit("Actualizando plot del layout")
                    self.tab.plot_layout(result["layout_vars"], params)
                except Exception:
                    pass

                # Generar JSON de secuencia
                self.status.emit("Generando JSON de secuencia")

                try:
                    # La pestaña guarda en su carpeta por convención
                    self.tab.generate_sequence()
                except Exception as e:
                    self.error.emit(f"Error generando sequence JSON: {e}")
                    return

                # Buscar JSON generado por distintas rutas posibles
                seq_filename = f"sequence_{result['layout_id']}.json"
                search_paths = [
                    os.getcwd(),
                    os.path.dirname(__file__),
                    os.path.dirname(getattr(self.tab, "__file__", __file__)) if hasattr(self.tab, "__file__") else ""
                ]

                found = False
                for path in search_paths:
                    if not path:
                        continue
                    candidate = os.path.join(path, seq_filename)
                    if os.path.exists(candidate):
                        seq_filename = candidate
                        found = True
                        break

                if not found:
                    # último intento: mismo directorio de la pestaña (si sabemos)
                    candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), seq_filename)
                    if os.path.exists(candidate):
                        seq_filename = candidate
                        found = True

                if not found:
                    self.error.emit(f"No se encontró el JSON de secuencia: {seq_filename}")
                    return

                # Ejecutar secuencia (modo batch)
                self.status.emit(f"Lanzando simulación virtual (run {run_idx})")
                setattr(self.controller, "_batch_mode", True)

                try:
                    self.controller.execute_sequence(
                        path=seq_filename,
                        layout_id=result['layout_id'],
                        repeat=1
                    )
                except Exception as e:
                    self.error.emit(f"Error arrancando execute_sequence: {e}")
                    return

                # Obtener referencia directa al worker creado por controller
                worker = getattr(self.controller, "_sequence_worker", None)

                if worker is None:
                    self.error.emit("No se encontró _sequence_worker después de execute_sequence")
                    return

                # Esperar a que el hilo termine realmente
                # worker.wait espera milisegundos, timeout configurable
                if not worker.wait(self.timeout_per_run * 1000):
                    self.error.emit(f"Timeout esperando fin de simulación en run {run_idx}")
                    return

                # Si terminó correctamente → avanzar
                self.progress.emit(run_idx, self.n_runs)
                self.status.emit(f"Run {run_idx} finalizada")
                # pequeño delay entre runs
                time.sleep(0.12)

            # Si salió del loop normalmente
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))

        finally:
            # Restaurar velocidad original siempre y desactivar flag batch
            try:
                setattr(self.controller, "_batch_mode", False)
            except Exception:
                pass
            try:
                self.controller.speed_multiplier = orig_speed
            except Exception:
                pass


class BatchDatasetOrchestrator(QObject):
    """Wrapper ligero para controlar el worker desde la UI."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, controller, tab, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.tab = tab
        self.worker = None

    def start(self, n_runs=10, vary_bounds=False, speed_multiplier=10.0, timeout_per_run=600):
        if self.worker is not None and self.worker.isRunning():
            self.error.emit("Ya hay un batch corriendo")
            return

        self.worker = BatchWorker(self.controller, self.tab, n_runs=n_runs, vary_bounds=vary_bounds,
                                 speed_multiplier=speed_multiplier, timeout_per_run=timeout_per_run)
        self.worker.progress.connect(lambda a,b: self.progress.emit(a,b))
        self.worker.finished.connect(lambda: self.finished.emit())
        self.worker.error.connect(lambda msg: self.error.emit(msg))
        self.worker.status.connect(lambda s: self.status.emit(s))

        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop()