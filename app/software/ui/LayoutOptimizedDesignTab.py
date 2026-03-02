from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox,
    QLineEdit
)
import json 
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import app.software.ia_model.layout_optimizer as lo
from app.software.ia_model.ga_settings import GASettingsDialog
from app.software.ia_model.Trajectory_generator import generate_robot_sequence_industrial



class LayoutOptimizerDesignerTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        # 🔥 Aquí guardaremos los parámetros actuales del GA
        self.current_params = lo.default_params()

        self.layout_data = None
        self.home_position = [0.0, 145.0, -90.0, 0.0, 0.0]

        self.init_ui()
        self.load_default_layout()
        self.current_layout_id = "00001"


    def open_settings_dialog(self):

        dialog = GASettingsDialog(self.current_params, self)

        if dialog.exec():

            self.current_params = dialog.get_updated_params()
            print("Parámetros del GA actualizados.")

    # ==================================================
    # UI
    # ==================================================

    def init_ui(self):

        main_layout = QVBoxLayout()

        # ==============================
        # Canvas Layout 2D
        # ==============================
        self.figure_layout = Figure()
        self.canvas_layout = FigureCanvas(self.figure_layout)
        main_layout.addWidget(self.canvas_layout)

        # ==============================
        # Botones superiores
        # ==============================
        top_buttons_layout = QHBoxLayout()

        self.btn_optimize = QPushButton("Optimizar Layout")
        self.btn_settings = QPushButton("...")
        self.btn_settings.clicked.connect(self.open_settings_dialog)


        top_buttons_layout.addWidget(self.btn_optimize)
        top_buttons_layout.addWidget(self.btn_settings)

        self.btn_batch = QPushButton("Simulación en cadena")
        top_buttons_layout.addWidget(self.btn_batch)
        self.btn_batch.clicked.connect(self.open_batch_dialog)

        main_layout.addLayout(top_buttons_layout)

        

        # ==============================
        # Resultados GA
        # ==============================
        metrics_group = QGroupBox("Resultados GA")
        metrics_layout = QVBoxLayout()

        self.label_fitness = QLabel("Fitness: -")
        self.label_time = QLabel("Tiempo estimado: -")
        self.label_improvement = QLabel("% Mejora: -")

        metrics_layout.addWidget(self.label_fitness)
        metrics_layout.addWidget(self.label_time)
        metrics_layout.addWidget(self.label_improvement)

        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)

        # ==============================
        # Parámetros de trayectoria
        # ==============================
        traj_group = QGroupBox("Parámetros de trayectoria")
        traj_layout = QHBoxLayout()

        traj_layout.addWidget(QLabel("PS Z (mm):"))
        self.ps_input = QLineEdit("234.635322")

        traj_layout.addWidget(QLabel("Pick Depth (mm):"))
        self.pick_input = QLineEdit("150")

        traj_layout.addWidget(self.ps_input)
        traj_layout.addWidget(self.pick_input)

        traj_group.setLayout(traj_layout)
        main_layout.addWidget(traj_group)

        # ==============================
        # Botones inferiores
        # ==============================
        bottom_buttons_layout = QHBoxLayout()

        self.btn_layout_scene = QPushButton("Organizar Layout Escenario")
        self.btn_generate = QPushButton("Generar Secuencia")

        bottom_buttons_layout.addWidget(self.btn_layout_scene)
        bottom_buttons_layout.addWidget(self.btn_generate)



        main_layout.addLayout(bottom_buttons_layout)

        self.setLayout(main_layout)

        # ==============================
        # Conexiones (vacías por ahora)
        # ==============================
        self.btn_optimize.clicked.connect(self.run_optimization)
        self.btn_layout_scene.clicked.connect(self.organize_layout_scene)
        self.btn_generate.clicked.connect(self.generate_sequence)

    # ==================================================
    # Ejecutar GA
    # ==================================================

    def run_optimization(self):

        bounds_R1, bounds_R2 = lo.default_bounds(self.current_params)

        result = lo.run_ga(self.current_params, bounds_R1, bounds_R2)

        try:
            layouts_dir = os.path.join(os.path.dirname(__file__), "layouts")
            os.makedirs(layouts_dir, exist_ok=True)
            layout_file = os.path.join(layouts_dir, f"layout_{result['layout_id']}.json")
            with open(layout_file, "w") as lf:
                json.dump(result, lf, indent=4)
            print("Resultado AG guardado en:", layout_file)
        except Exception as e:
            print("No se pudo guardar resultado AG:", e)

        self.current_layout_id = result["layout_id"]
        self.current_layout_result = result

        self.layout_data = result
        self.parent.optimized_layout = result

        self.label_fitness.setText(f"Fitness: {result['fitness']:.4f}")
        self.label_time.setText(f"Tiempo estimado: {result['total_time']:.2f}")
        self.label_improvement.setText(
            f"% Mejora: {result['improvement_percent']:.2f}%"
        )

        self.plot_layout(result["layout_vars"], self.current_params)

    # ==================================================
    # Plot Layout
    # ==================================================

    def plot_layout(self, best_layout, params):

        self.figure_layout.clear()
        ax = self.figure_layout.add_subplot(111)

        # Si no hay layout_vars (layout base), usar stations_base directo
        if best_layout is None:
            stations = params["stations_base"]
        else:
            stations = lo.build_full_layout(best_layout, params)

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

        self.canvas_layout.draw()


    # ==================================================
    # Organizar Escenario
    # ==================================================

    def organize_layout_scene(self):

        if not hasattr(self.parent, "optimized_layout"):
            print("No hay layout optimizado disponible.")
            return

        full_layout = self.parent.optimized_layout["stations"]

        if len(full_layout) != 9:
            print("El layout reconstruido no tiene 9 estaciones.")
            return

        from app.software.simulation.controller import update_station_positions
        update_station_positions(full_layout)

        print("Layout de escenario actualizado.")
        print(full_layout)

    def generate_sequence(self):

        if not hasattr(self.parent, "optimized_layout"):
            print("No hay layout optimizado disponible.")
            return

        result = self.parent.optimized_layout   # 🔥 AQUÍ

        layout = result["stations"]
        print(layout)
        params = result["params"]
        print(params)

        ps_z = float(self.ps_input.text())
        pick_depth = float(self.pick_input.text())

        sequence_R1 = generate_robot_sequence_industrial(layout, params, "R1",
                                            self.parent.robot1_kin,
                                            ps_z, pick_depth)

        sequence_R2 = generate_robot_sequence_industrial(layout, params, "R2",
                                            self.parent.robot2_kin,
                                            ps_z, pick_depth)

        full_sequence = {
            "sequence": sequence_R1 + sequence_R2
        }

        output_path = os.path.join(
    os.path.dirname(__file__),
    f"sequence_{self.current_layout_id}.json"
)

        with open(output_path, "w") as f:
            json.dump(full_sequence, f, indent=4)

        print("JSON guardado en:", output_path)

        # ==================================================
# Cargar Layout Base Automáticamente
# ==================================================

    def load_default_layout(self):

        print("Cargando layout base por defecto...")

        base_stations = self.current_params["stations_base"]

        default_result = {
            "layout_vars": None,
            "stations": base_stations,
            "movable_indices": [
                i for i, fixed in enumerate(self.current_params["fixed_mask"])
                if not fixed
            ],
            "fixed_mask": self.current_params["fixed_mask"],
            "params": self.current_params,
            "fitness": 1.0,
            "total_time": self.current_params["baseline_time"],
            "baseline_time": self.current_params["baseline_time"],
            "delta_time": 0.0,
            "improvement_percent": 0.0,
            "best_history": [],
            "avg_history": []
        }

        self.layout_data = default_result
        self.parent.optimized_layout = default_result

        self.plot_layout(None, self.current_params)

        print("Layout base cargado correctamente.")

    def open_batch_dialog(self):
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QCheckBox, QDoubleSpinBox, QPushButton

            dlg = QDialog(self)
            dlg.setWindowTitle("Simulación en cadena - Parámetros")

            layout = QVBoxLayout(dlg)

            # Número de corridas
            h1 = QHBoxLayout()
            h1.addWidget(QLabel("Corridas:"))
            runs_sb = QSpinBox()
            runs_sb.setRange(1, 10000)
            runs_sb.setValue(20)
            h1.addWidget(runs_sb)
            layout.addLayout(h1)

            # Vary bounds?
            h2 = QHBoxLayout()
            vary_cb = QCheckBox("Variar ligeramente estaciones (aumentar diversidad)")
            h2.addWidget(vary_cb)
            layout.addLayout(h2)

            # Speed multiplier
            h3 = QHBoxLayout()
            h3.addWidget(QLabel("Acelerador simulación (speed_multiplier):"))
            speed_ds = QDoubleSpinBox()
            speed_ds.setRange(1.0, 100.0)
            speed_ds.setValue(getattr(self.parent, "speed_multiplier", 7.0))
            speed_ds.setSingleStep(1.0)
            h3.addWidget(speed_ds)
            layout.addLayout(h3)

            # Timeout por corrida (seg)
            h4 = QHBoxLayout()
            h4.addWidget(QLabel("Timeout por corrida (s):"))
            timeout_sb = QSpinBox()
            timeout_sb.setRange(30, 3600)
            timeout_sb.setValue(600)
            h4.addWidget(timeout_sb)
            layout.addLayout(h4)

            # Buttons
            btns = QHBoxLayout()
            btn_ok = QPushButton("Iniciar")
            btn_cancel = QPushButton("Cancelar")
            btns.addWidget(btn_ok)
            btns.addWidget(btn_cancel)
            layout.addLayout(btns)

            def on_ok():
                n = runs_sb.value()
                vary = vary_cb.isChecked()
                speed = speed_ds.value()
                timeout = timeout_sb.value()
                dlg.accept()
                self.start_batch_run(n, vary, speed, timeout)

            btn_ok.clicked.connect(on_ok)
            btn_cancel.clicked.connect(dlg.reject)

            dlg.exec()

    def start_batch_run(self, n_runs, vary_bounds, speed_multiplier, timeout_per_run):
            # crea el orchestrator (módulo externo)
            from app.software.ia_model.batch_orchestator import BatchDatasetOrchestrator

            # self.parent es RobotController (según tu estructura)
            self._batch_orchestrator = BatchDatasetOrchestrator(self.parent, self)

            # conectar señales para feedback
            self._batch_orchestrator.progress.connect(lambda a,b: print(f"Batch progress: {a}/{b}"))
            self._batch_orchestrator.status.connect(lambda s: print("[Batch] " + s))
            self._batch_orchestrator.error.connect(lambda e: print("[Batch][ERR] " + e))
            self._batch_orchestrator.finished.connect(lambda: print("[Batch] finished"))

            # iniciar
            self._batch_orchestrator.start(n_runs=n_runs, vary_bounds=vary_bounds,
                                        speed_multiplier=speed_multiplier,
                                        timeout_per_run=timeout_per_run)









