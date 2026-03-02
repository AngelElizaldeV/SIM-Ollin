from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit,
    QCheckBox, QGroupBox, QGridLayout,
    QComboBox, QPlainTextEdit
)

from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import app.software.ia_model.layout_optimizer as lo
import app.software.ia_model.Trajectory_generator as tg
import json, os


class SmartTrajectoryTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    # ==================================================
    # UI
    # ==================================================

    def init_ui(self):

        main_layout = QVBoxLayout()

        params = lo.default_params()

        # -----------------------------
        # Baseline input
        # -----------------------------
        baseline_layout = QHBoxLayout()
        baseline_layout.addWidget(QLabel("Tiempo baseline:"))
        self.baseline_input = QLineEdit("120.0")
        baseline_layout.addWidget(self.baseline_input)
        main_layout.addLayout(baseline_layout)

        # -----------------------------
        # Estaciones fijas
        # -----------------------------
        fixed_group = QGroupBox("Estaciones fijas")
        fixed_layout = QHBoxLayout()
        self.fixed_checkboxes = []

        for i, is_fixed in enumerate(params["fixed_mask"]):
            cb = QCheckBox(f"E{i+1}")
            cb.setChecked(is_fixed)
            fixed_layout.addWidget(cb)
            self.fixed_checkboxes.append(cb)


        fixed_group.setLayout(fixed_layout)
        main_layout.addWidget(fixed_group)

        robots_container = QHBoxLayout()

        # ==============================
        # Robot 1
        # ==============================
        r1_group = QGroupBox("Robot 1")
        r1_layout = QGridLayout()
        self.r1_checkboxes = []

        for i in range(len(params["stations_base"])):
            cb = QCheckBox(f"E{i+1}")
            cb.setChecked(i in params["robot1_stations"])

            row = i % 4
            col = i // 4

            r1_layout.addWidget(cb, row, col)
            self.r1_checkboxes.append(cb)

        r1_group.setLayout(r1_layout)
        robots_container.addWidget(r1_group)

        # ==============================
        # Robot 2
        # ==============================
        r2_group = QGroupBox("Robot 2")
        r2_layout = QGridLayout()
        self.r2_checkboxes = []

        for i in range(len(params["stations_base"])):
            cb = QCheckBox(f"E{i+1}")
            cb.setChecked(i in params["robot2_stations"])

            row = i % 4
            col = i // 4

            r2_layout.addWidget(cb, row, col)
            self.r2_checkboxes.append(cb)
            
        r2_group.setLayout(r2_layout)
        robots_container.addWidget(r2_group)

        main_layout.addLayout(robots_container)


        # -----------------------------
        # Shape selector
        # -----------------------------
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("Forma layout:"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["BASE", "S", "U"])
        shape_layout.addWidget(self.shape_combo)
        main_layout.addLayout(shape_layout)

        # -----------------------------
        # Botón optimizar
        # -----------------------------
        self.optimize_button = QPushButton("Optimizar Layout")
        self.optimize_button.clicked.connect(self.run_optimization)
        main_layout.addWidget(self.optimize_button)


        # ==============================
# Métricas
# ==============================
        metrics_group = QGroupBox("Resultados")
        metrics_layout = QVBoxLayout()

        self.label_fitness = QLabel("Fitness: -")
        self.label_time = QLabel("Tiempo estimado: -")
        self.label_improvement = QLabel("% Mejora: -")

        metrics_layout.addWidget(self.label_fitness)
        metrics_layout.addWidget(self.label_time)
        metrics_layout.addWidget(self.label_improvement)

        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)

        # -----------------------------
        # Canvas Layout
        # -----------------------------
        self.figure_layout = Figure()
        self.canvas_layout = FigureCanvas(self.figure_layout)
        

        # -----------------------------
        # Canvas Convergencia
        # -----------------------------
        self.figure_conv = Figure()
        self.canvas_conv = FigureCanvas(self.figure_conv)
        
        graphs_layout = QHBoxLayout()
        graphs_layout.addWidget(self.canvas_layout)
        graphs_layout.addWidget(self.canvas_conv)

        main_layout.addLayout(graphs_layout)

        self.setLayout(main_layout)

    # ==================================================
    # Ejecutar GA
    # ==================================================

    def run_optimization(self):

        baseline = float(self.baseline_input.text())

        fixed_mask = [cb.isChecked() for cb in self.fixed_checkboxes]
        robot1_stations = [i for i, cb in enumerate(self.r1_checkboxes) if cb.isChecked()]
        robot2_stations = [i for i, cb in enumerate(self.r2_checkboxes) if cb.isChecked()]

        params = lo.default_params()  # crearemos esta función
        params["baseline_time"] = baseline
        params["fixed_mask"] = fixed_mask
        params["robot1_stations"] = robot1_stations
        params["robot2_stations"] = robot2_stations
        params["shape_mode"] = self.shape_combo.currentText()

        bounds_R1, bounds_R2 = lo.default_bounds(params)

        result = lo.run_ga(params, bounds_R1, bounds_R2)
        self.parent.optimized_layout = result


        

        self.label_fitness.setText(f"Fitness: {result['fitness']:.4f}")
        self.label_time.setText(f"Tiempo estimado: {result['total_time']:.2f}")
        self.label_improvement.setText(
            f"% Mejora: {result['improvement_percent']:.2f}%"
        )

        self.plot_layout(result["layout_vars"], params)
        self.plot_convergence(result["best_history"], result["avg_history"])
        



        baseline = params["baseline_time"]

    # ==================================================
    # Plot Layout
    # ==================================================

    def plot_layout(self, best_layout, params):

        self.figure_layout.clear()
        ax = self.figure_layout.add_subplot(111)

        lo.plot_layout_on_axis(ax, best_layout, params)

        self.canvas_layout.draw()


    # ==================================================
    # Plot Convergencia
    # ==================================================

    def plot_convergence(self, best_history, avg_history):

        self.figure_conv.clear()
        ax = self.figure_conv.add_subplot(111)

        ax.plot(best_history, label="Mejor")
        ax.plot(avg_history, linestyle="--", label="Promedio")

        ax.legend()
        ax.grid(True)

        self.canvas_conv.draw()
