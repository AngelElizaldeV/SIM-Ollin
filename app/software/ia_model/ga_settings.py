from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QCheckBox,
    QGroupBox, QGridLayout,
    QComboBox, QDialogButtonBox
)

import app.software.ia_model.layout_optimizer as lo
# al inicio del archivo GA_settings.py
import copy


class GASettingsDialog(QDialog):

    def __init__(self, current_params, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Configuración del Algoritmo Genético")
        self.current_params = current_params

        self.init_ui()

    # ==================================================
    # UI
    # ==================================================

    def init_ui(self):

        main_layout = QVBoxLayout()

        params = self.current_params

        # -----------------------------
        # Baseline
        # -----------------------------
        baseline_layout = QHBoxLayout()
        baseline_layout.addWidget(QLabel("Tiempo baseline:"))

        self.baseline_input = QLineEdit(str(params["baseline_time"]))
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

        # -----------------------------
        # Robot 1
        # -----------------------------
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
        main_layout.addWidget(r1_group)

        # -----------------------------
        # Robot 2
        # -----------------------------
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
        main_layout.addWidget(r2_group)

        # -----------------------------
        # Shape
        # -----------------------------
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("Forma layout:"))

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["BASE", "S", "U", "L"])
        self.shape_combo.setCurrentText(params["shape_mode"])

        shape_layout.addWidget(self.shape_combo)
        main_layout.addLayout(shape_layout)

        times_group = QGroupBox("Tiempos por estación (seg)")
        times_layout = QGridLayout()

        self.time_inputs = []

        for i, t in enumerate(params["tiempos_estacion"]):

            label = QLabel(f"E{i+1}")
            line = QLineEdit(str(t))

            row = i % 5
            col = (i // 5) * 2

            times_layout.addWidget(label, row, col)
            times_layout.addWidget(line, row, col + 1)

            self.time_inputs.append(line)

        times_group.setLayout(times_layout)
        main_layout.addWidget(times_group)

        # -----------------------------
        # Botones OK / Cancel
        # -----------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        main_layout.addWidget(buttons)

        self.setLayout(main_layout)

    # ==================================================
    # Obtener parámetros actualizados
    # ==================================================

    def get_updated_params(self):

        try:
            # No volver a default_params() — queremos partir de los params actuales
            new_params = copy.deepcopy(self.current_params)

            # Baseline
            new_params["baseline_time"] = float(self.baseline_input.text())

            # Fijas
            new_params["fixed_mask"] = [
                cb.isChecked() for cb in self.fixed_checkboxes
            ]

            # Robots
            r1 = [i for i, cb in enumerate(self.r1_checkboxes) if cb.isChecked()]
            r2 = [i for i, cb in enumerate(self.r2_checkboxes) if cb.isChecked()]

            if not r1:
                raise ValueError("Robot 1 debe tener al menos una estación.")

            if not r2:
                raise ValueError("Robot 2 debe tener al menos una estación.")

            new_params["robot1_stations"] = r1
            new_params["robot2_stations"] = r2

            # Forma
            new_params["shape_mode"] = self.shape_combo.currentText()

            # Tiempos
            new_params["tiempos_estacion"] = [
                float(inp.text()) for inp in self.time_inputs
            ]

            return new_params

        except Exception as e:
            print("Error en parámetros:", e)
            return self.current_params
