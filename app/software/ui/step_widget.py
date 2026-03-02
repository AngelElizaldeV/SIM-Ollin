from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QComboBox, QHBoxLayout,
    QMessageBox, QListWidget, QFileDialog, QDialog, QDoubleSpinBox, QLineEdit, QListWidgetItem
)

import json, copy 


class StepWidget(QWidget):
    """Widget que representa un paso editable."""
    def __init__(self, parent=None, step_data=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)

        # --- Robot ---
        self.robot_cb = QComboBox()
        self.robot_cb.addItems(["Robot 1", "Robot 2"])
        layout.addWidget(self.robot_cb)

        # --- Joints ---
        self.joints_edit = QLineEdit()
        self.joints_edit.setPlaceholderText("j0,j1,j2,j3,j4")
        layout.addWidget(self.joints_edit)

        # --- Gripper ---
        self.gripper_cb = QComboBox()
        self.gripper_cb.addItems(["Cerrado", "Abierto"])
        layout.addWidget(self.gripper_cb)

        # --- Tiempo de espera ---
        self.wait_input = QDoubleSpinBox()
        self.wait_input.setRange(0.0, 999.0)
        self.wait_input.setSingleStep(0.5)
        self.wait_input.setDecimals(2)
        layout.addWidget(self.wait_input)

        # --- Etiqueta ---
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Etiqueta opcional")
        layout.addWidget(self.label_input)

        # Si se pasa un paso preexistente, se cargan los datos
        if step_data:
            self.set_data(step_data)

    def get_data(self):
        """Devuelve los datos actuales del widget en formato dict."""
        joints_text = self.joints_edit.text()
        try:
            joints = [float(x.strip()) for x in joints_text.split(",")]
        except ValueError:
            joints = [0] * 5

        gripper = 1 if self.gripper_cb.currentText() == "Abierto" else 0

        return {
            "robot": self.robot_cb.currentText(),
            "joints": joints,
            "gripper": gripper,
            "wait": float(self.wait_input.value()),
            "label": self.label_input.text() if self.label_input.text() else None
        }

    def set_data(self, data):
        """Carga datos en el widget desde un diccionario."""
        self.robot_cb.setCurrentText(data.get("robot", "Robot 1"))
        self.joints_edit.setText(",".join(str(round(j, 2)) for j in data.get("joints", [0]*5)))
        self.gripper_cb.setCurrentText("Abierto" if data.get("gripper", 0) else "Cerrado")
        self.wait_input.setValue(data.get("wait", 0.0))
        self.label_input.setText(data.get("label", "") or "")

class SequenceDialog(QDialog):
    """Diálogo para crear y editar una secuencia de pasos."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crear Nueva Secuencia")
        self.resize(800, 400)
        self.parent = parent  # referencia a RobotController

        layout = QVBoxLayout(self)

        # --- Lista de pasos ---
        self.list_steps = QListWidget()
        layout.addWidget(self.list_steps)

        # --- Botones ---
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Agregar Paso desde posición actual")
        self.btn_add.clicked.connect(self.add_step_from_current)
        btn_layout.addWidget(self.btn_add)

        self.btn_duplicate = QPushButton("Duplicar paso seleccionado")
        self.btn_duplicate.clicked.connect(self.duplicate_step)
        btn_layout.addWidget(self.btn_duplicate)

        self.btn_delete = QPushButton("Eliminar Paso")
        self.btn_delete.clicked.connect(self.delete_step)
        btn_layout.addWidget(self.btn_delete)

        self.btn_save = QPushButton("Guardar Secuencia")
        self.btn_save.clicked.connect(self.save_sequence)
        btn_layout.addWidget(self.btn_save)

        layout.addLayout(btn_layout)


    def add_step_from_current(self):
        """Agrega un paso tomando los valores actuales de los sliders del robot."""
        try:
            # Determinar robot seleccionado (por si tienes combo)
            robot_choice = "Robot 1"
            if hasattr(self.parent, "robot_selector"):
                robot_choice = self.parent.robot_selector.currentText()

            # Detectar si sliders_joints es lista o diccionario
            sliders_attr = getattr(self.parent, "sliders_joints", None)
            if sliders_attr is None:
                raise AttributeError("No se encontró 'sliders_joints' en el controlador principal.")

            if isinstance(sliders_attr, dict):
                joints = [slider.value() for slider in sliders_attr.values()]
            else:
                joints = [slider.value() for slider in sliders_attr]

            # Leer estado del gripper
            gripper = getattr(self.parent, "gripper_state", 0)

            step_data = {
                "robot": robot_choice,
                "joints": joints,
                "gripper": gripper,
                "wait": 0.0,
                "label": None
            }

            self._add_step_widget(step_data)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo leer la posición actual del robot.\n\n{e}")


    def _add_step_widget(self, step_data=None):
        """Crea y agrega un StepWidget al final de la lista."""
        step_widget = StepWidget(step_data=step_data)
        item = QListWidgetItem()
        item.setSizeHint(step_widget.sizeHint())
        self.list_steps.addItem(item)
        self.list_steps.setItemWidget(item, step_widget)

    def duplicate_step(self):
        """Duplica el paso seleccionado y lo agrega al final."""
        current_row = self.list_steps.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Advertencia", "Selecciona un paso para duplicar.")
            return

        item = self.list_steps.item(current_row)
        widget = self.list_steps.itemWidget(item)
        data_copy = copy.deepcopy(widget.get_data())
        self._add_step_widget(data_copy)

    def delete_step(self):
        """Elimina el paso seleccionado."""
        row = self.list_steps.currentRow()
        if row >= 0:
            self.list_steps.takeItem(row)

    def save_sequence(self):
        """Guarda todos los pasos en un archivo JSON."""
        if self.list_steps.count() == 0:
            QMessageBox.warning(self, "Advertencia", "No hay pasos en la secuencia.")
            return

        sequence = []
        for i in range(self.list_steps.count()):
            item = self.list_steps.item(i)
            widget = self.list_steps.itemWidget(item)
            sequence.append(widget.get_data())

        data_to_save = {"sequence": sequence}

        path, _ = QFileDialog.getSaveFileName(self, "Guardar Secuencia", "", "JSON Files (*.json)")
        if path:
            with open(path, "w") as f:
                json.dump(data_to_save, f, indent=4)
            QMessageBox.information(self, "Éxito", "Secuencia guardada correctamente.")



