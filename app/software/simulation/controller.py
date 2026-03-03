
import time
import csv
import uuid
import math
import json
from datetime import datetime, timedelta
import threading
from threading import Lock

# -------------------------
# 2. THIRD-PARTY LIBRARIES
# -------------------------
import numpy as np
import pybullet as p
from serial.tools import list_ports
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton,
    QSlider, QLabel, QComboBox, QTabWidget, QHBoxLayout,
    QMessageBox, QSpinBox, QListWidget, QFileDialog, QGroupBox, QLineEdit, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QCoreApplication

import pandas as pd
import os

# -------------------------
# 3. LOCAL PROJECT MODULES
# -------------------------
import app.public.globals as globals
from app.software.robot.Dorna_Controller import DornaController
from app.software.ui.step_widget import SequenceDialog
from app.software.simulation.scene import load_robot, load_asset
from app.public.utils import set_joint_positions
from app.software.robot.robot_kinematics import RobotKinematics
from app.software.simulation.virtual_controller import VirtualRobotController
from app.software.simulation.physical_sync import PhysicalRobotSync
from app.software.simulation.sequence_worker import SequenceWorker
from app.software.ia_model.rn_dataset_builder import RNDatasetBuilder


from app.software.ui.LayoutOptimizedDesignTab import LayoutOptimizerDesignerTab
from PyQt6.QtCore import pyqtSignal

class SliderUpdater(QObject):
    update_sliders_signal = pyqtSignal()


class RobotController(QTabWidget):

    sequence_finished_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # Robots digitales
        
        self.robot1 = load_robot(position=[0.53,0.65,0.665], fixed=True)
        self.robot2 = load_robot(position=[0.53,1.2,0.665], fixed=True)
        set_joint_positions(self.robot1, np.radians([0,154,-142,90,0]))
        set_joint_positions(self.robot2, np.radians([0,154,-142,90,0]))
        self.selected_robot = self.robot1
        self.end_effector_index = 4
        self.cube_id = load_asset()
        self.grab_constraint = None
        self.dorna = None
        self.current_trajectory = []
        self.slider_updater = SliderUpdater()
        self.slider_updater.update_sliders_signal.connect(self.update_joint_sliders_from_robot)
        self.robot1_trajectory = []
        self.robot2_trajectory = []
        # trayectoria "activa" (alias a la del robot seleccionado para UI)
        self.current_trajectory = self.robot1_trajectory

        self._phys_lock = Lock()
        # Evitar que actualizaciones desde el hilo sobrescriban interacción manual (por robot)
        self._last_manual_update_robot1 = 0.0
        self._last_manual_update_robot2 = 0.0
        self._manual_hang_seconds = 0.25  # ajustar si quieres más o menos tolerancia

        # Lecturas físicas separadas por robot (antes había solo `latest_physical_positions`)
        self.latest_physical_positions_robot1 = None
        self.latest_physical_positions_robot2 = None

        # Instancias físicas
        self.robot1_instance = None
        self.robot2_instance = None
        self.robot1_thread = None
        self.robot2_thread = None

        self.robot1_connected = False
        self.robot2_connected = False


        self.latest_timestamp_robot1 = None
        self.latest_timestamp_robot2 = None

        # Log de latencias virtuales (ms)
        # log y último valor de latencia virtual (ms)
        self.virtual_latency_log = []  # muestras en ms
        self.last_virtual_latency_robot1 = None
        self.last_virtual_latency_robot2 = None
        
        self.robot1_kin = RobotKinematics()
        self.robot2_kin = RobotKinematics()
        self.virtual = VirtualRobotController(self)
        self.physical = PhysicalRobotSync(self)
        self.rn_builder = RNDatasetBuilder(self)


        self._setup_tabs()
        self._setup_timer()
        self.speed_multiplier = 5.0 # 1.0 velocidad normal
        
    def move_relative(self, xyz):

        instance = None
        digital = None
        kin_model = None

        if self.selected_robot == self.robot1:
            instance = self.robot1_instance
            digital = self.robot1
            kin_model = self.robot1_kin

        elif self.selected_robot == self.robot2:
            instance = self.robot2_instance
            digital = self.robot2
            kin_model = self.robot2_kin

        # ==============================
        # ROBOT FÍSICO
        # ==============================
        if instance and instance.ready_to_use():

            instance.move_line(xyz)   # relativo ya lo maneja tu API
            self.last_move_time = time.time()
            return

        # ==============================
        # ROBOT VIRTUAL
        # ==============================
        print("🤖 Ejecutando move_relative en virtual")

        # Obtener posición actual del end effector
        current_joints = [
            math.degrees(p.getJointState(digital, i)[0])
            for i in range(5)
        ]

        current_xyz = kin_model.joint_to_xyz(current_joints)

        x = current_xyz[0] + xyz[0]
        y = current_xyz[1] + xyz[1]
        z = current_xyz[2] + xyz[2]


        self.virtual.virtual_move_line_blocking(
            digital,
            kin_model,
            [x, y, z]
        )

        self.last_move_time = time.time()


    # ----------------------------------------
    # Setup de pestañas
    # ----------------------------------------
    def _setup_tabs(self):
        # --- 🔹 Pestaña de conexión ---
        self.move_tab = QWidget()
        layout = QVBoxLayout(self.move_tab)
        robots_layout = QHBoxLayout()

        self.panel_robot1 = self._crear_panel_robot("Robot 1")
        self.panel_robot2 = self._crear_panel_robot("Robot 2")
        robots_layout.addWidget(self.panel_robot1)
        robots_layout.addWidget(self.panel_robot2)

        self.home_all_button = QPushButton("🏠 Home All Robots")
        self.home_all_button.clicked.connect(self.physical.home_all)
        layout.addWidget(self.home_all_button)
        layout.addLayout(robots_layout)
        self.addTab(self.move_tab, "Conexión")    

        # --- Tab Control y Trayectorias ---
        self.control_tab = QWidget()
        control_layout = QVBoxLayout(self.control_tab)

        self.Layout_tab = LayoutOptimizerDesignerTab(self)
        self.addTab(self.Layout_tab, "Trayectoria Inteligente")


        """# --- NUEVA TAB: Trayectoria Inteligente ---
        self.smart_tab = SmartTrajectoryTab(self)
        self.addTab(self.smart_tab, "Trayectoria Inteligente")


        self.trajectory_desgigner = TrajectoryDesignerTab(self)
        self.addTab(self.trajectory_desgigner, "Trayectoria Automatica (BETA)")"""

        # Selector de robot para el panel de control (ponerlo antes de crear los sliders)
        self.control_robot_selector = QComboBox()
        self.control_robot_selector.addItems(["Robot 1", "Robot 2"])
        self.control_robot_selector.currentIndexChanged.connect(self.on_control_robot_changed)
        control_layout.addWidget(QLabel("Seleccionar robot a controlar:"))
        control_layout.addWidget(self.control_robot_selector)
        self.btn_home = QPushButton("Mover a Home Pos")
        self.btn_home.setMinimumHeight(30)
        self.btn_home.clicked.connect(self.move_to_homePos)
        control_layout.addWidget(self.btn_home)

        self.sliders_joints = {}
        self.joint_spinboxes = {}
        self.manual_inputs = {}

        for i in range(len(globals.JOINT_LIMITS)):
            low, high = globals.JOINT_LIMITS[i]

            try:
                name = p.getJointInfo(self.selected_robot, i)[1].decode()
            except Exception:
                name = f"j{i}"

            row = QHBoxLayout()
            row.addWidget(QLabel(f"Joint {i} - {name}:"))

            # --- Slider (teleoperación en tiempo real) ---
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(low), int(high))
            s.valueChanged.connect(self.update_joint_positions)
            row.addWidget(s)

            # --- SpinBox (solo visual) ---
            sp = QSpinBox()
            sp.setRange(int(low), int(high))
            sp.setReadOnly(False)
            sp.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
            sp.valueChanged.connect(lambda v, sl=s: sl.setValue(v))
            row.addWidget(sp)

            # --- Input manual ---
            angle_input = QLineEdit()
            angle_input.setPlaceholderText("°")
            angle_input.setFixedWidth(50)
            row.addWidget(angle_input)

            # --- Botón mover ---
            btn_move = QPushButton("Mover")
            btn_move.clicked.connect(lambda _, j=i: self.apply_manual_angle(j))
            row.addWidget(btn_move)

            control_layout.addLayout(row)

            self.sliders_joints[i] = s
            self.joint_spinboxes[i] = sp
            self.manual_inputs[i] = angle_input


        num_joints = p.getNumJoints(self.selected_robot)
        for i in range(num_joints):
            info = p.getJointInfo(self.selected_robot, i)
            print(i, info[1].decode("utf-8"))

        finger_left_name = "Pinza1"
        finger_right_name = "Pinza2"

        # Buscar índices de forma segura
        self.finger_left_index = None
        self.finger_right_index = None
        for i in range(num_joints):
            nm = p.getJointInfo(self.selected_robot, i)[1].decode("utf-8")
            if nm == finger_left_name:
                self.finger_left_index = i
            if nm == finger_right_name:
                self.finger_right_index = i

        if self.finger_left_index is not None:
            p.changeDynamics(self.selected_robot, self.finger_left_index, lateralFriction=5.0, restitution=0.0)
        if self.finger_right_index is not None:
            p.changeDynamics(self.selected_robot, self.finger_right_index, lateralFriction=5.0, restitution=0.0)
        p.changeDynamics(self.cube_id, -1, lateralFriction=2.0, restitution=0.0)

        self.OC = QPushButton("Abrir/Cerrar Gripper")
        self.OC.setCheckable(True)
        self.OC.clicked.connect(self.Open_Close)
        control_layout.addWidget(self.OC)
        self.gripper_state = 0

        # Botones de trayectoria
        self.addTab(self.control_tab, "Control y Trayectorias")

        self.btn_create_sequence = QPushButton("Crear Nueva Secuencia")
        self.btn_create_sequence.clicked.connect(self.open_sequence_dialog)
        control_layout.addWidget(self.btn_create_sequence)

        self.btn_execute_sequence = QPushButton("Ejecutar Secuencia")
        self.btn_execute_sequence.clicked.connect(self.execute_sequence)
        control_layout.addWidget(self.btn_execute_sequence)


        # Movimiento relativo XYZ
        # Movimiento relativo XYZ (cruceta)
        grp_move_rel = QGroupBox("Movimiento Relativo Efector Final")

        grid = QGridLayout()

        btn_up = QPushButton("↑ Arriba")
        btn_down = QPushButton("↓ Abajo")
        btn_left = QPushButton("← Izquierda")
        btn_right = QPushButton("→ Derecha")
        btn_forward = QPushButton("Adelante")
        btn_back = QPushButton("Atrás")

        # Cruceta central (X, Z)
        grid.addWidget(btn_up,    0, 1)
        grid.addWidget(btn_left,  1, 0)
        grid.addWidget(btn_right, 1, 2)
        grid.addWidget(btn_down,  2, 1)

        # Movimiento en Y (adelante / atrás)
        grid.addWidget(btn_forward, 3, 1)
        grid.addWidget(btn_back,    4, 1)

        grp_move_rel.setLayout(grid)
        control_layout.addWidget(grp_move_rel)


        btn_left.clicked.connect(lambda: self.move_relative([-20,0,0]))
        btn_right.clicked.connect(lambda: self.move_relative([20,0,0]))
        btn_up.clicked.connect(lambda: self.move_relative([0,0,20]))
        btn_down.clicked.connect(lambda: self.move_relative([0,0,-20]))
        btn_forward.clicked.connect(lambda: self.move_relative([0,20,0]))
        btn_back.clicked.connect(lambda: self.move_relative([0,-20,0]))

        # --- Tab Escenario ---
        self.scenario_tab = QWidget()
        scen_layout = QVBoxLayout(self.scenario_tab)
        scen_layout.addWidget(QLabel("URDFs en simulación (ID: Nombre @ [X, Y, Z]):"))
        self.urdf_list = QListWidget(); scen_layout.addWidget(self.urdf_list)
        btns = QHBoxLayout()
        btn_refresh = QPushButton("Actualizar"); btn_refresh.clicked.connect(self.refresh_urdf_list)
        btns.addWidget(btn_refresh)
        btn_load = QPushButton("Cargar"); btn_load.clicked.connect(self.load_urdf)
        btns.addWidget(btn_load)
        btn_delete = QPushButton("Eliminar"); btn_delete.clicked.connect(self.delete_selected_urdf)
        btns.addWidget(btn_delete)
        scen_layout.addLayout(btns)
        self.scenario_tab.setLayout(scen_layout)
        self.addTab(self.scenario_tab, "Escenario")
        self.refresh_urdf_list()

    def open_sequence_dialog(self):
        dlg = SequenceDialog(self)
        dlg.show()
    
    def load_urdf(self):
        pass

    # ----------------------------------------
    # Crear panel de robot con conexión y botones
    # ----------------------------------------
    def _crear_panel_robot(self, nombre_robot):
        grp = QGroupBox(nombre_robot)
        layout = QVBoxLayout(grp)

        layout.addWidget(QLabel("Puerto COM:"))
        combo_com = QComboBox()
        # CORRECCIÓN: no usar 'p' aquí (es pybullet). usar list_ports correctamente.
        puertos = [port.device for port in list_ports.comports()]
        combo_com.addItems(puertos if puertos else ["Sin puertos detectados"])
        layout.addWidget(combo_com)

        layout.addWidget(QLabel("Archivo de configuración (.yaml):"))
        yaml_selector = QPushButton("Seleccionar archivo")
        yaml_label = QLabel("Ninguno seleccionado")
        layout.addWidget(yaml_selector); layout.addWidget(yaml_label)

        def seleccionar_yaml():
            path, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo YAML", "", "YAML Files (*.yaml)")
            if path: yaml_label.setText(path)
        yaml_selector.clicked.connect(seleccionar_yaml)

        # Botones
        btn_conectar = QPushButton("Conectar")
        btn_desconectar = QPushButton("Desconectar")
        btn_home = QPushButton("Homing")
        btn_move00 = QPushButton("Mover 0-0")
        btn_select = QPushButton("Seleccionar como activo")
        for b in [btn_conectar, btn_desconectar, btn_home, btn_move00, btn_select]: layout.addWidget(b)

        # --- Lógica de botones ---
        def conectar_robot():
            com = combo_com.currentText(); yaml_path = yaml_label.text()
            if "Ninguno" in yaml_path or not yaml_path.endswith(".yaml"):
                QMessageBox.warning(self, "Advertencia", f"Selecciona un archivo YAML para {nombre_robot}")
                return
            try:
                instance = DornaController(yaml_path, com)
                if nombre_robot == "Robot 1":
                    self.robot1_instance = instance
                    self.robot1_thread = self.physical.start_robot_thread(self.robot1_instance, self.robot1)
                    self.robot1_connected = True                       # <-- marcar conectado
                    self.physical.control_gripper(False)
                else:
                    self.robot2_instance = instance
                    self.robot2_thread = self.physical.start_robot_thread(self.robot2_instance, self.robot2)
                    self.robot2_connected = True                       # <-- marcar conectado
                    self.physical.control_gripper(False)

                QMessageBox.information(self, "Conexión exitosa", f"{nombre_robot} conectado en {com}")

                # actualizar UI inmediatamente para evitar que sliders vuelvan a home
                try:
                    self.slider_updater.update_sliders_signal.emit()
                except Exception:
                    pass

            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo conectar {nombre_robot}\n{e}")


        def desconectar_robot():
            if nombre_robot == "Robot 1" and self.robot1_instance:
                try:
                    self.robot1_instance.running = False
                    self.robot1_instance.disconnect()
                except Exception as e:
                    print(f"Error desconectando Robot 1: {e}")
                self.robot1_instance = None
                self.robot1_connected = False                       # <-- marcar desconectado
            elif nombre_robot == "Robot 2" and self.robot2_instance:
                try:
                    self.robot2_instance.running = False
                    self.robot2_instance.disconnect()
                except Exception as e:
                    print(f"Error desconectando Robot 2: {e}")
                self.robot2_instance = None
                self.robot2_connected = False                       # <-- marcar desconectado
            QMessageBox.information(self, "Desconexión", f"{nombre_robot} desconectado.")

            # forzar actualización UI para reflejar cambio (leer URDF ahora)
            try:
                self.slider_updater.update_sliders_signal.emit()
            except Exception:
                pass


        def homing_robot():
            instance = self.robot1_instance if nombre_robot == "Robot 1" else self.robot2_instance
            digital = self.robot1 if nombre_robot == "Robot 1" else self.robot2
            if instance:
                self.physical.home_robot_with_digital(instance, digital)
            else:
                QMessageBox.warning(self, "Advertencia", f"{nombre_robot} no está conectado.")

        def mover00():
            instance = self.robot1_instance if nombre_robot == "Robot 1" else self.robot2_instance
            if instance:
                try:
                    instance.move00()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"No se pudo mover 0-0: {e}")
            else:
                QMessageBox.warning(self, "Advertencia", f"{nombre_robot} no está conectado.")
        


        def seleccionar_activo():
            if nombre_robot == "Robot 1":
                self.selected_robot = self.robot1
            else:
                self.selected_robot = self.robot2
            QMessageBox.information(self, "Robot seleccionado", f"{nombre_robot} seleccionado como activo.")

        btn_conectar.clicked.connect(conectar_robot)
        btn_desconectar.clicked.connect(desconectar_robot)
        btn_home.clicked.connect(homing_robot)
        btn_move00.clicked.connect(mover00)
        btn_select.clicked.connect(seleccionar_activo)

        return grp

    # ----------------------------------------
    # Timer principal para simulación
    # ----------------------------------------
    def _setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_simulation_and_read)
        self.timer.start(int(globals.TIME_STEP * 1000))

    def _update_simulation_and_read(self):
        """
        Actualiza la simulación (sin leer robots físicos).
        """
        try:
            p.stepSimulation()

        except Exception as e:
            print(f"Error en stepSimulation: {e}")

    def on_control_robot_changed(self, index):
        try:
            self.selected_robot = self.robot1 if index == 0 else self.robot2
            # Actualizar referencia de trayectoria activa
            self.current_trajectory = self.robot1_trajectory if self.selected_robot == self.robot1 else self.robot2_trajectory
            # Actualizar sliders para que reflejen el robot seleccionado
            self.update_joint_sliders_from_robot()
        except Exception as e:
            print(f"Error al cambiar robot de control: {e}")
    
    def update_joint_sliders_from_robot(self):
        """
        Actualiza sliders/spinboxes/labels con la posición actual
        del robot seleccionado.

        - Si el robot físico está conectado → usa lectura física suavizada.
        - Si NO está conectado → usa estado del URDF (simulación).
        - Respeta interacción manual reciente.
        """
        try:
            now = time.time()

            # --- No sobrescribir si el usuario interactuó recientemente ---
            if self.selected_robot == self.robot1:
                if now - self._last_manual_update_robot1 < self._manual_hang_seconds:
                    return
                robot_connected = getattr(self, "robot1_connected", False)
            else:
                if now - self._last_manual_update_robot2 < self._manual_hang_seconds:
                    return
                robot_connected = getattr(self, "robot2_connected", False)  # <- CORRECCIÓN

            # --- No sobrescribir si el usuario está arrastrando slider ---
            for sl in self.sliders_joints.values():
                try:
                    if sl.isSliderDown():
                        return
                except Exception:
                    pass

            n_joints = len(self.sliders_joints)
            degrees = [0] * n_joints

            phys_positions = None
            timestamp_physical = None

            # --- Leer datos físicos SOLO si está conectado ---
            if robot_connected:
                with self._phys_lock:
                    if self.selected_robot == self.robot1:
                        if self.latest_physical_positions_robot1 is not None:
                            phys_positions = list(self.latest_physical_positions_robot1)
                            timestamp_physical = self.latest_timestamp_robot1
                    else:
                        if self.latest_physical_positions_robot2 is not None:
                            phys_positions = list(self.latest_physical_positions_robot2)
                            timestamp_physical = self.latest_timestamp_robot2

            # ===== MODO FÍSICO (suavizado desde lecturas físicas) =====
            if robot_connected and phys_positions:
                # Calcular latencia
                if timestamp_physical is not None:
                    latency_ms = (time.time() - timestamp_physical) * 1000.0
                    self.virtual_latency_log.append(latency_ms)
                    if len(self.virtual_latency_log) > 10000:
                        self.virtual_latency_log.pop(0)

                    if self.selected_robot == self.robot1:
                        self.last_virtual_latency_robot1 = latency_ms
                    else:
                        self.last_virtual_latency_robot2 = latency_ms

                # Inicializar buffer de suavizado si no existe
                if not hasattr(self, "_prev_phys_positions"):
                    self._prev_phys_positions = [None, None]

                idx = 0 if self.selected_robot == self.robot1 else 1

                if self._prev_phys_positions[idx] is None:
                    self._prev_phys_positions[idx] = list(phys_positions)

                prev = self._prev_phys_positions[idx]

                alpha = 0.1  # suavizado
                smoothed = [
                    prev[i] + alpha * (phys_positions[i] - prev[i])
                    for i in range(min(n_joints, len(phys_positions)))
                ]

                self._prev_phys_positions[idx] = smoothed

                for i in range(min(n_joints, len(smoothed))):
                    try:
                        degrees[i] = int(smoothed[i])
                    except Exception:
                        degrees[i] = 0

            # ===== MODO SIMULACIÓN (leer URDF) =====
            else:
                for i in range(n_joints):
                    pos = p.getJointState(self.selected_robot, i)[0]
                    deg = int(np.rad2deg(pos))
                    low, high = globals.JOINT_LIMITS[i]
                    deg = max(int(low), min(int(high), deg))
                    degrees[i] = deg

            # ===== ACTUALIZAR UI =====
            for i, deg in enumerate(degrees):
                sl = self.sliders_joints[i]
                sp = self.joint_spinboxes[i]

                sl.blockSignals(True)
                sp.blockSignals(True)

                sl.setValue(int(deg))
                sp.setValue(int(deg))

                sl.blockSignals(False)
                sp.blockSignals(False)

        except Exception as e:
            print(f"Error actualizando sliders desde robot seleccionado: {e}")





    # ----------------------------------------
    # Métodos existentes de control, gripper y trayectorias
    # ----------------------------------------
    def Open_Close(self):
        if self.OC.isChecked():
            self.gripper_state = 1
            # --- Control del gripper en simulador ---
            if self.finger_left_index is not None:
                p.setJointMotorControl2(self.selected_robot, self.finger_left_index, p.POSITION_CONTROL, targetPosition=0.6, force=50)
            if self.finger_right_index is not None:
                p.setJointMotorControl2(self.selected_robot, self.finger_right_index, p.POSITION_CONTROL, targetPosition=0.6, force=50)
            self.OC.setText("Gripper: Abierto")

            # --- Control físico ---
            self.physical.control_gripper(True)

        else:
            self.gripper_state = 0
            # --- Control del gripper en simulador ---
            if self.finger_left_index is not None:
                p.setJointMotorControl2(self.selected_robot, self.finger_left_index, p.POSITION_CONTROL, targetPosition=-0.08, force=50)
            if self.finger_right_index is not None:
                p.setJointMotorControl2(self.selected_robot, self.finger_right_index, p.POSITION_CONTROL, targetPosition=-0.08, force=50)
            self.OC.setText("Gripper: Cerrado")

            # --- Control físico ---
            self.physical.control_gripper(False)


    def execute_trajectory_from_file(self):
        """
        Carga y ejecuta una trayectoria desde un archivo JSON,
        ya sea de un solo robot o de ambos.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar trayectoria", "", "JSON Files (*.json)")
        if not path:
            return

        with open(path, "r") as f:
            data = json.load(f)

        # Si contiene ambos robots
        if isinstance(data, dict) and "robot1" in data and "robot2" in data:
            self.execute_dual_trajectory(data)
        else:
            # Trayectoria simple
            self.current_trajectory = data
            self.execute_on_robot()

    def execute_step(self, step):

        robot_choice = step.get("robot")
        joints_deg = step.get("joints", [])
        gripper = step.get("gripper", 0)
        wait_time = step.get("wait", 0.0)
        line_target = step.get("line", None)
        movement_mode = step.get("movement", "absolute")

        if robot_choice == "Robot 1":
            robot_instance = self.robot1_instance
            digital_robot = self.robot1
            kin_model = self.robot1_kin
            latency_ms = self.virtual.get_virtual_latency(robot_choice)
        elif robot_choice == "Robot 2":
            robot_instance = self.robot2_instance
            digital_robot = self.robot2
            kin_model = self.robot2_kin
            latency_ms = self.virtual.get_virtual_latency(robot_choice)
        else:
            print(f"⚠️ Robot desconocido: {robot_choice}")
            return None

        is_physical = robot_instance and robot_instance.ready_to_use()

        simulated_time_accumulator = 0.0
        start_time_real = time.time()

        # ==========================
        # ===== MOVIMIENTO =========
        # ==========================

        if is_physical:
            try:
                if line_target is not None:
                    absolute = True if movement_mode == "absolute" else False
                    robot_instance.move_line(line_target, absolute=absolute)

                elif joints_deg:
                    robot_instance.move_joints(joints_deg)

                if "gripper" in step:
                    robot_instance.gripper(bool(gripper))

                while not robot_instance.ready_to_use():
                    time.sleep(0.02)

            except Exception as e:
                print(f"❌ Error robot físico: {e}")

            elapsed_time = time.time() - start_time_real

        else:
            movement_time_simulated = 0.0

            if line_target is not None:

                x, y, z = line_target

                if movement_mode == "relative":
                    current_state = p.getLinkState(digital_robot, self.end_effector_index)
                    current_pos = current_state[0]
                    x += current_pos[0] * 1000
                    y += current_pos[1] * 1000
                    z += current_pos[2] * 1000

                alpha = 0
                beta = 0

                result = kin_model.xyz_to_joint([x, y, z, alpha, beta])

                if result is None or result["status"] == 2 or result["joint"] is None:
                    print("❌ IK inválida")
                    return None

                joints_deg = result["joint"].tolist()

                movement_time_simulated = self.virtual.virtual_move_line_blocking(
                    digital_robot,
                    kin_model,
                    [x, y, z]
                )

            elif joints_deg:

                movement_time_simulated = self.virtual.virtual_move_joints_blocking(
                    digital_robot,
                    joints_deg
                )

            simulated_time_accumulator += movement_time_simulated

            elapsed_time = simulated_time_accumulator  # 🔥 CLAVE

        #digital_robot = self.robot1 if robot_choice == "Robot 1" else self.robot2

           # Gripper virtual
        if self.finger_left_index is not None and self.finger_right_index is not None:
            if gripper:
                p.setJointMotorControl2(digital_robot, self.finger_left_index, p.POSITION_CONTROL, targetPosition=0.6, force=50)
                p.setJointMotorControl2(digital_robot, self.finger_right_index, p.POSITION_CONTROL, targetPosition=0.6, force=50)
            else:
                p.setJointMotorControl2(digital_robot, self.finger_left_index, p.POSITION_CONTROL, targetPosition=-0.08, force=50)
                p.setJointMotorControl2(digital_robot, self.finger_right_index, p.POSITION_CONTROL, targetPosition=-0.08, force=50)

            """ # --- animar el gripper: ejecutar varios pasos de simulación para que la pinza se mueva ---
            n_steps = max(1, int(0.15 / globals.TIME_STEP))  # 150 ms de animación (ajusta si quieres)
            for _ in range(n_steps):
                p.stepSimulation()
                time.sleep(globals.TIME_STEP)"""

        # ==========================
        # ===== WAIT (AMBOS) =======
        # ==========================

        if wait_time > 0:

            if is_physical:
                time.sleep(wait_time)
                elapsed_time += wait_time

            else:
                end_wait = time.time() + (wait_time / self.speed_multiplier)
                while time.time() < end_wait:
                    QCoreApplication.processEvents()
                    time.sleep(0.01 / self.speed_multiplier)

                elapsed_time += wait_time

        # ==========================
        # ===== LECTURA FINAL ======
        # ==========================

        joints_final = joints_deg
        xyz_real = [None, None, None]
        xyz_virtual = [None, None, None]

        if is_physical:
            try:
                pos = robot_instance.get_positions()
                if pos:
                    joints_final = pos[:5]
                    xyz_real = pos[5:8]
            except:
                pass

        try:
            if isinstance(joints_deg, (list, np.ndarray)) and len(joints_deg) >= 5:
                xyz_calc = kin_model.joint_to_xyz(joints_deg)
                if xyz_calc is not None:
                    xyz_virtual = xyz_calc[:3]
        except:
            pass

        return {
            "robot": robot_choice,
            "joints_cmd": joints_deg,
            "joints_final": joints_final,
            "x_real": xyz_real[0],
            "y_real": xyz_real[1],
            "z_real": xyz_real[2],
            "x_virtual": xyz_virtual[0],
            "y_virtual": xyz_virtual[1],
            "z_virtual": xyz_virtual[2],
            "gripper": gripper,
            "latency_ms": latency_ms,
            "wait_time": wait_time,
            "elapsed_time": round(elapsed_time, 3),
        }




    def execute_sequence(self, path=None, layout_id=None, repeat=1):
        layout_id = globals.current_layout_id
    # --- Seleccionar JSON ---
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Seleccionar secuencia", "", "JSON Files (*.json)"
            )
            if not path:
                return

        try:
            with open(path, "r") as f:
                data = json.load(f)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo leer el JSON:\n{e}")
            return

        sequence = data.get("sequence", data)
        if not sequence:
            QMessageBox.warning(self, "Advertencia", "La secuencia está vacía.")
            return

        # --- Guardar CSV ---
        csv_path =f"execution_{layout_id}.csv"

        # Deshabilitar botones mientras corre
        self.btn_execute_sequence.setEnabled(False)
        self.btn_create_sequence.setEnabled(False)

        # Crear worker y conectar señales
        self._sequence_worker = SequenceWorker(
                                                self, sequence, csv_path, repeat=repeat, layout_id=layout_id
                                            )
        self._sequence_worker.progress.connect(lambda run, step: print(f"Progress run {run} step {step}"))
        self._sequence_worker.finished.connect(self._on_sequence_finished)
        self._sequence_worker.error.connect(self._on_sequence_error)
        

        # iniciar
        self._sequence_worker.finished.connect(self._sequence_worker.deleteLater)
        self._sequence_worker.start()

    def _on_sequence_finished(self, csv_path):

        self.btn_execute_sequence.setEnabled(True)
        self.btn_create_sequence.setEnabled(True)

        self.rn_builder.build(csv_path)

        # 🔥 SOLO mostrar popup si NO estamos en batch
        if not getattr(self, "_batch_mode", False):
            QMessageBox.information(
                self,
                "Secuencia terminada",
                f"Dataset guardado:\n{csv_path}"
            )

        self.sequence_finished_signal.emit(csv_path)
        

    def _on_sequence_error(self, msg):
        self.btn_execute_sequence.setEnabled(True)
        self.btn_create_sequence.setEnabled(True)
        QMessageBox.critical(self, "Error en ejecución", msg)
    

    def update_joint_positions(self):
        """
        Cuando el usuario mueve sliders:
        - Actualiza spinboxes.
        - Si el robot físico está conectado, envía MOVE_JOINTS.
        - Si NO hay robot físico, mueve solo el simulador.

        Nota: ya NO emitimos la señal update_sliders_signal aquí para evitar
        sobrescrituras inmediatas de la UI.
        """
        now = time.time()
        if self.selected_robot == self.robot1:
            self._last_manual_update_robot1 = now
        else:
            self._last_manual_update_robot2 = now

        # 1) Leer sliders y sincronizar spinboxes
        targets_deg = []
        for i in range(len(globals.JOINT_LIMITS)):
            deg = int(self.sliders_joints[i].value())

            # actualizar spinbox sin disparar señales
            self.joint_spinboxes[i].blockSignals(True)
            self.joint_spinboxes[i].setValue(deg)
            self.joint_spinboxes[i].blockSignals(False)

            targets_deg.append(deg)

        # 2) Convertir a radianes para el simulador
        targets_rad = [math.radians(d) for d in targets_deg]

        # 3) Seleccionar instancia física
        instance = None
        if self.selected_robot == self.robot1:
            instance = self.robot1_instance
        elif self.selected_robot == self.robot2:
            instance = self.robot2_instance

        # 4) Enviar al robot físico (si existe)
        if instance:
            def send_to_physical():
                try:
                    # debug: ver lo que se manda
                    print(f"[DEBUG] Enviando a físico ({'R1' if self.selected_robot==self.robot1 else 'R2'}): {targets_deg}")
                    instance.move_joints(targets_deg)  # grados
                except Exception as e:
                    print(f"Error enviando joints al robot físico: {e}")

            threading.Thread(target=send_to_physical, daemon=True).start()

            # También actualizamos objetivos virtuales para feedback inmediato (radianes)
            with self._phys_lock:
                if self.selected_robot == self.robot1:
                    self.latest_virtual_targets_robot1 = targets_rad
                else:
                    self.latest_virtual_targets_robot2 = targets_rad

            # Aplicar inmediatamente al URDF para que el visual responda ahora mismo
            try:
                set_joint_positions(self.selected_robot, targets_rad)
                # NO forzamos update_joint_sliders_from_robot() ni emitimos la señal aquí:
                # la actualización periódica o el hilo del robot se encargará de sincronizar y
                # la comprobación de _last_manual_update_* evitará sobrescrituras inmediatas.
            except Exception as e:
                print(f"Error aplicando posiciones al simulador tras mover físico: {e}")

        # 5) Si no hay físico → simulador directo
        else:
            try:
                set_joint_positions(self.selected_robot, targets_rad)
                # No emitimos la señal aquí: la actualización periódica leerá del URDF y refrescará
                # los sliders, pero respetará el tiempo manual (_manual_hang_seconds).
            except Exception as e:
                print(f"Error aplicando posiciones al robot (sim): {e}")





    def refresh_urdf_list(self): 
        self.urdf_list.clear() 
        for body_id in range(p.getNumBodies()): 
            try:
                info = p.getBodyInfo(body_id)[1].decode() 
                pos, _ = p.getBasePositionAndOrientation(body_id) 
                self.urdf_list.addItem(f"{body_id}: {info} @ [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            except Exception as e:
                print(f"Error listando body {body_id}: {e}")
 

    def delete_selected_urdf(self): 
        sel = self.urdf_list.currentItem() 
        if sel: 
            body_id = int(sel.text().split(":")[0]) 
            try:
                p.removeBody(body_id)
            except Exception as e:
                print(f"Error removiendo body {body_id}: {e}")
            self.refresh_urdf_list()

    def apply_manual_angle(self, joint_index):
        text = self.manual_inputs[joint_index].text()
        try:
            value = float(text)
        except ValueError:
            return

        low, high = globals.JOINT_LIMITS[joint_index]
        value = max(low, min(high, value))

        # Mover slider → dispara update_joint_positions
        self.sliders_joints[joint_index].setValue(int(value))
    
    def get_active_instance(self):
        return (
            self.robot1_instance
            if self.control_robot_selector.currentText() == "Robot 1"
            else self.robot2_instance
        )

    def move_to_homePos(self):
        instance = self.get_active_instance()
        if not instance:
            QMessageBox.warning(self, "Aviso", "Robot no conectado")
            return

        instance.HomePosition()
