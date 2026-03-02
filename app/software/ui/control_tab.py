from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QSpinBox
)
from PyQt6.QtCore import Qt
import numpy as np
import pybullet as p
import math
import globals
from utils import set_joint_positions


class ControlTab(QWidget):
    def __init__(self, robot1_urdf, robot2_urdf, conexion_tab):
        super().__init__()
        self.robot1_urdf = robot1_urdf
        self.robot2_urdf = robot2_urdf
        self.conexion_tab = conexion_tab  # Para acceder a las instancias físicas
        self.sliders_r1 = {}
        self.sliders_r2 = {}

        layout = QVBoxLayout(self)

        # --- Panel Robot 1 ---
        grp1 = QGroupBox("Control Robot 1")
        grp1_layout = QVBoxLayout()
        for i, (low, high) in enumerate(globals.JOINT_LIMITS):
            name = p.getJointInfo(self.robot1_urdf, i)[1].decode()
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}:"))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(low, high)
            slider.valueChanged.connect(lambda v, j=i: self._on_slider_changed(1, j, v))
            spin = QSpinBox(); spin.setRange(low, high)
            spin.valueChanged.connect(lambda v, sl=slider: sl.setValue(v))
            row.addWidget(slider)
            row.addWidget(spin)
            grp1_layout.addLayout(row)
            self.sliders_r1[i] = slider
        grp1.setLayout(grp1_layout)
        layout.addWidget(grp1)

        # --- Panel Robot 2 ---
        grp2 = QGroupBox("Control Robot 2")
        grp2_layout = QVBoxLayout()
        for i, (low, high) in enumerate(globals.JOINT_LIMITS):
            name = p.getJointInfo(self.robot2_urdf, i)[1].decode()
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}:"))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(low, high)
            slider.valueChanged.connect(lambda v, j=i: self._on_slider_changed(2, j, v))
            spin = QSpinBox(); spin.setRange(low, high)
            spin.valueChanged.connect(lambda v, sl=slider: sl.setValue(v))
            row.addWidget(slider)
            row.addWidget(spin)
            grp2_layout.addLayout(row)
            self.sliders_r2[i] = slider
        grp2.setLayout(grp2_layout)
        layout.addWidget(grp2)

        # --- Botones de control conjunto ---
        btns = QHBoxLayout()
        self.btn_home_all = QPushButton("🏠 Home All Robots")
        self.btn_home_all.clicked.connect(self.home_all)
        btns.addWidget(self.btn_home_all)

        self.btn_sync = QPushButton("🔄 Sincronizar Sliders con Robots Físicos")
        self.btn_sync.clicked.connect(self.sync_from_physical)
        btns.addWidget(self.btn_sync)
        layout.addLayout(btns)

    # ------------------------------------------------------------------
    # 🔹 Cuando se mueve un slider (URDF y robot físico)
    # ------------------------------------------------------------------
    def _on_slider_changed(self, robot_id, joint_index, value):
        urdf = self.robot1_urdf if robot_id == 1 else self.robot2_urdf
        set_joint_positions(urdf, [math.radians(value) if i == joint_index else p.getJointState(urdf, i)[0] 
                                   for i in range(len(globals.JOINT_LIMITS))])

        # Enviar al robot físico si está conectado
        robot_thread = self.conexion_tab.robot1_thread if robot_id == 1 else self.conexion_tab.robot2_thread
        if robot_thread and robot_thread.robot.ready_to_use():
            joint_values = [math.degrees(p.getJointState(urdf, i)[0]) for i in range(5)]
            robot_thread.robot.move_joints(joint_values)

    # ------------------------------------------------------------------
    # 🔹 Manda a home todos los robots
    # ------------------------------------------------------------------
    def home_all(self):
        if self.conexion_tab.robot1_thread:
            self.conexion_tab.robot1_thread.home()
        if self.conexion_tab.robot2_thread:
            self.conexion_tab.robot2_thread.home()

    # ------------------------------------------------------------------
    # 🔹 Actualiza sliders desde el estado físico actual
    # ------------------------------------------------------------------
    def sync_from_physical(self):
        if self.conexion_tab.robot1_thread:
            try:
                joints1 = self.conexion_tab.robot1_thread.robot.read_responses()[:5]
                for i, j in enumerate(joints1):
                    self.sliders_r1[i].setValue(int(j))
            except Exception as e:
                print("Error sincronizando Robot 1:", e)

        if self.conexion_tab.robot2_thread:
            try:
                joints2 = self.conexion_tab.robot2_thread.robot.read_responses()[:5]
                for i, j in enumerate(joints2):
                    self.sliders_r2[i].setValue(int(j))
            except Exception as e:
                print("Error sincronizando Robot 2:", e)
