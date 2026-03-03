# controllers/virtual_controller.py

import time
import math
import numpy as np
import pybullet as p
from PyQt6.QtCore import QCoreApplication
import app.public.globals as globals
from app.public.utils import set_joint_positions


class VirtualRobotController:
    """
    Controlador encargado únicamente del comportamiento virtual (simulación).
    No maneja UI directamente.
    """

    def __init__(self, parent_controller):
        self.parent = parent_controller  # referencia al RobotController


    # ==========================================================
    # LATENCIA
    # ==========================================================
    def get_virtual_latency(self, robot_choice):
        if robot_choice == "Robot 1":
            last_lat = getattr(self.parent, "last_virtual_latency_robot1", None)
        else:
            last_lat = getattr(self.parent, "last_virtual_latency_robot2", None)

        return round(last_lat, 3) if last_lat is not None else ""


    # ==========================================================
    # MOVIMIENTO EN JOINTS (SIMULADO)
    # ==========================================================
    def virtual_move_joints_blocking(
        self,
        robot_id,
        target_deg,
        speed_deg_s=28,
        tol=0.01
    ):

        target_rad = np.radians(target_deg)

        current_rad = np.array([
            p.getJointState(robot_id, i)[0]
            for i in range(len(target_rad))
        ])

        max_delta = np.max(np.abs(target_rad - current_rad))

        move_time_real = max_delta / math.radians(speed_deg_s * self.parent.speed_multiplier)
        move_time_simulated = max_delta / math.radians(speed_deg_s)

        steps = max(1, int(move_time_real / globals.TIME_STEP))

        for step in range(steps + 1):
            alpha = step / steps
            interp = current_rad + alpha * (target_rad - current_rad)

            set_joint_positions(robot_id, interp)

            p.stepSimulation()

            QCoreApplication.processEvents()
            self.parent.update_joint_sliders_from_robot()

            time.sleep(globals.TIME_STEP / self.parent.speed_multiplier)

        return move_time_simulated


    # ==========================================================
    # MOVIMIENTO LINEAL CARTESIANO (SIMULADO)
    # ==========================================================
    def virtual_move_line_blocking(
        self,
        digital_robot,
        kin_model,
        target_xyz,
        speed_mm_s=100,
        steps=50
    ):

        current_joints = [
            math.degrees(p.getJointState(digital_robot, i)[0])
            for i in range(5)
        ]

        current_xyz = kin_model.joint_to_xyz(current_joints)

        if current_xyz is None:
            print("❌ No se pudo obtener FK actual")
            return 0.0

        x0, y0, z0 = current_xyz[:3]
        x1, y1, z1 = target_xyz
        alpha_current = current_xyz[3]
        beta_current = current_xyz[4]

        for step in range(steps + 1):

            alpha = step / steps

            xi = x0 + alpha * (x1 - x0)
            yi = y0 + alpha * (y1 - y0)
            zi = z0 + alpha * (z1 - z0)

            result = kin_model.xyz_to_joint(
                [xi, yi, zi, alpha_current, beta_current]
            )

            if result is None or result["status"] == 2:
                print("⚠️ Límite workspace alcanzado")
                break

            joints = result["joint"]

            set_joint_positions(digital_robot, np.radians(joints))

            p.stepSimulation()

            QCoreApplication.processEvents()
            self.parent.update_joint_sliders_from_robot()

            time.sleep(globals.TIME_STEP / self.parent.speed_multiplier)

        # estimación simple de tiempo
        distance = math.dist([x0, y0, z0], [x1, y1, z1])
        return distance / speed_mm_s