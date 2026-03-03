# controllers/physical_sync.py

import time
import math
import threading
import pybullet as p
import app.public.globals as globals
from app.public.utils import set_joint_positions


class PhysicalRobotSync:
    """
    Maneja toda la sincronización físico ↔ digital.
    No contiene UI directa.
    """

    def __init__(self, parent_controller):
        self.parent = parent_controller


    # ==========================================================
    # HILO PRINCIPAL DE LECTURA FÍSICA
    # ==========================================================
    def start_robot_thread(self, robot_instance, digital_robot):

        def loop():
            print(f"[HILO] Iniciando escucha para {digital_robot}")

            while getattr(robot_instance, "running", True):

                try:
                    joints = robot_instance.read_responses()

                    if joints is not None and len(joints) >= 5:

                        ts = time.time()

                        with self.parent._phys_lock:

                            if digital_robot == self.parent.robot1:
                                self.parent.robot1_connected = True
                                self.parent.latest_physical_positions_robot1 = joints[:5]
                                self.parent.latest_timestamp_robot1 = ts
                            else:
                                self.parent.robot2_connected = True
                                self.parent.latest_physical_positions_robot2 = joints[:5]
                                self.parent.latest_timestamp_robot2 = ts

                        # 🔹 Filtro exponencial
                        target_angles = [math.radians(j) for j in joints[:5]]
                        beta = 0.2

                        if digital_robot == self.parent.robot1:

                            if not hasattr(self.parent, "filtered_targets_robot1"):
                                self.parent.filtered_targets_robot1 = target_angles
                            else:
                                self.parent.filtered_targets_robot1 = [
                                    self.parent.filtered_targets_robot1[i] +
                                    beta * (target_angles[i] - self.parent.filtered_targets_robot1[i])
                                    for i in range(len(target_angles))
                                ]

                            self.parent.latest_virtual_targets_robot1 = self.parent.filtered_targets_robot1

                        else:

                            if not hasattr(self.parent, "filtered_targets_robot2"):
                                self.parent.filtered_targets_robot2 = target_angles
                            else:
                                self.parent.filtered_targets_robot2 = [
                                    self.parent.filtered_targets_robot2[i] +
                                    beta * (target_angles[i] - self.parent.filtered_targets_robot2[i])
                                    for i in range(len(target_angles))
                                ]

                            self.parent.latest_virtual_targets_robot2 = self.parent.filtered_targets_robot2

                        try:
                            self.parent.slider_updater.update_sliders_signal.emit()
                        except:
                            pass

                    # ===============================
                    # Aplicar targets suavizados
                    # ===============================
                    if digital_robot == self.parent.robot1 and hasattr(self.parent, "latest_virtual_targets_robot1"):
                        target = self.parent.latest_virtual_targets_robot1
                    elif digital_robot == self.parent.robot2 and hasattr(self.parent, "latest_virtual_targets_robot2"):
                        target = self.parent.latest_virtual_targets_robot2
                    else:
                        target = None

                    if target is not None:

                        joint_indices = list(range(5))
                        current_states = p.getJointStates(digital_robot, joint_indices)
                        current_positions = [s[0] for s in current_states]

                        alpha = 0.15
                        deadband = 0.01

                        smooth_positions = []

                        for i in range(len(current_positions)):
                            error = target[i] - current_positions[i]

                            if abs(error) < deadband:
                                smooth_positions.append(current_positions[i])
                            else:
                                smooth_positions.append(
                                    current_positions[i] + alpha * error
                                )

                        set_joint_positions(digital_robot, smooth_positions)

                        p.stepSimulation()

                except Exception as e:
                    print(f"Error en hilo del robot {digital_robot}: {e}")

                time.sleep(globals.TIME_STEP)

            print(f"[HILO] Finalizando escucha para {digital_robot}")

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t


    # ==========================================================
    # HOMING
    # ==========================================================
    def home_robot_with_digital(self, robot_instance, digital_robot):

        def loop():
            try:
                robot_instance.homing()

                while not robot_instance.ready_to_use():

                    joints = robot_instance.read_responses()

                    if joints:
                        target_angles = [math.radians(j) for j in joints[:5]]
                        set_joint_positions(digital_robot, target_angles)

                    time.sleep(globals.TIME_STEP)

            except Exception as e:
                print(f"Error en homing: {e}")

        threading.Thread(target=loop, daemon=True).start()


    def home_all(self):
        if self.parent.robot1_instance:
            self.home_robot_with_digital(
                self.parent.robot1_instance,
                self.parent.robot1
            )

        if self.parent.robot2_instance:
            self.home_robot_with_digital(
                self.parent.robot2_instance,
                self.parent.robot2
            )


    # ==========================================================
    # GRIPPER FÍSICO
    # ==========================================================
    def control_gripper(self, open_gripper: bool):

        instance = None

        if self.parent.selected_robot == self.parent.robot1:
            instance = self.parent.robot1_instance
        elif self.parent.selected_robot == self.parent.robot2:
            instance = self.parent.robot2_instance

        if instance:
            try:
                instance.gripper(open_gripper)
            except Exception as e:
                print(f"⚠️ Error controlando gripper físico: {e}")


    # ==========================================================
    # CALLBACK LATENCIA
    # ==========================================================
    def on_physical_state(self, robot_id, joints_rad, t_physical):

        for i, q in enumerate(joints_rad):
            p.resetJointState(robot_id, i, q)

        t_virtual = time.time()
        self.parent.last_latency_ms = round(
            (t_virtual - t_physical) * 1000,
            3
        )