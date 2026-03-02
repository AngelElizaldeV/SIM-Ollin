from dorna import Dorna
import json
import threading
import time

class DornaController:
    def __init__(self, yaml_path, com_port=None):
        """
        Crea una instancia de un robot Dorna, conecta y lanza un hilo
        para escuchar las respuestas.
        """
        self.yaml_path = yaml_path
        self.com_port = com_port
        self.robot = Dorna(self.yaml_path)
        self.openServo = 300
        self.closedServo = 30

        try:
            if self.com_port:
                self.robot.connect(self.com_port)
                print(f"✅ Robot conectado en {self.com_port}")
            else:
                self.robot.connect()
                print("✅ Robot conectado sin COM explícito")
        except Exception as e:
            print(f"❌ Error al conectar el robot: {e}")
            return

        # Bandera de ejecución del hilo
        self._running = True
        # Hilo de escucha de respuestas
        self.thread = threading.Thread(target=self._response_listener, daemon=True)
        self.thread.start()

    # -----------------------------
    # MÉTODOS PRINCIPALES
    # -----------------------------
    def ready_to_use(self):
        """Verifica si el robot está listo para moverse."""
        try:
            estado = json.loads(self.robot.device())
            return estado.get("state", -1) == 0
        except Exception as e:
            print(f"⚠️ Error al leer estado del robot: {e}")
            return False

    def move_joints(self, joints, absolute = False):
        """Mueve el robot en modo articular (joint)."""

        movement_mode = 0 if absolute else 1
        cmd = {
            "command": "move",
            "prm": {
                "path": "joint",
                "movement": 0,
                "speed": 10000,
                "j0": joints[0],
                "j1": joints[1],
                "j2": joints[2],
                "j3": joints[3],
                "j4": joints[4],
            },
        }
        self.robot.play(cmd, append=False)

    def move_line(self, xyz, absolute=False):
        """Mueve el efector en línea recta (XYZ)."""

        movement_mode = 0 if absolute else 1

        cmd = {
            "command": "move",
            "prm": {
                "path": "line",
                "movement": movement_mode,
                "speed": 10000,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
            },
        }

        self.robot.play(cmd, append=False)


    def homing(self):
        """Ejecuta la secuencia completa de homing."""
        try:
            print(f"🏠 Iniciando homing en {self.com_port or 'robot'}...")
            self.robot.homed()
            self.robot.home(["j0", "j1", "j2", "j3", "j4"])
            time.sleep(1)
            home_pose = {
                "command": "move",
                "prm": {
                    "path": "joint",
                    "movement": 0,
                    "speed": 10000,
                    "j0": 0, "j1": 145, "j2": -90, "j3": 0, "j4": 0
                },
            }
            self.robot.play(home_pose)
            close = {"command": "set_io", "prm": {"servo": 60}}
            self.robot.play(close)
            print(f"✅ Homing completado en {self.com_port or 'robot'}.")
        except Exception as e:
            print(f"❌ Error durante homing: {e}")

    def move00(self):
        try:
            print(f"Moviendo a 0 en {self.com_port or 'robot'}...")
            zero = {
                "command": "move",
                "prm": {
                    "path": "joint",
                    "movement": 0,
                    "speed": 10000,
                    "j0": 0, "j1": 0, "j2": 0, "j3": 0, "j4": 0
                },
            }
            self.robot.play(zero)

        except Exception as e:
            print("Error Inesperado")

    def HomePosition(self):
        try:
            print(f"Moviendo a 0 en {self.com_port or 'robot'}...")
            HP = {
                "command": "move",
                "prm": {
                    "path": "joint",
                    "movement": 0,
                    "speed": 10000,
                    "j0": 0, "j1": 145, "j2": -90, "j3": 0, "j4": 0
                },
            }
            self.robot.play(HP)

        except Exception as e:
            print("Error Inesperado")



    def get_positions(self):
        """Devuelve las posiciones actuales (joints + xyz)."""
        try:
            joints = json.loads(self.robot.position())
            xyz = json.loads(self.robot.position("xyz"))
            return joints + xyz
        except Exception as e:
            print(f"⚠️ Error al leer posiciones: {e}")
            return None

    # -----------------------------
    # HILO DE ESCUCHA DE RESPUESTAS
    # -----------------------------
    def _response_listener(self):
        """Hilo que escucha continuamente las respuestas del robot."""
        while self._running:
            try:
                resp = self.robot.read()
                if resp:
                    print(f"[{self.com_port}] → {resp.strip()}")
            except Exception:
                pass
            time.sleep(0.05)

    def read_responses(self):
        """Lee las posiciones actuales (joints + XYZ)."""
        try:
            joints = json.loads(self.robot.position())
            xyz = json.loads(self.robot.position("xyz"))
            return joints + xyz
        except Exception as e:
            print(f"⚠️ Error al leer posiciones del robot en {self.com_port}: {e}")
            return None

    def stop(self):
        """Detiene el hilo de escucha."""
        self._running = False
        print(f"🛑 Hilo detenido para {self.com_port}")


    def gripper(self, open_gripper: bool):
        try:
        # --- Control del gripper físico (Dorna) ---
            if open_gripper:
                # Cerrar gripper
                cmd = {"command": "set_io", "prm": {"servo": self.openServo}}
            else:
                # Abrir gripper
                cmd = {"command": "set_io", "prm": {"servo": self.closedServo}}
            self.robot.play(cmd)

            print(f"✅ Gripper {'cerrado' if open else 'abierto'} correctamente.")

        except Exception as e:
            print(f"⚠️ Error al controlar el gripper: {e}")

    def disconnect(self):

        self.robot.disconnect()

    
        