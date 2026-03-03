# controllers/sequence_worker.py

import csv
import time
from PyQt6.QtCore import QThread, pyqtSignal


class SequenceWorker(QThread):

    progress = pyqtSignal(int, int)   # run_idx, step_idx
    finished = pyqtSignal(str)        # csv_path
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, controller, sequence, csv_path, repeat=1, layout_id=None, parent=None):
        super().__init__(parent)

        self.controller = controller
        self.sequence = sequence
        self.csv_path = csv_path
        self.layout_id = layout_id
        self.repeat = repeat
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:

            total_elapsed_simulated = 0.0

            header = [
                "layout_id","run", "step", "robot",
                "j_cmd_0","j_cmd_1","j_cmd_2","j_cmd_3","j_cmd_4",
                "gripper_cmd",
                "j_final_0","j_final_1","j_final_2","j_final_3","j_final_4",
                "x_real","y_real","z_real",
                "x_virtual","y_virtual","z_virtual",
                "latency_ms","wait_time","elapsed_time","total_elapsed_time"
            ]

            with open(self.csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

                for run_idx in range(1, self.repeat + 1):

                    if self._stop:
                        break

                    for step_idx, step in enumerate(self.sequence):

                        if self._stop:
                            break

                        result = self.controller.execute_step(step)

                        if not result:
                            continue

                        total_elapsed_simulated += result["elapsed_time"]

                        row = [
                            self.layout_id,
                            run_idx,
                            step_idx + 1,
                            result["robot"],
                            *result["joints_cmd"][:5],
                            result["gripper"],
                            *result["joints_final"][:5],
                            result["x_real"],
                            result["y_real"],
                            result["z_real"],
                            result["x_virtual"],
                            result["y_virtual"],
                            result["z_virtual"],
                            result["latency_ms"],
                            result["wait_time"],
                            result["elapsed_time"],
                            round(total_elapsed_simulated, 3)
                        ]

                        writer.writerow(row)

                        self.progress.emit(run_idx, step_idx + 1)

            self.finished.emit(self.csv_path)

        except Exception as e:
            self.error.emit(str(e))