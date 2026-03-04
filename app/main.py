import sys
from PyQt6.QtWidgets import QApplication
from app.software.simulation.controller import RobotController
import app.software.simulation.simulation_settings as simulation_settings
import app.software.simulation.scene as scene


def main():

    simulation_settings.initialize_simulation()
    scene.load_scene()

    # 🔥 MUY IMPORTANTE: usar sys.argv
    app = QApplication(sys.argv)

    controller = RobotController()
    controller.resize(550, 780)
    controller.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()