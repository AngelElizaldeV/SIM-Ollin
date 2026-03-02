import pybullet as p
import pybullet_data
import app.public.globals as globals


def initialize_simulation():
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, globals.GRAVITY)
    p.resetDebugVisualizerCamera(
        globals.CAMERA_DISTANCE,
        globals.CAMERA_YAW,
        globals.CAMERA_PITCH,
        globals.CAMERA_TARGET_POS,
    )
