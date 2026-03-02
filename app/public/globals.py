JOINT_LIMITS = [
    (-175, 180),  # Joint 0
    (-90,  180),  # Joint 1
    (-142, 142),  # Joint 2
    (-135, 135),  # Joint 3
    (-360, 360),  # Joint 4
]

station_ids = []
feeder_ids = []
current_layout_id = None

HOME_POSITIONS = [0, 2.530727415391778, -1.5707963267948966, 0, 0]
MOVE_STEP = 0.2  
TIME_STEP = 0.001   

rbt = "./app/public/Assets/URDF/Dorna_URDF2/urdf/Dorna_URDF2.urdf"
mt = "./app/public/Assets/URDF/WorkBench/urdf/WorkBench.urdf"
clz = "./app/public/Assets/URDF/BasePlate/urdf/BasePlate.urdf"
est = "./app/public/Assets/URDF/Estationv3/urdf/Estationv3.urdf"
cube = "./app/public/Assets/URDF/Cube/urdf/Cube.urdf"
feeder = "./app/public/Assets/URDF/Feeder/urdf/Feeder.urdf"


GRAVITY = -9.81
CAMERA_DISTANCE = 1.5
CAMERA_YAW = 90
CAMERA_PITCH = -30
CAMERA_TARGET_POS = [0.47, 0.9, 0.6]