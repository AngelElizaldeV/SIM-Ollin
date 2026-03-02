import pybullet as p
import app.public.globals as globals

RBT, MT, CLZ, EST, CUBE, FEEDER = globals.rbt, globals.mt, globals.clz, globals.est, globals.cube, globals.feeder

def load_scene():

    p.loadURDF("plane.urdf")
    p.loadURDF(MT, basePosition=[0, 0, 0], useFixedBase=True)

    globals.station_ids = []   # 🔥 IMPORTANTE

    for pos in [[0.47, 0.60, 0.6], [0.47, 1.15, 0.6]]:
        p.loadURDF(CLZ, basePosition=pos, useFixedBase=True)

    stations = [
        ([0.80, 0.525, 0.632], [1, 0, 0, 1]),
        ([0.80, 0.655, 0.632], [0.2, 0.2, 0.2, 1]),
        ([0.80, 0.79, 0.632], [1, 0, 0, 1]),
        ([0.70, 0.93, 0.632], [0.2, 0.2, 0.2, 1]),
        ([0.80, 1.06, 0.632], [1, 0, 0, 1]),
        ([0.80, 1.19, 0.632], [0.2, 0.2, 0.2, 1]),
        ([0.80, 1.325, 0.632], [1, 0, 0, 1])
    ]

    for pos, color in stations:
        sid = p.loadURDF(EST, basePosition=pos, useFixedBase=True)
        p.changeVisualShape(sid, -1, rgbaColor=color)

        globals.station_ids.append(sid)  # 🔥 GUARDAMOS ID

    feeders = [
         ([0.61, 0.39, 0.667]),
         ([0.61, 1.45, 0.667])

    ]

    for pos in feeders:
         sid = p.loadURDF(FEEDER, basePosition = pos, useFixedBase = True)
         globals.feeder_ids.append(sid)



def load_robot(position, fixed):
    robot_id  = p.loadURDF(RBT, basePosition=position, useFixedBase=fixed)
    for robot in [robot_id]:
            for link_idx in range(p.getNumJoints(robot) - 2):
                p.changeVisualShape(robot, link_idx, rgbaColor=[0.2, 0.2, 0.2, 1.0])
    return robot_id

def load_asset():
     cubo = p.loadURDF("cube_small.urdf", [0.61, 0.39, 0.7], globalScaling=0.5)
     return cubo