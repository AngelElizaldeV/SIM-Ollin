import pybullet as p
import numpy as np
import app.public.globals as globals

def set_joint_positions(robot, positions):
    for i, pos in enumerate(positions):
        p.setJointMotorControl2(
            robot,
            i,
            p.POSITION_CONTROL,
            targetPosition=pos,
            force=120,        # mucho menor
            positionGain=0.08,
            velocityGain=1.0
        )
        
def get_end_effector_position(robot_id, ee_index):

    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    link_state = p.getLinkState(robot_id, ee_index)
    ee_world = link_state[0]

    inv_base_pos, inv_base_orn = p.invertTransform(base_pos, base_orn)
    ee_local, _ = p.multiplyTransforms(
        inv_base_pos, inv_base_orn,
        ee_world, [0,0,0,1]
    )

    return ee_local


def update_station_positions(new_positions):
    """
    Actualiza las posiciones de:
    - Feeder 1
    - 7 estaciones de proceso
    - Feeder 2
    Manteniendo la altura Z actual.
    """

    if len(new_positions) != 9:
        print("Se esperaban 9 posiciones.")
        return

    feeder1 = new_positions[0]
    process_positions = new_positions[1:8]
    feeder2 = new_positions[8]

    # 🔴 Feeder 1
    sid = globals.feeder_ids[0]
    current_pos, current_orn = p.getBasePositionAndOrientation(sid)

    p.resetBasePositionAndOrientation(
        sid,
        [feeder1[0], feeder1[1], current_pos[2]],
        current_orn
    )

    # 🔵 Estaciones proceso
    for sid, (x, y) in zip(globals.station_ids, process_positions):

        current_pos, current_orn = p.getBasePositionAndOrientation(sid)

        p.resetBasePositionAndOrientation(
            sid,
            [x, y, current_pos[2]],
            current_orn
        )

    # 🔴 Feeder 2
    sid = globals.feeder_ids[1]
    current_pos, current_orn = p.getBasePositionAndOrientation(sid)

    p.resetBasePositionAndOrientation(
        sid,
        [feeder2[0], feeder2[1], current_pos[2]],
        current_orn
    )


