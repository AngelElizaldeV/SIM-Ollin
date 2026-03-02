import pybullet as p
import numpy as np

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
    import pybullet as p

    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    link_state = p.getLinkState(robot_id, ee_index)
    ee_world = link_state[0]

    inv_base_pos, inv_base_orn = p.invertTransform(base_pos, base_orn)
    ee_local, _ = p.multiplyTransforms(
        inv_base_pos, inv_base_orn,
        ee_world, [0,0,0,1]
    )

    return ee_local


