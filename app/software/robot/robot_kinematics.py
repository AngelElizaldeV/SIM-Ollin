# robot_kinematics.py
import math
import numpy as np

class RobotKinematics:

    def __init__(self):

        # Dimensiones en mm
        self._bx = 3.759 * 25.4
        self._bz = 8.111 * 25.4
        self._l1 = 8.0 * 25.4
        self._l2 = 6.0 * 25.4

        self._tool_x = 44.196
        self._delta_e = 0.001


    def f_k(self, joint):

        try:
            teta = [math.radians(j) for j in joint[:5]]

            tmp = (
                self._bx
                + self._l1 * math.cos(teta[1])
                + self._l2 * math.cos(teta[1] + teta[2])
                + self._tool_x * math.cos(teta[1] + teta[2] + teta[3])
            )

            x = tmp * math.cos(teta[0])
            y = tmp * math.sin(teta[0])
            z = (
                self._bz
                + self._l1 * math.sin(teta[1])
                + self._l2 * math.sin(teta[1] + teta[2])
                + self._tool_x * math.sin(teta[1] + teta[2] + teta[3])
            )

            alpha = math.degrees(teta[1] + teta[2] + teta[3])
            beta = math.degrees(teta[4])

            return [x, y, z, alpha, beta]

        except Exception as e:
            print("FK error:", e)
            return None


    def i_k(self, xyz):

        try:
            x, y, z, alpha, beta = xyz[:5]

            alpha = math.radians(alpha)
            beta = math.radians(beta)

            teta_0 = math.atan2(y, x)

            x_plane = math.sqrt(x ** 2 + y ** 2)

            x_plane -= (self._bx + self._tool_x * math.cos(alpha))
            z -= (self._bz + self._tool_x * math.sin(alpha))

            L = math.sqrt(x_plane ** 2 + z ** 2)

            if L > (self._l1 + self._l2):
                return None

            teta_l1_L = math.acos(
                (self._l1 ** 2 + L ** 2 - self._l2 ** 2)
                / (2 * self._l1 * L)
            )

            teta_L_x = math.atan2(z, x_plane)

            teta_1 = teta_l1_L + teta_L_x

            teta_l1_l2 = math.acos(
                (self._l1 ** 2 + self._l2 ** 2 - L ** 2)
                / (2 * self._l1 * self._l2)
            )

            teta_2 = teta_l1_l2 - math.pi

            teta_3 = alpha - teta_1 - teta_2
            teta_4 = beta

            return [
                math.degrees(teta_0),
                math.degrees(teta_1),
                math.degrees(teta_2),
                math.degrees(teta_3),
                math.degrees(teta_4),
            ]

        except Exception as e:
            print("IK error:", e)
            return None
        

    def joint_to_xyz(self, joint): 
            if joint is None:
                return None

            if not isinstance(joint, (list, np.ndarray)):
                return None

            if any(j is None for j in list(joint)):
                return None



            # joint to radian
            teta_0 = math.radians(joint[0])
            teta_1 = math.radians(joint[1])
            teta_2 = math.radians(joint[2])
            teta_3 = math.radians(joint[3])
            teta_4 = math.radians(joint[4])

            # first we find x, y, z assuming base rotation is zero (teta_0 = 0). Then we rotate everything
            # then we rotate the robot around z axis for teta_0
            tmp = self._bx + self._l1 * math.cos(teta_1) + self._l2 * math.cos(teta_1 + teta_2) + self._tool_x * math.cos(teta_1 + teta_2 + teta_3)
            x = tmp * math.cos(teta_0)
            y = tmp * math.sin(teta_0)
            z = self._bz + self._l1 * math.sin(teta_1) + self._l2 * math.sin(teta_1 + teta_2) + self._tool_x * math.sin(teta_1 + teta_2 + teta_3)
            alpha = teta_1 + teta_2 + teta_3
            beta = teta_4

            alpha = math.degrees(alpha)
            beta = math.degrees(beta)

            if len(joint) == 6:
                return np.array([x, y, z, alpha, beta, joint[5]]) # [x, y, z, alpha, beta, joints[5]]
            else:
                return np.array([x, y, z, alpha, beta]) # [x, y, z, alpha, beta]


    def xyz_to_joint(self,xyz):
          
            if xyz is None:
                return None

            if not isinstance(xyz, (list, np.ndarray)):
                return None

            if any(j is None for j in list(xyz)):
                return None


            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
            alpha = xyz[3]
            beta = xyz[4]

            alpha = math.radians(alpha)
            beta = math.radians(beta)

            # first we find the base rotation
            teta_0 = math.atan2(y, x)

            # next we assume base is not rotated and everything lives in x-z plane
            x = math.sqrt(x ** 2 + y ** 2)

            # next we update x and z based on base dimensions and hand orientation
            x -= (self._bx + self._tool_x * math.cos(alpha))
            z -= (self._bz + self._tool_x * math.sin(alpha))

            # at this point x and z are the summation of two vectors one from lower arm and one from upper arm of lengths l1 and l2
            # let L be the length of the overall vector
            # we can calculate the angle between l1 , l2 and L
            L = math.sqrt(x ** 2 + z ** 2)
            L = np.round(L,13) # ???
            # not valid
            if L > (self._l1 + self._l2) or self._l1 > (self._l2 + L) or self._l2 > (self._l1 + L):  # in this case there is no solution
                return {"joint": np.array([None for i in range(len(xyz))]), "status": 2}

            # init status
            status = 0
            if L > (self._l1 + self._l2) - self._delta_e or self._l1 > (self._l2 + L) - self._delta_e: # in this case there is no solution
                status = 1

            teta_l1_L = math.acos((self._l1 ** 2 + L ** 2 - self._l2 ** 2) / (2 * self._l1 * L))  # l1 angle to L
            teta_L_x = math.atan2(z, x)  # L angle to x axis
            teta_1 = teta_l1_L + teta_L_x
            # note that the other solution would be to set teta_1 = teta_L_x - teta_l1_L. But for the dynamics of the robot the first solution works better.
            teta_l1_l2 = math.acos((self._l1 ** 2 + self._l2 ** 2 - L ** 2) / (2 * self._l1 * self._l2))  # l1 angle to l2
            teta_2 = teta_l1_l2 - math.pi
            teta_3 = alpha - teta_1 - teta_2
            teta_4 = beta
            teta_0 = math.degrees(teta_0)
            teta_1 = math.degrees(teta_1)
            teta_2 = math.degrees(teta_2)
            teta_3 = math.degrees(teta_3)
            teta_4 = math.degrees(teta_4)


            if len(xyz) == 6:
                joint = np.array([teta_0, teta_1, teta_2, teta_3, teta_4, xyz[5]])
            else:
                joint = np.array([teta_0, teta_1, teta_2, teta_3, teta_4])

            return {"joint": joint, "status": status}
