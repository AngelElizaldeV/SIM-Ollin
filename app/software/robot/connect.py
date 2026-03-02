from dorna import Dorna

def get_robot(config_path="C:/Users/g477e/Desktop/Config2.yaml"):

    robot = Dorna(config_path)
    robot.connect()
    return robot