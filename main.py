from RobotArm import RobotArm
# from RobotArmMonitor import RobotArmMonitor
# from RobotArmSender import RobotArmSender
import threading
import time
import Sender
import math



# Initialize the robot arm
robot_arm = RobotArm("RobotArm.urdf", end_effector_link_index=5, speed=1)



# Change the robot arm speed
robot_arm.set_speed(0.4)
robot_arm.goto(0.5, -0.3, 0.5)

while True:

    print("iteration")
    
    time.sleep(3)
    
    robot_arm.goto_coords(0.5, 0.3, 0.3)

    time.sleep(3)

    robot_arm.goto_coords(0.4, 0.3, 0.3)

    print("")
   
