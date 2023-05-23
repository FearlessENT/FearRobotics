import time
import pybullet as p
import pybullet_data
import numpy as np
import Sender




class RobotArm:
    def __init__(self, urdf_path, end_effector_link_index, speed=1):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1) # change this for the sim

        plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        self.end_effector_link_index = end_effector_link_index
        self.speed = speed

        joint_positions = [0] * p.getNumJoints(self.robot_id)
        self.set_joint_positions(joint_positions)

    def set_joint_positions(self, joint_positions):
        num_joints = p.getNumJoints(self.robot_id)
        active_joints = [i for i in range(num_joints) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]

        for i, joint_index in enumerate(active_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=p.getJointInfo(self.robot_id, joint_index)[10],
                maxVelocity=self.speed * 2,  # Adjust maxVelocity based on the speed
            )


    def move_to_target(self, target_position):
        ik_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.end_effector_link_index,
            targetPosition=target_position,
        )

        # Return ik_joint_positions here without setting them to the joints
        return ik_joint_positions

        self.set_joint_positions(ik_joint_positions)

        position_change_threshold = 1e-4
        stable_iterations_needed = 10
        stable_iterations = 0

        previous_end_effector_position = None

        while stable_iterations < stable_iterations_needed:
            if previous_end_effector_position is not None:
                current_end_effector_position, _ = p.getLinkState(self.robot_id, self.end_effector_link_index)[:2]
                position_change = np.linalg.norm(np.array(previous_end_effector_position) - np.array(current_end_effector_position))
                if position_change < position_change_threshold:
                    stable_iterations += 1
                else:
                    stable_iterations = 0
                previous_end_effector_position = current_end_effector_position
            else:
                previous_end_effector_position, _ = p.getLinkState(self.robot_id, self.end_effector_link_index)[:2]

            p.stepSimulation()
            time.sleep(1./240.)



    def get_target_angles(self, x, y, z):
        print("Initial joint angles:", np.degrees(self.get_joint_angles()))

        target_position = [x, y, z]
        ik_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.end_effector_link_index,
            targetPosition=target_position,
        )

        print("Final joint angles (before actual movement):", np.degrees(ik_joint_positions))
        # print("IK JOINT POSISTIONS: ", ik_joint_positions)
        return list(ik_joint_positions)

    def goto(self, x, y, z):
        target_position = [x, y, z]
        ik_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.end_effector_link_index,
            targetPosition=target_position,
        )

        # Print joint angles in degrees
        print("Joint angles as is", ik_joint_positions)
        print("Joint angles (in degrees): ", np.degrees(ik_joint_positions))

        # Now set the final joint positions to the joints and let the simulation begin
        self.set_joint_positions(ik_joint_positions)

        position_change_threshold = 1e-4
        stable_iterations_needed = 10
        stable_iterations = 0

        previous_end_effector_position = None

        while stable_iterations < stable_iterations_needed:
            if previous_end_effector_position is not None:
                current_end_effector_position, _ = p.getLinkState(self.robot_id, self.end_effector_link_index)[:2]
                position_change = np.linalg.norm(np.array(previous_end_effector_position) - np.array(current_end_effector_position))
                if position_change < position_change_threshold:
                    stable_iterations += 1
                else:
                    stable_iterations = 0
                previous_end_effector_position = current_end_effector_position
            else:
                previous_end_effector_position, _ = p.getLinkState(self.robot_id, self.end_effector_link_index)[:2]

            p.stepSimulation()
            time.sleep(1./240.)




    def get_motion_time2(self, x, y, z):
        target_position = [x, y, z]

        # initial_joint_positions = self.get_joint_angles()
        initial_end_effector_position, _ = p.getLinkState(self.robot_id, self.end_effector_link_index)[:2]

        # Calculate the Euclidean distance between initial and target positions
        distance = np.linalg.norm(np.array(initial_end_effector_position) - np.array(target_position))

        # Calculate the motion time based on the speed of the arm
        speed = self.speed 
        motion_time = distance / speed

        return motion_time

    def get_motion_time(self, x, y, z):
        target_position = [x, y, z]
        ik_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.end_effector_link_index,
            targetPosition=target_position,
        )

        # set the final joint positions to the joints without starting the simulation
        self.set_joint_positions(ik_joint_positions)

        position_change_threshold = 1e-4
        stable_iterations_needed = 10
        stable_iterations = 0

        previous_end_effector_position = None

        step_count = 0

        while stable_iterations < stable_iterations_needed:
            if previous_end_effector_position is not None:
                current_end_effector_position, _ = p.getLinkState(self.robot_id, self.end_effector_link_index)[:2]
                position_change = np.linalg.norm(np.array(previous_end_effector_position) - np.array(current_end_effector_position))
                if position_change < position_change_threshold:
                    stable_iterations += 1
                else:
                    stable_iterations = 0
                previous_end_effector_position = current_end_effector_position
            else:
                previous_end_effector_position, _ = p.getLinkState(self.robot_id, self.end_effector_link_index)[:2]

            step_count += 1

        # Multiply the step count by the time each step represents to get the total time
        total_time = step_count / 240.0

        return total_time


    def set_speed(self, speed):
        if speed > 0:
            self.speed = speed
        else:
            raise ValueError("Speed must be a positive value.")


    def get_joint_angles(self):
        joint_angles = [p.getJointState(self.robot_id, i)[0] for i in range(p.getNumJoints(self.robot_id))]
        # print("FUNCTION JOINT ANGLES", joint_angles)
        return joint_angles[1:]
    
    
        

    def get_angle_difference(self, old_angles, new_angles):
        return [new - old for old, new in zip(old_angles, new_angles)]
    



    def goto_coords(self, x, y , z):


        motion_time = self.get_motion_time2(x, y, z)
        before_angles = self.get_joint_angles()
        after_angles = self.get_target_angles(x, y, z)
        difference_angles = self.get_angle_difference(before_angles, after_angles)

        Sender.send_angles(difference_angles, motion_time)
        
        self.goto(x, y, z)