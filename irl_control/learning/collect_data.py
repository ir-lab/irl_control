from mujoco_py import GlfwContext
from mujoco_py.mjviewer import MjViewer
import numpy as np
from typing import Tuple
import threading
from irl_control import MujocoApp, OSC
from irl_control.utils import Target
from transforms3d.euler import quat2euler, euler2quat, quat2mat, mat2euler, euler2mat
from enum import Enum
import os
import irl_control
import yaml
from transforms3d.affines import compose
from typing import Dict
from control_base import ControlBase

DEFAULT_EE_ROT = np.deg2rad([0, -90, -90])
DEFAULT_EE_ORIENTATION = quat2euler(euler2quat(*DEFAULT_EE_ROT, 'sxyz'), 'rxyz')

class Action(Enum):
    """
    Action Enums are used to force the action sequence instructions (strings)
    to be converted into valid actions
    """
    WP = 0,
    GRIP = 1

class ActiveArm:
    def __init__(self):
        self.name = "ur5right"
        self.max_vel = [0]

"""
The purpose of this example is to test out the robot configuration
to see how well the gains perform on stabilizing the base and the
arm that does move rapidly. One of the arms in this demo will move
wildly to test out this stabilization capability.
"""
class CollectData(ControlBase):
    """
    Implements the OSC and Dual UR5 robot
    """
    def __init__(self, robot_config_file : str =None, scene_file : str = None):
        # Initialize the Parent class with the config file
        action_config_name = 'iros2022_task.yaml'
        self.action_config = self.get_action_config(action_config_name)
        super().__init__(self.action_config["device_config"], robot_config_file, scene_file)
           
    def run(self, demo_type: str, demo_duration: int):        
        action_sequence_name = 'iros2022_action_sequence'
        action_object_names = ['iros2022_action_objects']
        self.action_objects = self.action_config[action_object_names[0]]
        self.initialize_action_objects()
        self.run_sequence(self.action_config[action_sequence_name])
 
        # # counters/indexers used to keep track of waypoints
        # right_wp_idx = 0
        # left_wp_idx = 0
        # for step in range(200000):
        #     right_target = None
        #     with open("d.txt", "r") as fh:
        #         right_target = [float(v) for i, v in enumerate(fh.readline().split(","))]

        #     # Set the target values for the robot's devices
        #     # targets['ur5right'].setAllQuat(*right_target)
        #     self.targets['ur5right'].setAllQuat(*right_wps[right_wp_idx])

        #     # targets['ur5left'].setAllQuat(*right_target)
        #     self.targets['ur5left'].setAllQuat(*left_wps[left_wp_idx])
        #     self.targets['base'].abg[2] = np.deg2rad(-90)
            
        #     # Generate an OSC signal to steer robot toward the targets
        #     ctrlr_output = self.controller.generate(self.targets)
            
        #     # Generate an OSC signal to steer robot toward the targets
        #     for force_idx, force  in zip(*ctrlr_output):
        #         self.sim.data.ctrl[force_idx] = force

        #     state = self.robot.getState()
        #     # print("Step", step)

        #     # Step simulator / Render scene
        #     self.sim.step()
        #     self.viewer.render()     

        # for action_object_name in action_object_names:
        #     # Initialize the action objects and active arm
        #     action_config = self.get_action_config(action_config_name)
        #     self.action_objects = action_config[action_object_name]
        #     self.initialize_action_objects()
        #     # Run the sequence of all actions (main loop)
        #     self.run_sequence(action_config[action_sequence_name])         
        
        # # Join threads / Stop the simulator 
        # time_thread.join()
        self.robot.stop()
        self.robot_data_thread.join()

# Main entrypoint
if __name__ == "__main__":
    # Initialize the gain test demo
    demo = CollectData(robot_config_file="iros2022.yaml", scene_file="iros2022.xml")
    # Run the gain test
    demo_name1 = "gain_test"
    demo.run(demo_name1, 3000)