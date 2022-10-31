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
from BIP.bip_bimanual import IntprimStream
from transforms3d.euler import quat2euler, euler2quat

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
        intprim = IntprimStream()

        action_object_names = ['iros2022_action_objects']
        self.action_objects = self.action_config[action_object_names[0]]
        self.initialize_action_objects()
        self.run_sequence(self.action_config['iros2022_pickup_sequence'])

        targets = {}
        targets['base'] = Target()
        targets['ur5left'] = Target()
        targets['ur5right'] = Target()

        self.sim.step()
        step = 0
        while not intprim.is_done():
            state = self.get_clean_state(self.robot.get_device_states())
            phase, prediction = intprim.update_stream(state)
            if phase is not None:
                print("Phase at step {}: {:.2f}".format(step, phase))
            
                # Set Targets:
                targets['ur5left'].setAllQuat(*np.take(prediction, [25,26,27, 31,32,33,34]))
                targets['ur5right'].setAllQuat(*np.take(prediction, [28,29,30, 35,36,37,38]))
                targets['base'].abg[2] = prediction[0]
            
            # Generate an OSC signal to steer robot toward the targets
            ctrlr_output = self.controller.generate(targets)
            self.send_forces(ctrlr_output, gripper_force=self.gripper_force, update_errors=True)
            step += 1        

            # if step > 2000 and step < 4000:
            #     for id in  [self.sim.model.body_name2id(v) for v in ["ur_EE_ur5left", "ur_EE_ur5right"]]:
            #         self.sim.data.xfrc_applied[id] = [20,0,0,0,0,0] 
        
        self.targets = targets
        
        self.run_sequence(self.action_config['iros2022_release_sequence'])

        # Close Simulator
        self.robot.stop()
        self.robot_data_thread.join()

# Main entrypoint
if __name__ == "__main__":
    # Initialize the gain test demo
    demo = CollectData(robot_config_file="iros2022.yaml", scene_file="iros2022.xml")
    # Run the gain test
    demo_name1 = "gain_test"
    demo.run(demo_name1, 3000)