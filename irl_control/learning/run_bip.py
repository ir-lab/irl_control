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

    def send_bip(self, forces, gripper_force:float=None, update_errors:bool=True):      
        # Apply forces to the main robot
        for force_idx, force  in zip(*forces):
            self.sim.data.ctrl[force_idx] = force
        # Apply gripper force to the active arm
        if gripper_force:
            for idx in [7,14]:
                self.sim.data.ctrl[idx] = gripper_force
                
        # Update the errors for every device
        if update_errors:
            for device_name in self.targets.keys():
                # Calculate the euclidean norm between target and current to obtain error
                self.errors[device_name] = np.linalg.norm(
                    self.controller.calc_error(self.targets[device_name], self.robot.get_device(device_name)))
           
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
            state = self.get_clean_state(self.robot.getState())
            phase, prediction = intprim.update_stream(state)
            if phase is not None:
                print("Phase at step {}: {:.2f}".format(step, phase))
            
                # Set Targets:
                targets['ur5left'].setAllQuat(*np.take(prediction, [26,27,28, 32,33,34,35]))
                targets['ur5right'].setAllQuat(*np.take(prediction, [29,30,31, 36,37,38,39]))
                targets['base'].abg[2] = prediction[0]
            
            # Generate an OSC signal to steer robot toward the targets
            ctrlr_output = self.controller.generate(targets)
            self.send_bip(ctrlr_output, gripper_force=self.gripper_force, update_errors=True)
            
            # Step simulator / Render scene
            self.sim.step()
            self.viewer.render()   
            step += 1        
        
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