import numpy as np
from irl_control.utils import Target
from transforms3d.euler import quat2euler, euler2quat
from enum import Enum
from typing import Dict
from dual_insertion import DualInsertion
from bip_bimanual import IntprimStream


class RunBIP(DualInsertion):
    """
    Implements the OSC and Dual UR5 robot
    """
    def __init__(self, robot_config_file : str =None, scene_file : str = None):
        # Initialize the Parent class with the config file
        action_config_name = 'iros2022_task.yaml'
        self.action_config = self.get_action_config(action_config_name)
        super().__init__(self.action_config["device_config"], robot_config_file, scene_file)
           
    def run(self):
        intprim = IntprimStream()

        action_object_names = ['iros2022_action_objects']
        self.action_objects = self.action_config[action_object_names[0]]
        self.initialize_action_objects()
        self.run_sequence(self.action_config['iros2022_pickup_sequence'])

        targets: Dict[str, Target] = {
            'base' : Target(),
            'ur5left' : Target(),
            'ur5right' : Target()
        }

        self.sim.step()
        step = 0
        while not intprim.is_done():
            state = self.get_clean_state(self.robot.get_device_states())
            phase, prediction = intprim.update_stream(state)
            prediction = np.asarray(prediction)
            if phase is not None:
                print("Phase at step {}: {:.2f}".format(step, phase))
                # Set Targets:
                targets['ur5left'].set_all_quat(prediction[[25,26,27]], prediction[[31,32,33,34]])
                targets['ur5right'].set_all_quat(prediction[[28,29,30]], prediction[[35,36,37,38]])
                targets['base'].set_abg([0, 0, prediction[0]])
            
            # Generate an OSC signal to steer robot toward the targets
            ctrlr_output = self.controller.generate(targets)
            self.send_forces(ctrlr_output, gripper_force=self.gripper_force)
            self.sim.step()
            self.viewer.render()
            step += 1        
        
        self.targets = targets
        
        self.run_sequence(self.action_config['iros2022_release_sequence'])

# Main entrypoint
if __name__ == "__main__":
    # Initialize the gain test demo
    demo = RunBIP(robot_config_file="iros2022.yaml", scene_file="iros2022.xml")
    demo.run()