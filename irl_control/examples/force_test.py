import numpy as np
import mujoco_py as mjp
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
import threading
from typing import Dict, Tuple
import irl_control
from irl_control import OSC, MujocoApp, Device
from irl_control.utils import Target, ControllerConfig
import csv

class ForceTest(MujocoApp):
    def __init__(self, robot_config_file: str = None, scene_file: str = None, active_arm: str = 'right'):
        # Initialize the Parent class with the config file
        super().__init__(robot_config_file, scene_file)
        # Specify the robot in the scene that we'd like to use
        self.robot = self.get_robot(robot_name="DualUR5")
        # Specify the controller configuations that should be used for
        # the corresponding devices
        admit_device_configs = [
            ('ur5right', self.get_controller_config('osc1')),
            ('ur5left', self.get_controller_config('osc1'))
        ]
        # Get the configuration for the nullspace controller
        nullspace_config = self.get_controller_config('nullspace')
        self.controller = OSC(self.robot, self.sim, admit_device_configs, nullspace_config,admittance = True)
        # Start collecting device states from simulator
        # NOTE: This is necessary when you are using OSC, as it assumes
        #       that the robot.start() thread is running.
        self.robot_data_thread = threading.Thread(target=self.robot.start)
        self.robot_data_thread.start()
        # Keep track of device target errors
        self.errors = dict()
        self.errors['ur5right'] = 0
        self.viewer = MjViewer(self.sim)
    
    def gen_target(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the target position for both arms 
        """  
        right_wp = np.array([0.3, 0.46432, 0.36243])

        left_wp = np.array([
            [-0.3, 0.46432, 0.5],
            [-0.3, 0.5, 0.5],
            [-0.3, 0.55, 0.5],
            [-0.3, 0.575, 0.5],
            [-0.3, 0.6, 0.5],
            [-0.3, 0.625, 0.5],
            [-0.3, 0.65, 0.5],
            [-0.3, 0.675, 0.5],
            [-0.3, 0.7, 0.5],
            [-0.3, 0.75, 0.5],
        ])

        return (right_wp, left_wp)

    def run(self):
        """
        This is the main function that gets called. Uses the 
        Admittance Controller  on the DualUR5 to apply a force on the external 
        object.
        """
        #start the thread timmer
        time_thread = threading.Thread(target=self.sleep_for, args=(200,))
        time_thread.start()
        threshold_ee = 0.01
        #Define targets for both arms
        targets = { 
            'ur5right' : Target(), 
            'ur5left' : Target(),  
        }
        #Get the targets for both arms
        right_wps, left_wps = self.gen_target()
        ur5right = self.robot.sub_devices_dict['ur5right']
        ur5left = self.robot.sub_devices_dict['ur5left']
        #Define target indices
        right_wp_index = 0
        left_wp_index = 0
        #Initialize force variables
        x = 0
        #Define field names tp write to CSV
        fieldnames = ["x", "force_x", "force_y", "force_z"]
        #Write the fieldnames to CSV
        with open('data.csv', 'w') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            
        while self.timer_running:
            #set the targets position and orientation of both arms
            targets['ur5right'].xyz = right_wps[right_wp_index]
            targets['ur5left'].xyz = left_wps[left_wp_index]
            targets['ur5left'].abg = np.array([0,0,-1*np.pi/2])
            #set the mocap position to target position
            self.sim.data.set_mocap_pos('target_red', right_wps[right_wp_index])
            self.sim.data.set_mocap_pos('target_blue', left_wps[left_wp_index])
            #Genetrate the admittance control output
            ctrlr_output = self.controller.generate(targets)
            for force_idx, force  in zip(*ctrlr_output):
                self.sim.data.ctrl[force_idx] = force
            #Measure the errors 
            self.errors['ur5left'] = np.linalg.norm(ur5left.get_state('ee_xyz') - targets['ur5left'].xyz)
            #Move to next target if error is less then threshold
            if self.errors['ur5left']  < threshold_ee:
                if left_wp_index < left_wps.shape[0] - 1 :
                    left_wp_index += 1
                else:
                    left_wp_index = 0  
            self.sim.step()
            self.viewer.render()
            functions.mj_inverse(self.model,self.sim.data)
            #Measure force on left end-effector
            force = ur5left.get_force()
            x += 1
            #write force values to CSV
            with open('data.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                info = {
                    "x": x,
                    "force_x": force[0],
                    "force_y": force[1],
                    "force_z": force[2],
                    }
                csv_writer.writerow(info)
        time_thread.join()
        self.robot_data_thread.join()
        glfw.destroy_window(self.viewer.window)

if __name__ == "__main__":
    ur5 = ForceTest(robot_config_file="default_xyz_abg.yaml", scene_file="force_test_scene.xml")
    ur5.run()
