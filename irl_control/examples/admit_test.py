import numpy as np
import mujoco_py as mjp
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
import threading
from typing import Dict, Tuple
from enum import Enum
import irl_control
from irl_control import OSC, MujocoApp, Device
from irl_control.utils import Target, ControllerConfig
import time
import yaml
import os
from transforms3d.euler import quat2euler, euler2quat, quat2mat, mat2euler, euler2mat
from transforms3d.affines import compose





class AdmitTest(MujocoApp):
    def __init__(self, robot_config_file: str = None, scene_file: str = None, active_arm: str = 'right'):
        # Initialize the Parent class with the config file
        super().__init__(robot_config_file, scene_file)
        # Specify the robot in the scene that we'd like to use
        self.robot = self.get_robot(robot_name="DualUR5")
        # Specify the controller configuations that should be used for
        # the corresponding devices
        admit_device_configs = [
            ('ur5right', self.get_controller_config('osc2')),
            ('ur5left', self.get_controller_config('osc2'))
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
    
    def gen_target(self) -> Tuple[np.ndarray, np.ndarray]:  #Generates the target position for both arms 
        right_wp = np.array([0.3, 0.46432, 0.5])

        left_wp = np.array([
            [-0.3, 0.46432, 0.5],
        ])
        return (right_wp, left_wp)

    def run(self):
        count = 0
        time_thread = threading.Thread(target=self.sleep_for, args=(50,))
        time_thread.start()
        threshold_ee = 0.1

        targets = { 
            'ur5right' : Target(), 
            'ur5left' : Target(),  
        }
        
        right_wps, left_wps = self.gen_target()
        ur5right = self.robot.sub_devices_dict['ur5right']
        ur5left = self.robot.sub_devices_dict['ur5left']
        
        right_wp_index = 0
        left_wp_index = 0
        
            
        while self.timer_running:
            count += 1
            targets['ur5right'].xyz = right_wps[right_wp_index]
            targets['ur5left'].xyz = left_wps[left_wp_index]
            targets['ur5left'].abg = np.array([0,-1*np.pi/2,0])
            
            self.sim.data.set_mocap_pos('target_red', right_wps[right_wp_index])
            self.sim.data.set_mocap_pos('target_blue', left_wps[left_wp_index])

            ctrlr_output = self.controller.generate(targets)
            #print('control',ctrlr_output)
            for force_idx, force  in zip(*ctrlr_output):
                self.sim.data.ctrl[force_idx] = force
            
            self.errors['ur5left'] = np.linalg.norm(ur5left.get_state('ee_xyz') - targets['ur5left'].xyz)
            
            self.sim.data.xfrc_applied[38] = [0,0,0,0,0,0]
            if count > 3000 and count < 5000:
                self.sim.data.xfrc_applied[38] = [20,0,0,0,0,0]  #32 #36"""
                
            if self.errors['ur5left']  < threshold_ee:
                if left_wp_index < left_wps.shape[0] - 1 and count > 5000:
                    left_wp_index += 1
                else:
                    left_wp_index = 0
                
            self.sim.step()
            self.viewer.render()
            functions.mj_inverse(self.model,self.sim.data)

        
        time_thread.join()
        self.robot_data_thread.join()
        glfw.destroy_window(self.viewer.window)

ur5 = AdmitTest(robot_config_file="DualUR5Scene.yaml", scene_file="main_dual_ur5.xml")
ur5.run()