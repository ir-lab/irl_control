import numpy as np
from mujoco_py import MjViewer, functions
import threading
from typing import Dict, Tuple
from irl_control import OSC, MujocoApp
from irl_control.utils import Target

class AdmitTest(MujocoApp):
    """
    This class implements the Admittance Controller on Dual UR5 robot
    """
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
        """
        Generates the target position for both arms 
        """    
        right_wp = np.array([0.3, 0.46432, 0.5])
        left_wp = np.array([-0.3, 0.46432, 0.5])
        return (right_wp, left_wp)

    def run(self):
        """
        This is the main function that gets called. Uses the 
        Admittance Controller  to control the DualUR5
        and adjust to the external forces acting on the end effector.
        """
        #start the count and timer
        count = 0
        time_thread = threading.Thread(target=self.sleep_for, args=(50,))
        time_thread.start()
        threshold_ee = 0.1
        #Define targets for both arms
        targets = { 
            'ur5right' : Target(), 
            'ur5left' : Target(),  
        }
        #Get the targets for both arms
        right_wps, left_wps = self.gen_target()
        ur5right = self.robot.sub_devices_dict['ur5right']
        ur5left = self.robot.sub_devices_dict['ur5left']
        while self.timer_running:
            #set the targets position and orientation of both arms
            count += 1
            targets['ur5right'].xyz = right_wps
            targets['ur5left'].xyz = left_wps
            targets['ur5left'].abg = np.array([0,-1*np.pi/2,0])
            #set the mocap position to target position
            self.sim.data.set_mocap_pos('target_red', right_wps)
            self.sim.data.set_mocap_pos('target_blue', left_wps)
            #Genetrate the admittance control output
            ctrlr_output = self.controller.generate(targets)
            for force_idx, force  in zip(*ctrlr_output):
                self.sim.data.ctrl[force_idx] = force
            #Apply external force on left end effector
            self.sim.data.xfrc_applied[38] = [0,0,0,0,0,0]
            if count > 3000 and count < 5000:
                self.sim.data.xfrc_applied[38] = [0,0,20,0,0,0]   
            self.sim.step()
            self.viewer.render()
            functions.mj_inverse(self.model,self.sim.data)
        time_thread.join()
        self.robot_data_thread.join()
        glfw.destroy_window(self.viewer.window)
        
if __name__ == "__main__":
    ur5 = AdmitTest(robot_config_file="default_xyz_abg.yaml", scene_file="admit_test_scene.xml")
    ur5.run()