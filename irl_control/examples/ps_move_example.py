import sys
from transforms3d.euler import quat2euler, euler2quat, quat2mat, mat2euler, euler2mat
from transforms3d.affines import compose
import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim
from mujoco_py.mjviewer import MjViewer
import numpy as np
from typing import List, Tuple, Dict
import threading
from irl_control import MujocoApp, OSC
from irl_control.utils import Target
from irl_control.input_devices.ps_move import PSMoveInterface, MoveName
from irl_control.device import DeviceState

"""
This demo uses the PS Move controllers to control both robot arms
of the Dual UR5. In this demo, you can open the gripper by pressing
the Circle button and close the gripper by pressing the Triangle button.
The back trigger enables the robot, such that it will travel to the end
effector that is displayed by the mocap object attached to the 
corresponding move controller.
"""
class PSMoveExample(MujocoApp):
    """
    Implements the OSC and Dual UR5 Robot
    """
    def __init__(self, robot_config_file : str =None, scene_file : str = None):
        # Initialize the Parent class with the config file
        super().__init__(robot_config_file, scene_file)
        self.set_free_joint_qpos('free_joint_female', euler2quat(0, 0, np.pi/2))
        self.set_free_joint_qpos('free_joint_male', euler2quat(0, 0, np.pi/2))
        
        # Specify the robot in the scene that we'd like to use
        self.robot = self.get_robot(robot_name="DualUR5")
        osc_device_configs = [
            ('base', self.get_controller_config('osc2')),
            ('ur5right', self.get_controller_config('osc2')),
            ('ur5left', self.get_controller_config('osc2'))
        ]
        
        # Get the configuration for the nullspace controller
        nullspace_config = self.get_controller_config('nullspace')
        self.controller = OSC(self.robot, self.sim, osc_device_configs, nullspace_config)
        self.viewer = MjViewer(self.sim)
        
        # Set the scene camera distance and angle
        self.viewer.cam.azimuth = -120
        self.viewer.cam.elevation = -40
        self.viewer.cam.distance = self.model.stat.extent
        
        # Initialize the move interface (without multiprocessing)
        self.move_interface = PSMoveInterface(multiprocess=False)
        self.move_states = self.move_interface.move_states
        self.grip_pos = dict([(move_name, 0.0) for move_name in MoveName])
        
        # Start collecting the robot states
        # self.robot_data_thread = threading.Thread(target=self.robot.start)
        # self.robot_data_thread.start()

    def update_move_button_states(self, sleep_time = 0.1):
        """
        Get the move button states from the Move Interface
        at a rate specified by the sleep time so that the
        gripper (open and close) state can be updated
        """
        # Specify the min/max positions of the gripper (found experimentally)
        max_pos = 0.9
        min_pos = 0.0
        # Keep track of the grip positions locally
        grip_pos = dict([(move_name, 0.0) for move_name in MoveName])
        while True:
            for move_name in MoveName:
                close = self.move_states[move_name].get('circle')
                open = self.move_states[move_name].get('triangle')
                # Apply a fixed increment if the button is pressed
                if close:
                    grip_pos[move_name] -= 0.05
                if open:
                    grip_pos[move_name] += 0.05
                # Limit the grip positoin based on the min/max values specified above
                grip_pos[move_name] = min(max_pos, max(min_pos, grip_pos[move_name]))
                # Update the class grip positions with the local position
                self.grip_pos[move_name] = grip_pos[move_name]
            time.sleep(sleep_time)

    def run(self):
        # Start a timer for the demo
        targets: Dict[str, Target] = { 
            'ur5right' : Target(),
            'ur5left' : Target(),
            'base' : Target()
        }
        
        # Start a thread to read the move buttons from the interface
        move_button_thread = threading.Thread(target=self.update_move_button_states, args=())
        move_button_thread.start()
        
        # Get the ur5 devices from the robot
        ur5right = self.robot.get_device('ur5right')
        ur5left = self.robot.get_device('ur5left')
        
        while True:
            # Get the xyz/abg states from the move interface  for the left and right move controllers
            xyz_r = self.move_states[MoveName.RIGHT].get('pos')
            ang_r = self.move_states[MoveName.RIGHT].get('quat')
            xyz_l = self.move_states[MoveName.LEFT].get('pos')
            ang_l = self.move_states[MoveName.LEFT].get('quat')
            
            # Apply the given transformation to the end effector based on the orientation of the move, 
            # since we wish to control the end effector orientation using the move orientations
            angle_mat_r = quat2mat(ang_r)
            tfmat1 = compose(xyz_r, angle_mat_r, [1,1,1])
            tfmat_r = compose([0.0,0,0], np.eye(3), [1,1,1])
            tfmat_r = np.matmul(tfmat1, tfmat_r)
            # The following transformation (below) is used to orient the end effector in a way that faces
            # down the y axis for easier gripping of objects
            tfmat_r = np.matmul(tfmat_r, compose([0.0,0,0], euler2mat(np.pi/2,0,np.pi/2), [1,1,1]))
            
            # Similar to above, apply the transformations to the end effector 
            # of the left arm using the left move
            angle_mat_l = quat2mat(ang_l)
            tfmat2 = compose(xyz_l, angle_mat_l, [1,1,1])
            tfmat_l = compose([0.05,0,0], np.eye(3), [1,1,1])
            tfmat_l = np.matmul(tfmat2, tfmat_l)
            tfmat_l = np.matmul(tfmat_l, compose([-0.15,0,0], euler2mat(0,0,-np.pi/2), [1,1,1]))
            
            # Extract the final rotation from the rotation matrices for the
            #  left and right arms / moves
            r_xyz = tfmat_r[0:3,-1].flatten()
            l_xyz = tfmat_l[0:3,-1].flatten()
            r_ang = np.array(mat2euler(tfmat_r[:3, :3]))
            l_ang = np.array(mat2euler(tfmat_l[:3, :3]))
            
            # Set the targets for operational space control if the trigger is pressed
            if self.move_states[MoveName.RIGHT].get('trigger'):
                ur5right.ctrlr_dof_abg = [True, True, True]
                targets['ur5right'].set_xyz(r_xyz)
                targets['ur5right'].set_abg(r_ang)
            else:
                ur5right.ctrlr_dof_abg = [False, False, False]
                targets['ur5right'].set_xyz(ur5right.get_state(DeviceState.EE_XYZ))

            # Set the targets for operational space control if the trigger is pressed
            if self.move_states[MoveName.LEFT].get('trigger'):
                ur5left.ctrlr_dof_abg = [True, True, True]
                targets['ur5left'].set_xyz(l_xyz)
                targets['ur5left'].set_abg(l_ang)
            else:
                ur5left.ctrlr_dof_abg = [False, False, False]
                targets['ur5left'].set_xyz(ur5left.get_state(DeviceState.EE_XYZ))

            # Get the control from the operational space control based on the targets
            ctrlr_output = self.controller.generate(targets)
            # Send the forces to the robot (including gripper values)
            for force_idx, force  in zip(*ctrlr_output):
                self.sim.data.ctrl[force_idx] = force
                self.sim.data.ctrl[7] = self.grip_pos[MoveName.RIGHT]
                self.sim.data.ctrl[14] = self.grip_pos[MoveName.LEFT]
            
            # Set the position of the mocap corresponding the the right move
            self.sim.data.set_mocap_pos('hand_right', r_xyz)
            # self.sim.data.set_mocap_quat('hand_ur5right', euler2quat(r_ang[0], r_ang[1], r_ang[2]))
            
            # Set the position of the mocap corresponding the the left move
            self.sim.data.set_mocap_pos('hand_left', l_xyz)
            # self.sim.data.set_mocap_quat('hand_left', euler2quat(l_ang[0], l_ang[1], l_ang[2]))
            
            # Apply rumble to the right controller based on the sensed force from the end effector
            ur5right_gripper_force = self.sim.data.sensordata[13]
            self.move_states[MoveName.RIGHT].set('rumble', ur5right_gripper_force)
            
            # Apply rumble to the left controller based on the sensed force from the end effector
            ur5left_gripper_force = self.sim.data.sensordata[16]
            self.move_states[MoveName.LEFT].set('rumble', ur5left_gripper_force)
            
            # Step the simulator / Render scene
            self.sim.step()
            self.viewer.render()
            
        # Join threads / Stop the simulator 
        # self.robot.stop()
        # self.robot_data_thread.join()

# Main entrypoint
if __name__ == "__main__":
    # Initialize the PS Move Demo
    demo = PSMoveExample(robot_config_file="default_xyz_abg.yaml", scene_file="ps_move_scene.xml")
    # Run the PS Move Demo until user terminates program
    demo.run()