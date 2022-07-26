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

class Demo5(MujocoApp):
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
        self.viewer.cam.azimuth = -120
        self.viewer.cam.elevation = -40
        self.viewer.cam.distance = self.model.stat.extent
        self.move_interface = PSMoveInterface(multiprocess=False)
        self.move_states = self.move_interface.move_states
        self.grip_pos = dict([(move_name, 0.0) for move_name in MoveName])

        # self.robot_data_thread = threading.Thread(target=self.robot.start)
        # self.robot_data_thread.start()

    def print_forces(self):
        grip_pos = dict([(move_name, 0.0) for move_name in MoveName])
        max_pos = 0.9
        min_pos = 0.0
        while True:
            for move_name in MoveName:
                close = self.move_states[move_name].get('circle')
                open = self.move_states[move_name].get('triangle')
                if close:
                    grip_pos[move_name] -= 0.05
                if open:
                    grip_pos[move_name] += 0.05
                grip_pos[move_name] = min(max_pos, max(min_pos, grip_pos[move_name]))
                self.grip_pos[move_name] = grip_pos[move_name]
            time.sleep(0.1)

    def run_demo(self):
        # Start a timer for the demo
        targets = { 
            'ur5right' : Target(),
            'ur5left' : Target(),
            'base' : Target()
        }
        data_thread = threading.Thread(target=self.print_forces, args=())
        data_thread.start()
        ur5right = self.robot.get_device('ur5right')
        ur5left = self.robot.get_device('ur5left')
        
        while True:
            xyz_r = self.move_states[MoveName.RIGHT].get('pos')
            ang_r = self.move_states[MoveName.RIGHT].get('quat')
            xyz_l = self.move_states[MoveName.LEFT].get('pos')
            ang_l = self.move_states[MoveName.LEFT].get('quat')
            
            angle_mat_r = quat2mat(ang_r)
            tfmat1 = compose(xyz_r, angle_mat_r, [1,1,1])
            tfmat_r = compose([0.0,0,0], np.eye(3), [1,1,1])
            tfmat_r = np.matmul(tfmat1, tfmat_r)
            tfmat_r = np.matmul(tfmat_r, compose([0.0,0,0], euler2mat(np.pi/2,0,np.pi/2), [1,1,1]))
            
            angle_mat_l = quat2mat(ang_l)
            tfmat2 = compose(xyz_l, angle_mat_l, [1,1,1])
            tfmat_l = compose([0.05,0,0], np.eye(3), [1,1,1])
            tfmat_l = np.matmul(tfmat2, tfmat_l)
            tfmat_l = np.matmul(tfmat_l, compose([-0.15,0,0], euler2mat(0,0,-np.pi/2), [1,1,1]))
            
            r_xyz = tfmat_r[0:3,-1].flatten()
            l_xyz = tfmat_l[0:3,-1].flatten()
            r_ang = np.array(mat2euler(tfmat_r[:3, :3]))
            l_ang = np.array(mat2euler(tfmat_l[:3, :3]))
            
            if self.move_states[MoveName.RIGHT].get('trigger'):
                ur5right.ctrlr_dof_abg = [True, True, True]
                targets['ur5right'].xyz = r_xyz # [xpos_r, ypos_r, zpos_r]
                targets['ur5right'].abg = r_ang
            else:
                ur5right.ctrlr_dof_abg = [False, False, False]
                targets['ur5right'].xyz = ur5right.get_state('ee_xyz')

            if self.move_states[MoveName.LEFT].get('trigger'):
                ur5left.ctrlr_dof_abg = [True, True, True]
                targets['ur5left'].xyz = l_xyz # [xpos_r, ypos_r, zpos_r]
                targets['ur5left'].abg = l_ang
            else:
                ur5left.ctrlr_dof_abg = [False, False, False]
                targets['ur5left'].xyz = ur5left.get_state('ee_xyz')

            ctrlr_output = self.controller.generate(targets)
            for force_idx, force  in zip(*ctrlr_output):
                self.sim.data.ctrl[force_idx] = force
                self.sim.data.ctrl[7] = self.grip_pos[MoveName.RIGHT]
                self.sim.data.ctrl[14] = self.grip_pos[MoveName.LEFT]
            
            self.sim.data.set_mocap_pos('hand_right', r_xyz)
            # self.sim.data.set_mocap_quat('hand_ur5right', euler2quat(r_ang[0], r_ang[1], r_ang[2]))
            
            self.sim.data.set_mocap_pos('hand_left', l_xyz)
            # self.sim.data.set_mocap_quat('hand_left', euler2quat(l_ang[0], l_ang[1], l_ang[2]))
            
            ur5right_gripper_force = self.sim.data.sensordata[13]
            self.move_states[MoveName.RIGHT].set('rumble', ur5right_gripper_force)
            
            ur5left_gripper_force = self.sim.data.sensordata[16]
            self.move_states[MoveName.LEFT].set('rumble', ur5left_gripper_force)
            
            self.sim.step()
            self.viewer.render()
            
        # Join threads / Stop the simulator 
        # self.robot.stop()
        # self.robot_data_thread.join()
        # time_thread.join()

if __name__ == "__main__":
    ur5 = Demo5(robot_config_file="Demo_PSMove.yaml", scene_file="ps_move_scene.xml")
    ur5.run_demo()