from mujoco_py import GlfwContext
from mujoco_py.mjviewer import MjViewer
import numpy as np
from typing import Tuple
import threading
from irl_control import OSC, MujocoApp
from transforms3d.euler import quat2euler, euler2quat, quat2mat, mat2euler, euler2mat
from transforms3d.utils import normalized_vector
from transforms3d.affines import compose
# from InverseKinematics import IK
from irl_control.utils import Target
from irl_control.input_devices.space_mouse import SpaceMouse

class SpaceMouseDemo(MujocoApp):
    """
    This class implements the OSC and Dual UR5 robot
    """
    def __init__(self, robot_config_file : str =None, scene_file : str = None):
        # Initialize the Parent class with the config file
        super().__init__(robot_config_file, scene_file)
        # Specify the robot in the scene that we'd like to use
        self.viewer = MjViewer(self.sim)
        self.robot = self.get_robot(robot_name="DualUR5")
        # self.ik = IK(self.robot, self.sim)

        osc_device_configs = [
            ('base', self.get_controller_config('osc2')),
            ('ur5right', self.get_controller_config('osc2')),
            ('ur5left', self.get_controller_config('osc2'))
        ]

        # Get the configuration for the nullspace controller
        nullspace_config = self.get_controller_config('nullspace')
        self.controller = OSC(self.robot, self.sim, osc_device_configs, nullspace_config)
        self.robot_data_thread = threading.Thread(target=self.robot.start)
        self.robot_data_thread.start()
        self.viewer = MjViewer(self.sim) 
    
    def run_ik_demo(self, demo_duration: int):
        # Start a timer for the demo
        time_thread = threading.Thread(target=self.sleep_for, args=(demo_duration,))
        time_thread.start()
        
        targets = { 
            'ur5right' : Target(), 
            # 'ur5left' : Target(), 
            # 'base' : Target() 
        }
        ur5right = self.robot.get_device('ur5right')
        # ur5left = self.robot.get_device('ur5left')
        sm = SpaceMouse(origin=[0.0, 0.5, 0.5, 0.0, 0.0, 0.0])

        while self.timer_running:
            # Set the target values for the robot's devices
            x, y, z, roll, pitch, yaw = sm.update_state()

            angle = euler2quat(pitch, roll, yaw, axes='rxyz')
            angle_mat = quat2mat(angle)
            tfmat1 = compose([x,y,z], angle_mat, [1,1,1])
            
            tfmat_r = compose([0.2,0,0], np.eye(3), [1,1,1])
            tfmat_r = np.matmul(tfmat1, tfmat_r)
            tfmat_r = np.matmul(tfmat_r, compose([0,0,0], euler2mat(np.pi/2,0,0), [1,1,1]))
            
            tfmat_l = compose([-0.2,0,0], np.eye(3), [1,1,1])
            tfmat_l = np.matmul(tfmat1, tfmat_l)
            tfmat_l = np.matmul(tfmat_l, compose([0,0,0], euler2mat(np.pi/2,0,0), [1,1,1]))
            
            r_xyz = tfmat_r[0:3,-1].flatten()
            l_xyz = tfmat_l[0:3,-1].flatten()
            r_ang = np.array(mat2euler(tfmat_r[:3, :3]))
            l_ang = np.array(mat2euler(tfmat_l[:3, :3]))
            
            self.sim.data.set_mocap_pos('plate', [x,y,z])
            targets['ur5right'].xyz = r_xyz
            targets['ur5right'].abg = r_ang
            # targets['ur5left'].xyz = l_xyz
            # targets['ur5left'].abg = l_ang
            
            path = self.ik.generate(targets=targets)
            # self.sim.data.qpos[ur5left.ctrl_idxs] += path[ur5left.actuator_trnids]
            self.sim.data.qpos[ur5right.ctrl_idxs] += 3*path[ur5right.actuator_trnids]
            
            self.sim.data.set_mocap_quat('plate', angle)
            self.sim.data.set_mocap_pos('hand_ur5right', r_xyz)
            self.sim.data.set_mocap_quat('hand_ur5right', euler2quat(r_ang[0], r_ang[1], r_ang[2]))
            self.sim.data.set_mocap_pos('hand_ur5left', l_xyz)
            self.sim.data.set_mocap_quat('hand_ur5left', euler2quat(l_ang[0], l_ang[1], l_ang[2]))
            # Step simulator / Render scene
            self.sim.forward()
            # self.sim.step()
            self.viewer.render()
            
        # Join threads / Stop the simulator 
        self.robot.stop()
        self.robot_data_thread.join()
        time_thread.join()        

    def run_demo(self, demo_duration: int):
        # Start a timer for the demo
        time_thread = threading.Thread(target=self.sleep_for, args=(demo_duration,))
        time_thread.start()
        
        targets = { 
            'ur5right' : Target(),
            'ur5left' : Target(),
            'base' : Target()
        }
        ur5left = self.robot.get_device('ur5left')
        ur5right = self.robot.get_device('ur5right')
        sm = SpaceMouse(origin=[0.0, 0.5, 0.5, 0.0, 0.0, 0.0])
        while self.timer_running:
            # Set the target values for the robot's devices
            x, y, z, roll, pitch, yaw = sm.update_state()
            angle = euler2quat(pitch, roll, yaw, axes='rxyz') # pitch and roll are flipped from SM API
            angle_mat = quat2mat(angle)
            tfmat1 = compose([x,y,z], angle_mat, [1,1,1])
            
            tfmat_r = compose([0.05,0,0], np.eye(3), [1,1,1])
            tfmat_r = np.matmul(tfmat1, tfmat_r)
            tfmat_r = np.matmul(tfmat_r, compose([0.15,0,0], euler2mat(0,0,0), [1,1,1]))
            
            tfmat_l = compose([-0.05,0,0], np.eye(3), [1,1,1])
            tfmat_l = np.matmul(tfmat1, tfmat_l)
            tfmat_l = np.matmul(tfmat_l, compose([-0.15,0,0], euler2mat(0,0,np.pi), [1,1,1]))
            
            r_xyz = tfmat_r[0:3,-1].flatten()
            l_xyz = tfmat_l[0:3,-1].flatten()
            r_ang = np.array(mat2euler(tfmat_r[:3, :3]))
            l_ang = np.array(mat2euler(tfmat_l[:3, :3]))
            
            self.sim.data.set_mocap_pos('plate', [x,y,z])
            targets['ur5right'].xyz = r_xyz
            targets['ur5right'].abg = r_ang
            targets['ur5left'].xyz = l_xyz
            targets['ur5left'].abg = l_ang
            targets['base'].abg[2] = np.arctan2(y, x) - np.pi/2
            ctrlr_output = self.controller.generate(targets)
            for force_idx, force  in zip(*ctrlr_output):
                self.sim.data.ctrl[force_idx] = force
            
            self.sim.data.set_mocap_quat('plate', angle)
            self.sim.data.set_mocap_pos('hand_ur5right', r_xyz)
            self.sim.data.set_mocap_quat('hand_ur5right', euler2quat(r_ang[0], r_ang[1], r_ang[2]))
            self.sim.data.set_mocap_pos('hand_ur5left', l_xyz)
            self.sim.data.set_mocap_quat('hand_ur5left', euler2quat(l_ang[0], l_ang[1], l_ang[2]))
            # error_left = self.controller.calc_error(targets['ur5left'], ur5left)
            # error_right = self.controller.calc_error(targets['ur5right'], ur5right)
            # print("orien")
            # print(error_right[3:])
            # print("xyz")
            # print(error_right[:3])

            # Step simulator / Render scene
            self.sim.step()
            self.viewer.render()
            
        # Join threads / Stop the simulator 
        self.robot.stop()
        self.robot_data_thread.join()
        time_thread.join()

if __name__ == "__main__":
    ur5 = SpaceMouseDemo(robot_config_file="DualUR5Scene.yaml", scene_file="main_dual_ur5.xml")
    ur5.run_demo(400)