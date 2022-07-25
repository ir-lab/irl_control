from mujoco_py.mjviewer import MjViewer
import numpy as np
import threading
from typing import Dict
from enum import Enum
import irl_control
from irl_control import OSC, MujocoApp, Device
from irl_control.utils import Target, ControllerConfig
import time
import yaml
import os
from transforms3d.euler import quat2euler, euler2quat, quat2mat, mat2euler, euler2mat
from transforms3d.affines import compose

DEFAULT_EE_ROT = np.deg2rad([0, -90, -90])
DEFAULT_EE_ORIENTATION = quat2euler(euler2quat(*DEFAULT_EE_ROT, 'sxyz'), 'rxyz')

class Action(Enum):
    WP = 0,
    GRIP = 1


class InsertionTask(MujocoApp):
    """
    This class implements the OSC and Dual UR5 robot
    """
    def __init__(self, robot_config_file: str = None, scene_file: str = None, active_arm: str = 'right'):
        # Initialize the Parent class with the config file
        super().__init__(robot_config_file, scene_file)
        # Specify the robot in the scene that we'd like to use
        self.robot = self.get_robot(robot_name="DualUR5")
        
        self.ur5right = self.robot.get_device('ur5right')
        self.ur5left = self.robot.get_device('ur5left')
        self.set_active_arm(active_arm)

        # Specify the controller configuations that should be used for
        # the corresponding devices
        osc_device_configs = [
            ('base', self.get_controller_config('osc0')),
            ('ur5right', self.get_controller_config('osc2')),
            ('ur5left', self.get_controller_config('osc2'))
        ]

        # Get the configuration for the nullspace controller
        nullspace_config = self.get_controller_config('nullspace')
        self.controller = OSC(self.robot, self.sim, osc_device_configs, nullspace_config)

        # Start collecting device states from simulator
        # NOTE: This is necessary when you are using OSC, as it assumes
        #       that the robot.start() thread is running.
        self.robot_data_thread = threading.Thread(target=self.robot.start)
        self.robot_data_thread.start()
        
        # Keep track of device target errors
        self.errors: Dict[str, float] = dict()
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -90
        self.viewer.cam.elevation = -40
        self.viewer.cam.distance = self.model.stat.extent
        self.action_map = self.get_action_map()
        self.DEFAULT_PARAMS: Dict[Action, Dict] = dict([(action, self.get_default_action_ctrl_params(action)) for action in Action])
        
        self.targets = { 
            self.active_arm.name : Target(), 
            self.passive_arm.name : Target(), 
        }

    def get_default_action_ctrl_params(self, action):
        if action == Action.WP:
            param_dict = {
                'kp' : 6,
                'max_error' : 0.0018,
                'gripper_force' : 0.0,
                'min_speed_xyz' : 0.1,
                'max_speed_xyz' : 3.0
            }
        elif action == Action.GRIP:
            param_dict = {
                'gripper_force' : -0.08,
                'gripper_duation' : 1.0
            }
        return param_dict

    def get_action_map(self):
        action_map: Dict[Action, function] = {
            Action.WP : self.go_to_waypoint,
            Action.GRIP : self.grip
        }
        return action_map
    
    def set_active_arm(self, active_arm):
        assert (active_arm == 'right' or active_arm == 'left'), "Demo only supports Dual UR5 configuration"
        if active_arm == 'right':
            self.active_arm = self.ur5right
            self.passive_arm = self.ur5left
        elif active_arm == 'left':
            self.active_arm = self.ur5left
            self.passive_arm = self.ur5right
    
    def get_action_config(self, config_file):
        main_dir = os.path.dirname(irl_control.__file__)
        action_obj_config_path = os.path.join(main_dir, "action_sequence_configs", config_file)
        with open(action_obj_config_path, 'r') as file:
            action_config = yaml.safe_load(file)
        return action_config

    def send_forces(self, forces, gripper_force:float=None, update_errors:str=None, render:bool=True):
        if self.active_arm.name == 'ur5right':
            gripper_idx = 7
        elif self.active_arm.name == 'ur5left':
            gripper_idx = 14
        
        for force_idx, force  in zip(*forces):
            self.sim.data.ctrl[force_idx] = force
        if gripper_force:
            self.sim.data.ctrl[gripper_idx] = gripper_force
        
        if render:
            self.sim.step()
            self.viewer.render()
        
        if update_errors:
            if type(update_errors) == list:
                for device_name in update_errors:
                    self.errors[device_name] = np.linalg.norm(
                        self.controller.calc_error(self.targets[device_name], self.robot.get_device(device_name)))
            elif type(update_errors) == str:
                device_name = update_errors
                self.errors[device_name] = np.linalg.norm(
                        self.controller.calc_error(self.targets[device_name], self.robot.get_device(device_name)))
    
    def string2action(self, string: str):
        if string == 'WP':
            return Action.WP
        elif string == 'GRIP':
            return Action.GRIP

    def grip(self, params):
        assert params['action'] == 'GRIP'
        self.update_action_ctrl_params(params, Action.GRIP)
        time_thread = threading.Thread(target=self.sleep_for, args=(params['gripper_duration'],))
        time_thread.start()
        while self.timer_running:
            ctrlr_output = self.controller.generate(self.targets)
            self.send_forces(ctrlr_output, gripper_force=params['gripper_force'], update_errors=self.active_arm.name)
    
    def set_waypoint_targets(self, params):
        # Set targets for passive arm
        self.targets[self.passive_arm.name].xyz = self.passive_arm.get_state('ee_xyz')
        self.targets[self.passive_arm.name].abg = DEFAULT_EE_ORIENTATION
        
        # Set targets for active arm
        if 'target_xyz' in params.keys():
            # NOTE: offset can be a string (instead of a list). This string must be the name of an
            # attribute of the action object specified by 'target_xyz' (and the attribute value must be 
            # a list with 3 entries)
            offset = params['offset'] if 'offset' in params.keys() else [0.0, 0.0, 0.0]
            
            # NOTE: target_xyz can be a string (instead of a list); there are 2 possibilites
            # 1) 'starting_pos': Must be set in python before running the WP Action
            # 2) '<action_object_name>': a string which must an action object, where the coordinates
            #    of this object are retrieved from the simulator
            if isinstance(params['target_xyz'], str):
                if params['target_xyz'] == 'start_pos':
                    target = self.start_pos
                else:
                    target_obj = self.action_objects[params['target_xyz']]
                    if isinstance(offset, str):
                        offset = target_obj[offset]
                    target_obj = self.action_objects[params['target_xyz']]['joint_name']
                    obj_pos = self.sim.data.get_joint_qpos(target_obj)[:3]
                    target = obj_pos + offset
            elif isinstance(params['target_xyz'], list):
                target = params['target_xyz'] + offset
            else:
                print("Invalid type for target_xyz!")
                raise ValueError
            self.targets[self.active_arm.name].xyz = target
        else:
            print("No Target Specified for Waypoint!")
            raise KeyError('target_xyz')
        
        if 'target_abg' in params.keys():
            if isinstance(params['target_abg'], str):
                target_obj = self.action_objects[params['target_abg']]
                obj_quat = self.sim.data.get_joint_qpos(target_obj['joint_name'])[-4:]
                grip_eul = DEFAULT_EE_ROT + [0,0,np.deg2rad(target_obj['grip_yaw'])]
                tfmat_obj = compose([0,0,0], quat2mat(obj_quat), [1,1,1])
                tfmat_grip = compose([0,0,0], euler2mat(*grip_eul, 'sxyz'), [1,1,1])
                tfmat = np.matmul(tfmat_obj, tfmat_grip)
                target_abg = np.array(mat2euler(tfmat[:3, :3], 'rxyz'))
            elif isinstance(params['target_xyz'], list):
                target_abg = np.deg2rad(params['target_abg'])
            else:
                print("Invalid type for target_xyz!")
                raise ValueError
            self.targets[self.active_arm.name].abg = target_abg
        else:
            self.targets[self.active_arm.name].abg = DEFAULT_EE_ORIENTATION
    
    def update_action_ctrl_params(self, params, action: Action):
        for key, default_val in self.DEFAULT_PARAMS[action].items():
            params[key] = params[key] if key in params.keys() else default_val

    def go_to_waypoint(self, params):
        assert params['action'] == 'WP'
        self.set_waypoint_targets(params)
        self.update_action_ctrl_params(params, Action.WP)
        self.errors[self.active_arm.name] = np.inf
        while self.errors[self.active_arm.name] > params['max_error']:
            self.active_arm.max_vel[0] = max(params['min_speed_xyz'], 
                min(params['max_speed_xyz'], params['kp']*self.errors[self.active_arm.name]))
            ctrlr_output = self.controller.generate(self.targets)
            self.send_forces(ctrlr_output, gripper_force=params['gripper_force'], update_errors=self.active_arm.name)
    
    def initialize_action_objects(self):
        for obj_name in self.action_objects:
            obj = self.action_objects[obj_name]
            target_quat = euler2quat(*(np.deg2rad(obj['initial_pos_abg']))) if 'initial_pos_abg' in obj.keys() else None
            target_pos = obj['initial_pos_xyz'] if 'initial_pos_xyz' in obj.keys() else None
            self.set_free_joint_qpos(obj['joint_name'], quat=target_quat, pos=target_pos)

    def run_sequence(self, action_sequence):
        self.start_pos = np.copy(self.active_arm.get_state('ee_xyz'))
        for action_entry in action_sequence:
            action = self.string2action(action_entry['action'])
            action_func = self.action_map[action]
            action_func(action_entry)
    
    def clear_action_objects(self, y_pos):
        male_obj = self.action_objects['male_object']
        male_obj['initial_pos_xyz'][0] = -0.3
        male_obj['initial_pos_xyz'][1] = y_pos 
        male_obj['initial_pos_abg'] = [0, 0, 0]
        self.set_free_joint_qpos(male_obj['joint_name'], quat=euler2quat(*male_obj['initial_pos_abg']), pos=male_obj['initial_pos_xyz'])
        
        female_obj = self.action_objects['female_object']
        female_obj['initial_pos_xyz'][0] = 0.3
        female_obj['initial_pos_xyz'][1] = y_pos
        female_obj['initial_pos_abg'] = [0, 0, 0]
        self.set_free_joint_qpos(female_obj['joint_name'], quat=euler2quat(*female_obj['initial_pos_abg']), pos=female_obj['initial_pos_xyz'])
    
    def initialize_action_objects_random(self, arm_name):
        mx = np.random.uniform(low=0.4, high=0.6)
        my = np.random.uniform(low=0.5, high=0.7)
        fx = np.random.uniform(low=0.0, high=0.3)
        fy = np.random.uniform(low=0.5, high=0.7)

        male_obj = self.action_objects['male_object']
        yaw_male = int(np.random.uniform(-20, 20))
        male_obj['initial_pos_abg'] = [0, 0, yaw_male]
        male_obj['initial_pos_xyz'][0] = mx if arm_name == 'right' else -1*mx
        male_obj['initial_pos_xyz'][1] = my
        self.set_free_joint_qpos(male_obj['joint_name'], quat=euler2quat(*male_obj['initial_pos_abg']), pos=male_obj['initial_pos_xyz'])
        
        female_obj = self.action_objects['female_object']
        yaw_female = int(np.random.uniform(-20, 20))
        female_obj['initial_pos_abg'] = [0, 0, yaw_female]
        female_obj['initial_pos_xyz'][0] = fx if arm_name == 'right' else -1*fx
        female_obj['initial_pos_xyz'][1] = fy
        self.set_free_joint_qpos(female_obj['joint_name'], quat=euler2quat(*female_obj['initial_pos_abg']), pos=female_obj['initial_pos_xyz'])

    def run(self, randomize = False):
        action_config_name = 'insertion_task.yaml'
        action_sequence_name = 'insertion_action_sequence'
        action_object_names = ['nist_action_objects', 'grommet_action_objects']
        arms = ['right', 'left']
        y_pos_arr = np.linspace(-0.5, -0.5*len(action_object_names))

        for idx, action_object_name in enumerate(action_object_names):
            action_config = self.get_action_config(action_config_name)
            self.action_objects = action_config[action_object_name]
            self.clear_action_objects(y_pos_arr[idx])
        if randomize:
            # obj_idx, arm_idx  = np.random.binomial(1, 0.5, 2)
            arm_idxs = [0,1,1,0]
            obj_idxs = [0,1,0,1]
            for arm_idx, obj_idx in zip(arm_idxs, obj_idxs):
                action_config = self.get_action_config(action_config_name)
                self.action_objects = action_config[action_object_names[obj_idx]]
                self.set_active_arm(arms[arm_idx])
                self.initialize_action_objects_random(arms[arm_idx])
                self.run_sequence(action_config[action_sequence_name])
                self.clear_action_objects(y_pos_arr[obj_idx])
        else:
            for arm_name, action_object_name in zip(arms, action_object_names):
                action_config = self.get_action_config(action_config_name)
                self.action_objects = action_config[action_object_name]
                self.set_active_arm(arm_name)
                self.initialize_action_objects()
                self.run_sequence(action_config[action_sequence_name])            
        
        self.robot.stop()
        self.robot_data_thread.join()

if __name__ == "__main__":
    demo = InsertionTask(robot_config_file="DualUR5Scene.yaml", scene_file="place_object.xml", active_arm='right')
    #demo.run(randomize=False)
    demo.run(randomize=True)