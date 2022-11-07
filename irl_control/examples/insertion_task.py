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
from transforms3d.quaternions import qmult
from transforms3d.affines import compose
from irl_control.device import DeviceState

# Define the default orientations of the end effectors
DEFAULT_EE_ROT = np.deg2rad([0, -90, -90])
DEFAULT_EE_ORIENTATION = quat2euler(euler2quat(*DEFAULT_EE_ROT, 'sxyz'), 'rxyz')
DEFAULT_EE_QUAT = euler2quat(*DEFAULT_EE_ROT)

class Action(Enum):
    """
    Action Enums are used to force the action sequence instructions (strings)
    to be converted into valid actions
    """
    WP = 0,
    GRIP = 1


"""
In this example, the robot performs a variety of insertion tasks,
using the male and female adapters, which are generated at fixed
or random locations (depending on the demo being run).
"""
class InsertionTask(MujocoApp):
    """
    This class implements the OSC and Dual UR5 robot.
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

        # self.robot_data_thread = threading.Thread(target=self.robot.start)
        # self.robot_data_thread.start()
        
        # Keep track of device target errors
        self.errors: Dict[str, float] = dict()
        self.viewer = MjViewer(self.sim)
        # Set the camera distance/angle
        self.viewer.cam.azimuth = -90
        self.viewer.cam.elevation = -40
        self.viewer.cam.distance = self.model.stat.extent
        # Get the functions for the actions in the sequence
        self.action_map = self.get_action_map()
        # Get the default parameters for the actions
        self.DEFAULT_PARAMS: Dict[Action, Dict] = dict([(action, self.get_default_action_ctrl_params(action)) for action in Action])
        
        self.targets = { 
            self.active_arm.name : Target(), 
            self.passive_arm.name : Target(), 
        }

    def get_default_action_ctrl_params(self, action):
        """
        Get the default gain, velocity, and gripper values for the insertion task.
        These can be changed, but experimentally these values have been found to work
        well with the insertion action sequence.
        """
        # Waypoint action defaults
        if action == Action.WP:
            param_dict = {
                'kp' : 6,
                'max_error' : 0.0018,
                'gripper_force' : 0.0,
                'min_speed_xyz' : 0.1,
                'max_speed_xyz' : 3.0
            }
        # Grip action defaults
        elif action == Action.GRIP:
            param_dict = {
                'gripper_force' : -0.08,
                'gripper_duation' : 1.0
            }
        return param_dict

    def get_action_map(self):
        """
        Return the functions associated with the action defined in the action sequence.
        """
        action_map: Dict[Action, function] = {
            Action.WP : self.go_to_waypoint,
            Action.GRIP : self.grip
        }
        return action_map
    
    def set_active_arm(self, active_arm):
        """
        Set the dual ur5 arm to be either the left or right arm,
        depending on which one you'd like to control and which one
        you'd like to be passive.
        """
        assert (active_arm == 'right' or active_arm == 'left'), "Demo only supports Dual UR5 configuration"
        # Set the ur5 arm based on the 'right' or 'left' input
        if active_arm == 'right':
            self.active_arm = self.ur5right
            self.passive_arm = self.ur5left
        elif active_arm == 'left':
            self.active_arm = self.ur5left
            self.passive_arm = self.ur5right
    
    def get_action_config(self, config_file: str):
        """
        Return the dictionary formatted data structure of the 
        configuration file passed into the function.
        config_file should be the name of the yaml file in the
        action_sequence_configs directory.
        """
        main_dir = os.path.dirname(irl_control.__file__)
        # Load the action config from the action_sequence_configs directory
        action_obj_config_path = os.path.join(main_dir, "action_sequence_configs", config_file)
        with open(action_obj_config_path, 'r') as file:
            action_config = yaml.safe_load(file)
        return action_config

    def send_forces(self, forces, gripper_force:float=None, update_errors:str=None, render:bool=True):
        """
        This function sends forces to the robot, using the values supplied.
        Optionally, you can render the scene and update errors of the devices,
        which are stored as class member variables.
        """
        # Determine gripper index based on active arm
        if self.active_arm.name == 'ur5right':
            gripper_idx = 7
        elif self.active_arm.name == 'ur5left':
            gripper_idx = 14
        
        # Apply forces to the main robot
        for force_idx, force  in zip(*forces):
            self.sim.data.ctrl[force_idx] = force
        # Apply gripper force to the active arm
        if gripper_force:
            self.sim.data.ctrl[gripper_idx] = gripper_force
        
        # Render the sim (optionally)
        if render:
            self.sim.step()
            self.viewer.render()
        
        # Update the errors for every device
        if update_errors:
            if type(update_errors) == list:
                for device_name in update_errors:
                    # Calculate the euclidean norm between target and current to obtain error
                    self.errors[device_name] = np.linalg.norm(
                        self.controller.calc_error(self.targets[device_name], self.robot.get_device(device_name)))
            elif type(update_errors) == str:
                device_name = update_errors
                # Calculate the euclidean norm between target and current to obtain error
                self.errors[device_name] = np.linalg.norm(
                        self.controller.calc_error(self.targets[device_name], self.robot.get_device(device_name)))
    
    def string2action(self, string: str):
        """
        Return the Enum associated with the action token.
        """
        if string == 'WP':
            return Action.WP
        elif string == 'GRIP':
            return Action.GRIP

    def grip(self, params):
        """
        This is an action which is responsbile for solely operating the gripper.
        This method assumes that self.targets is set for the arms beforehand, such that 
        the arms will remain in the current position (since no target is applied here).
        """
        assert params['action'] == 'GRIP'
        self.update_action_ctrl_params(params, Action.GRIP)
        # Sleep for the duration specified in the action sequence
        time_thread = threading.Thread(target=self.sleep_for, args=(params['gripper_duration'],))
        time_thread.start()
        # Apply gripper forces for duration specified
        while self.timer_running:
            ctrlr_output = self.controller.generate(self.targets)
            self.send_forces(ctrlr_output, gripper_force=params['gripper_force'], update_errors=self.active_arm.name)
    
    def set_waypoint_targets(self, params):
        """
        Set the targets for the robot devices (arms) based on the values 
        specified in the action sequence.
        """
        # Set targets for passive arm
        self.targets[self.passive_arm.name].set_xyz(self.passive_arm.get_state(DeviceState.EE_XYZ))
        self.targets[self.passive_arm.name].set_quat(DEFAULT_EE_QUAT)
        
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
                # Use the starting position (if specified)
                if params['target_xyz'] == 'start_pos':
                    target = self.start_pos
                else:
                    # Parse the action parameter for the target xyz location
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
            self.targets[self.active_arm.name].set_xyz(target)
        else:
            print("No Target Specified for Waypoint!")
            raise KeyError('target_xyz')
        
        # Set the target orientations for the arm
        if 'target_abg' in params.keys():
            if isinstance(params['target_abg'], str):
                # Apply the necessary yaw offet to the end effector
                target_obj = self.action_objects[params['target_abg']]
                # Get the quaternion for the target object
                obj_quat = self.sim.data.get_joint_qpos(target_obj['joint_name'])[-4:]
                # Add the ee offset to the default ee orientation
                grip_eul = DEFAULT_EE_ROT + [0, 0, np.deg2rad(target_obj['grip_yaw'])]
                # grip_quat = DEFAULT_EE_QUAT * euler2quat([0, 0, np.deg2rad(target_obj['grip_yaw'])])
                tfmat_obj = compose([0,0,0], quat2mat(obj_quat), [1,1,1])
                tfmat_grip = compose([0,0,0], euler2mat(*grip_eul), [1,1,1])
                tfmat = np.matmul(tfmat_obj, tfmat_grip)
                # Extract the end effector yaw from the final rotation matrix output
                target_abg = np.array(mat2euler(tfmat[:3, :3]))
            elif isinstance(params['target_abg'], list):
                target_abg = np.deg2rad(params['target_abg'])
            else:
                print("Invalid type for target_abg!")
                raise ValueError
            self.targets[self.active_arm.name].set_abg(target_abg)
        else:
            self.targets[self.active_arm.name].set_quat(DEFAULT_EE_QUAT)
    
    def update_action_ctrl_params(self, params, action: Action):
        """
        Apply the default values to the parameter, if it is not specified in the action.
        """
        for key, default_val in self.DEFAULT_PARAMS[action].items():
            params[key] = params[key] if key in params.keys() else default_val

    def go_to_waypoint(self, params):
        """
        This is the main action used in the insertion demo.
        Applies forces to the robot and gripper (as opposed to the gripper only, in the grip action)
        using the parameters specified by the action.
        """
        assert params['action'] == 'WP'
        # Se the targets (class member variables) 
        self.set_waypoint_targets(params)
        # Apply default parameter values to those that are unspecified
        self.update_action_ctrl_params(params, Action.WP)
        # Iterate the controller until the desired level of error is achieved
        self.errors[self.active_arm.name] = np.inf
        while self.errors[self.active_arm.name] > params['max_error']:
            # Limit the max velocity of the robot according to the given params
            self.active_arm.max_vel[0] = max(params['min_speed_xyz'], 
                min(params['max_speed_xyz'], params['kp']*self.errors[self.active_arm.name]))
            ctrlr_output = self.controller.generate(self.targets)
            self.send_forces(ctrlr_output, gripper_force=params['gripper_force'], update_errors=self.active_arm.name)
    
    def initialize_action_objects(self):
        """
        Apply the initial positions and orientations specified by the
        objects inside of the action_objects (action sequence file).
        """
        for obj_name in self.action_objects:
            obj = self.action_objects[obj_name]
            # Convert the degrees in the yaml file to radians
            target_quat = euler2quat(*(np.deg2rad(obj['initial_pos_abg']))) if 'initial_pos_abg' in obj.keys() else None
            # Parse the initial xyz position
            target_pos = obj['initial_pos_xyz'] if 'initial_pos_xyz' in obj.keys() else None
            # Set the position and quaternion of the simulator object
            self.set_free_joint_qpos(obj['joint_name'], quat=target_quat, pos=target_pos)

    def run_sequence(self, action_sequence):
        self.start_pos = np.copy(self.active_arm.get_state(DeviceState.EE_XYZ))
        for action_entry in action_sequence:
            action = self.string2action(action_entry['action'])
            action_func = self.action_map[action]
            action_func(action_entry)
    
    def clear_action_objects(self, y_pos):
        """
        Clear the objects from the scene by placing them behind the robot
        onto the line specified by the y_pos passed into the funciton
        """
        # Move the male object behind the robot (Effectively hides/clears the object)
        male_obj = self.action_objects['male_object']
        male_obj['initial_pos_xyz'][0] = -0.3
        male_obj['initial_pos_xyz'][1] = y_pos 
        male_obj['initial_pos_abg'] = [0, 0, 0]
        # Set the position and quaternion of the simulator object
        self.set_free_joint_qpos(male_obj['joint_name'], quat=euler2quat(*male_obj['initial_pos_abg']), pos=male_obj['initial_pos_xyz'])
        
        # Move the female object behind the robot, next to the male object (Effectively hides/clears the object)
        female_obj = self.action_objects['female_object']
        female_obj['initial_pos_xyz'][0] = 0.3
        female_obj['initial_pos_xyz'][1] = y_pos
        female_obj['initial_pos_abg'] = [0, 0, 0]
        # Set the position and quaternion of the simulator object
        self.set_free_joint_qpos(female_obj['joint_name'], quat=euler2quat(*female_obj['initial_pos_abg']), pos=female_obj['initial_pos_xyz'])
    
    def initialize_action_objects_random(self, arm_name):
        """
        Randomly generate the positions of the objects in the scene,
        such that the male/female do not fall onto each other
        and that the objects are within the coordinates given below
        """
        # Use these coordinates to randomly place the male object
        mx = np.random.uniform(low=0.4, high=0.6)
        my = np.random.uniform(low=0.5, high=0.7)
        # Use these coordinates to randomly place the female object
        fx = np.random.uniform(low=0.0, high=0.3)
        fy = np.random.uniform(low=0.5, high=0.7)

        # Apply the random orientation and position to the male object
        male_obj = self.action_objects['male_object']
        yaw_male = int(np.random.uniform(-20, 20))
        male_obj['initial_pos_abg'] = [0, 0, yaw_male]
        male_obj['initial_pos_xyz'][0] = mx if arm_name == 'right' else -1*mx
        male_obj['initial_pos_xyz'][1] = my
        # Set the position and quaternion of the simulator object
        self.set_free_joint_qpos(male_obj['joint_name'], quat=euler2quat(*male_obj['initial_pos_abg']), pos=male_obj['initial_pos_xyz'])
        
        # Apply the random orientation and position to the female object
        female_obj = self.action_objects['female_object']
        yaw_female = int(np.random.uniform(-20, 20))
        female_obj['initial_pos_abg'] = [0, 0, yaw_female]
        female_obj['initial_pos_xyz'][0] = fx if arm_name == 'right' else -1*fx
        female_obj['initial_pos_xyz'][1] = fy
        # Set the position and quaternion of the simulator object
        self.set_free_joint_qpos(female_obj['joint_name'], quat=euler2quat(*female_obj['initial_pos_abg']), pos=female_obj['initial_pos_xyz'])

    def run(self, randomize = False):
        """
        Runs the insertion demo by either using the randomly generated positions
        or the positions/orientations specified by the action objects
        in the action sequence file 
        """
        # Specify the desired action sequence file, 
        # the action sequence within this file, and the action objects within this file
        action_config_name = 'insertion_task.yaml'
        action_sequence_name = 'insertion_action_sequence'
        action_object_names = ['nist_action_objects', 'grommet_action_objects']
        arms = ['right', 'left']
        y_pos_arr = np.linspace(-0.5, -0.5*len(action_object_names))

        # First, clear the objects from the scene by placing them behind the robot
        for idx, action_object_name in enumerate(action_object_names):
            action_config = self.get_action_config(action_config_name)
            self.action_objects = action_config[action_object_name]
            self.clear_action_objects(y_pos_arr[idx])
        
        # If we want to run the randomized location/orientation demo
        if randomize:
            # obj_idx, arm_idx  = np.random.binomial(1, 0.5, 2)
            arm_idxs = [0,1,1,0]
            obj_idxs = [0,1,0,1]
            for arm_idx, obj_idx in zip(arm_idxs, obj_idxs):
                # Initialize the action objects and active arm
                action_config = self.get_action_config(action_config_name)
                self.action_objects = action_config[action_object_names[obj_idx]]
                self.set_active_arm(arms[arm_idx])
                self.initialize_action_objects_random(arms[arm_idx])
                # Run the sequence of all actions (main loop)
                self.run_sequence(action_config[action_sequence_name])
                self.clear_action_objects(y_pos_arr[obj_idx])
        # If we want to run the demo with object locations specified by the configuration file
        else:
            for arm_name, action_object_name in zip(arms, action_object_names):
                # Initialize the action objects and active arm
                action_config = self.get_action_config(action_config_name)
                self.action_objects = action_config[action_object_name]
                self.set_active_arm(arm_name)
                self.initialize_action_objects()
                # Run the sequence of all actions (main loop)
                self.run_sequence(action_config[action_sequence_name])            
        
        # Stop collecting the robot states and join threads
        # self.robot.stop()
        # self.robot_data_thread.join()

# Main entrypoint
if __name__ == "__main__":
    # Initialize the insertion Task
    demo = InsertionTask(robot_config_file="default_xyz_abg.yaml", scene_file="insertion_task_scene.xml", active_arm='right')
    # Run the non-randomized demo (user-specified object locations/orientations)
    demo.run(randomize=False)
    # Run the randomized demo (randomly generated object locations/orientations)
    demo.run(randomize=True)