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

DEFAULT_EE_ROT = np.deg2rad([0, -90, -90])
DEFAULT_EE_ORIENTATION = quat2euler(euler2quat(*DEFAULT_EE_ROT, 'sxyz'), 'rxyz')

class Action(Enum):
    """
    Action Enums are used to force the action sequence instructions (strings)
    to be converted into valid actions
    """
    WP = 0,
    GRIP = 1

"""
The purpose of this example is to test out the robot configuration
to see how well the gains perform on stabilizing the base and the
arm that does move rapidly. One of the arms in this demo will move
wildly to test out this stabilization capability.
"""
class ControlBase(MujocoApp):
    """
    Implements the OSC and Dual UR5 robot
    """
    def __init__(self, device_config, robot_config_file : str =None, scene_file : str = None):
        # Initialize the Parent class with the config file
        super().__init__(robot_config_file, scene_file)
        # Specify the robot in the scene that we'd like to use
        self.robot = self.get_robot(robot_name="DualUR5")
        
        # Specify the controller configuations that should be used for
        # the corresponding devices

        osc_device_configs = []
        self.targets = {}
        self.errors = {}

        for device, controller in zip(device_config["devices"], device_config["controllers"]):
            osc_device_configs.append((device, self.get_controller_config(controller)))
            self.targets[device] = Target()
            self.errors[device] = np.inf

        # Get the configuration for the nullspace controller
        nullspace_config = self.get_controller_config('nullspace')
        self.controller = OSC(self.robot, self.sim, osc_device_configs, nullspace_config)

        # Start collecting device states from simulator
        # NOTE: This is necessary when you are using OSC, as it assumes
        #       that the robot.start() thread is running.
        self.robot_data_thread = threading.Thread(target=self.robot.start)
        self.robot_data_thread.start()
        
        # Keep track of device target errors
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -30
        self.viewer.cam.distance = self.model.stat.extent*1.5
        self.action_map = self.get_action_map()
        self.DEFAULT_PARAMS: Dict[Action, Dict] = dict([(action, self.get_default_action_ctrl_params(action)) for action in Action])
        self.gripper_force = -0.1

    def grip(self, params):
        """
        This is an action which is responsbile for solely operating the gripper.
        This method assumes that self.targets is set for the arms beforehand, such that 
        the arms will remain in the current position (since no target is applied here).
        """
        assert params['action'] == 'GRIP'
        self.update_action_ctrl_params(params, Action.GRIP)
        self.gripper_force = params['gripper_force']
        # Sleep for the duration specified in the action sequence
        time_thread = threading.Thread(target=self.sleep_for, args=(params['gripper_duration'],))
        time_thread.start()
        # Apply gripper forces for duration specified
        while self.timer_running:
            ctrlr_output = self.controller.generate(self.targets)
            self.send_forces(ctrlr_output, gripper_force=self.gripper_force, update_errors=True)

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

    def is_done(self, max_error):
        for _, err in self.errors.items():
            if err > max_error:
                return False
        return True

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
        while not self.is_done(params["max_error"]):
            # Limit the max velocity of the robot according to the given params
            # self.active_arm.max_vel[0] = max(params['min_speed_xyz'], 
            #     min(params['max_speed_xyz'], params['kp']*self.errors["ur5right"]))
            ctrlr_output = self.controller.generate(self.targets)
            self.send_forces(ctrlr_output, gripper_force=self.gripper_force, update_errors=True)

    def send_forces(self, forces, gripper_force:float=None, update_errors:bool=True, render:bool=True):
        """
        This function sends forces to the robot, using the values supplied.
        Optionally, you can render the scene and update errors of the devices,
        which are stored as class member variables.
        """
        # Determine gripper index based on active arm
        # if self.active_arm.name == 'ur5right':
        #     gripper_idx = 7
        # elif self.active_arm.name == 'ur5left':
        #     gripper_idx = 14
        
        # Apply forces to the main robot
        for force_idx, force  in zip(*forces):
            self.sim.data.ctrl[force_idx] = force
        # Apply gripper force to the active arm
        if gripper_force:
            for idx in [7,14]:
                self.sim.data.ctrl[idx] = gripper_force
        
        # Render the sim (optionally)
        if render:
            self.sim.step()
            self.viewer.render()
        
        # Update the errors for every device
        if update_errors:
            for device_name in self.targets.keys():
                # Calculate the euclidean norm between target and current to obtain error
                self.errors[device_name] = np.linalg.norm(
                    self.controller.calc_error(self.targets[device_name], self.robot.get_device(device_name)))

    def update_action_ctrl_params(self, params, action: Action):
        """
        Apply the default values to the parameter, if it is not specified in the action.
        """
        for key, default_val in self.DEFAULT_PARAMS[action].items():
            params[key] = params[key] if key in params.keys() else default_val
    
    def apply_keys(self, d, keys):
        temp = d
        for key in keys.split("."):
            temp = temp[key]
        return temp

    def set_waypoint_targets(self, params):
        """
        Set the targets for the robot devices (arms) based on the values 
        specified in the action sequence.
        """
        # # Set targets for passive arm
        # self.targets[self.passive_arm.name].xyz = self.passive_arm.get_state('ee_xyz')
        # self.targets[self.passive_arm.name].abg = DEFAULT_EE_ORIENTATION
        
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
                # Parse the action parameter for the target xyz location
                target_obj = self.apply_keys(self.action_objects, params['target_xyz'])
                for d, t in zip(self.targets.keys(), target_obj):
                    self.targets[d].xyz = t
            elif isinstance(params['target_xyz'], list):
                # target = params['target_xyz'] + offset
                for d, t in zip(self.targets.keys(), params["target_xyz"]):
                    self.targets[d].xyz = t
            else:
                print("Invalid type for target_xyz!")
                raise ValueError
            # TODO: Make this a "set all deal"
            # self.targets['ur5right'].xyz = target
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
                grip_eul = DEFAULT_EE_ROT + [0,0,np.deg2rad(target_obj['grip_yaw'])]
                tfmat_obj = compose([0,0,0], quat2mat(obj_quat), [1,1,1])
                tfmat_grip = compose([0,0,0], euler2mat(*grip_eul, 'sxyz'), [1,1,1])
                tfmat = np.matmul(tfmat_obj, tfmat_grip)
                # Extract the end effector yaw from the final rotation matrix output
                target_abg = np.array(mat2euler(tfmat[:3, :3], 'rxyz'))
            elif isinstance(params['target_xyz'], list):
                target_abg = np.deg2rad(params['target_abg'])
            else:
                print("Invalid type for target_xyz!")
                raise ValueError
            #TODO: General set
            self.targets["ur5right"].abg = target_abg
        else:
            self.targets["ur5right"].abg = DEFAULT_EE_ORIENTATION
        
        # Set the target orientations for the arm
        if 'target_quat' in params.keys():
            for d, t in zip(self.targets.keys(), params["target_quat"]):
                self.targets[d].setQuat(*t)

    def get_action_map(self):
        """
        Return the functions associated with the action defined in the action sequence.
        """
        action_map: Dict[Action, function] = {
            Action.WP : self.go_to_waypoint,
            Action.GRIP : self.grip
        }
        return action_map

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
    
    def initialize_action_objects(self):
        """
        Apply the initial positions and orientations specified by the
        objects inside of the action_objects (action sequence file).
        """
        for obj_name in self.action_objects:
            obj = self.action_objects[obj_name]
            # Convert the degrees in the yaml file to radians
            target_quat = obj['initial_pos_quat'] if 'initial_pos_quat' in obj.keys() else None
            # Parse the initial xyz position
            target_pos = obj['initial_pos_xyz'] if 'initial_pos_xyz' in obj.keys() else None
            # Set the position and quaternion of the simulator object
            print("Setting target for", obj_name, "to", target_pos, "and", target_quat)
            self.set_free_joint_qpos(obj['joint_name'], quat=target_quat, pos=target_pos)

    def string2action(self, string: str):
        """
        Return the Enum associated with the action token.
        """
        if string == 'WP':
            return Action.WP
        elif string == 'GRIP':
            return Action.GRIP

    def run_sequence(self, action_sequence):
        # self.start_pos = np.copy(self.active_arm.get_state('ee_xyz'))
        for action_entry in action_sequence:
            if "name" in action_entry.keys():
                print("Starting Action", action_entry["name"])
            for key in self.errors.keys():
                self.errors[key] = np.inf
            action = self.string2action(action_entry['action'])
            action_func = self.action_map[action]
            action_func(action_entry)