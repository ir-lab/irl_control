from mujoco_py import GlfwContext
from mujoco_py.mjviewer import MjViewer
import numpy as np
from typing import Tuple
import threading
from irl_control import MujocoApp, OSC
from irl_control.device import DeviceState
from irl_control.utils import Target
from transforms3d.euler import quat2euler, euler2quat, quat2mat, mat2euler, euler2mat
from enum import Enum
import os
import irl_control
import yaml
from transforms3d.affines import compose
from typing import Dict
import calendar
import time
from hashids import Hashids

DEFAULT_EE_ROT = np.deg2rad([0, -90, -90])
DEFAULT_EE_ORIENTATION = quat2euler(euler2quat(*DEFAULT_EE_ROT, 'sxyz'), 'rxyz')

class Action(Enum):
    """
    Action Enums are used to force the action sequence instructions (strings)
    to be converted into valid actions
    """
    WP = 0,
    GRIP = 1
    INTERP = 2

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
        self.controller = OSC(self.robot, self.sim, osc_device_configs, nullspace_config, admittance = True)

        # Start collecting device states from simulator
        # NOTE: This is necessary when you are using OSC, as it assumes
        #       that the robot.start() thread is running.
        # self.robot_data_thread = threading.Thread(target=self.robot.start)
        # self.robot_data_thread.start()

        # Keep track of device target errors
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -30
        self.viewer.cam.distance = self.model.stat.extent*1.5
        self.action_map = self.get_action_map()
        self.DEFAULT_PARAMS: Dict[Action, Dict] = dict([(action, self.get_default_action_ctrl_params(action)) for action in Action])
        self.gripper_force = 0.00
        self.recording = []
        self.record = False
        self.hashids = Hashids()
    
    def set_record(self, val):
        self.record = val
        if val == False and len(self.recording) > 0:
            fn = os.path.join("./data/BIP/", self.hashids.encode(calendar.timegm(time.gmtime())) + ".csv")
            np.savetxt(fn, self.recording, delimiter=', ')
            self.recording = []
            print("Recording saved as:", fn)

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
                'max_speed_xyz' : 3.0,
                'noise': [0.0, 0.0]
            }
        # Grip action defaults
        elif action == Action.GRIP:
            param_dict = {
                'gripper_force' : -0.08,
                'gripper_duation' : 1.0
            }
        # Interpolation action defaults
        elif action == Action.INTERP:
            param_dict = {
                'method' : "linear",
                'steps' : 2
            }
        return param_dict

    def is_done(self, max_error, step):
        # Based on steps
        if step < 25:
            return False
        # Based on velocity
        vel = []
        for device_name in self.errors.keys():
            vel += self.robot.get_device(device_name).get_state(DeviceState.DQ).tolist()
        vel = np.asarray(vel)
        if np.all(np.isclose(np.zeros_like(vel), vel, rtol=max_error, atol=max_error)):
            return True
        return False

        # Based on error:
        # done = True
        # for _, err in self.errors.items():
        #     if err > max_error:
        #         print(err)
        #         done = False
        # return done

    def go_to_waypoint(self, params):
        """
        This is the main action used in the insertion demo.
        Applies forces to the robot and gripper (as opposed to the gripper only, in the grip action)
        using the parameters specified by the action.
        """
        assert params['action'] == 'WP'
        # Apply default parameter values to those that are unspecified
        self.update_action_ctrl_params(params, Action.WP)
        # Se the targets (class member variables) 
        self.set_waypoint_targets(params)
        # Iterate the controller until the desired level of error is achieved
        step = 0
        while not self.is_done(params["max_error"], step):
            step += 1
            ctrlr_output = self.controller.generate(self.targets)
            self.send_forces(ctrlr_output, gripper_force=self.gripper_force, update_errors=True)
    
    def fix_rot(self, rot):
        rot = np.asarray(rot)
        if rot[0] < 0:
            rot *= -1.0
        return rot.tolist()

    def send_forces(self, forces, gripper_force:float=None, update_errors:bool=True, render:bool=True):
        """
        This function sends forces to the robot, using the values supplied.
        Optionally, you can render the scene and update errors of the devices,
        which are stored as class member variables.
        """
        
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
        
        if self.record:
            feedback = self.robot.get_device_states()
            self.recording.append(self.get_clean_state(feedback))
        
        # Update the errors for every device
        if update_errors:
            for device_name in self.targets.keys():
                # Calculate the euclidean norm between target and current to obtain error
                self.errors[device_name] = np.linalg.norm(
                    self.controller.calc_error(self.targets[device_name], self.robot.get_device(device_name)))

    def get_clean_state(self, state):
        #TODO: This needs a smarter selector
        return np.concatenate(( 
                state['base'][DeviceState.Q_ACTUATED], state['ur5left'][DeviceState.Q_ACTUATED], state['ur5right'][DeviceState.Q_ACTUATED], # 0-1, 2-7, 8-13
                state['ur5left'][DeviceState.FORCE], state['ur5left'][DeviceState.TORQUE], state['ur5right'][DeviceState.FORCE], state['ur5right'][DeviceState.TORQUE], # 14-16, 17-19, 20-22, 23-25
                state['ur5left'][DeviceState.EE_XYZ], state['ur5right'][DeviceState.EE_XYZ], # 26-28, 29-31
                self.fix_rot(state['ur5left'][DeviceState.EE_QUAT]), self.fix_rot(state['ur5right'][DeviceState.EE_QUAT]) # 32-35, 36-39
        ))

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
                    self.targets[d].xyz = (np.asarray(t) + np.random.normal(params['noise'][0], params['noise'][1], size=np.asarray(t).shape)).tolist()
            elif isinstance(params['target_xyz'], list):
                # target = params['target_xyz'] + offset
                for d, t in zip(self.targets.keys(), params["target_xyz"]):
                    self.targets[d].xyz = (np.asarray(t) + np.random.normal(params['noise'][0], params['noise'][1], size=np.asarray(t).shape)).tolist()
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
            Action.GRIP : self.grip,
            Action.INTERP : self.interpolate_waypoint,
        }
        return action_map
    
    def interpolate_dof(self, data):
        for device in range(data.shape[1]):
            for dof in range(data.shape[2]):
                data[:, device, dof] = np.linspace(data[0,device,dof],data[-1,device,dof],data.shape[0])
        return data
    
    def interpolate_waypoint(self, params):
        # Figure out where we currently are
        current_pos = []
        current_rot = []
        for device_name in self.errors.keys():
            current_pos.append(self.robot.get_device(device_name).get_state(DeviceState.EE_XYZ))
            rot = np.asarray(self.robot.get_device(device_name).get_state(DeviceState.EE_QUAT))
            if rot[0] < 0:
                rot *= -1 # Make sure the w component is always positive
            current_rot.append(rot)
        current_pos = np.asarray(current_pos)
        current_rot = np.asarray(current_rot)

        # Interpolate the thing...
        steps = params["steps"]
        pos_wps = np.zeros((steps, current_pos.shape[0], current_pos.shape[1]))
        rot_wps = np.zeros((steps, current_rot.shape[0], current_rot.shape[1]))
        pos_wps[0,:,:] = np.asarray(current_pos)
        pos_wps[-1,:,:] = np.asarray(params["target_xyz"])
        rot_wps[0,:,:] = np.asarray(current_rot)
        rot_wps[-1,:,:] = np.asarray(params["target_quat"])

        pos_wps = self.interpolate_dof(pos_wps)
        rot_wps = self.interpolate_dof(rot_wps)

        # Create parameters for waypoint
        for i in range(steps):
            wp_params = params.copy()
            wp_params["action"] = "WP"
            wp_params["target_xyz"] = pos_wps[i].tolist()
            wp_params["target_quat"] = rot_wps[i].tolist()
            wp_params["name"] = params["name"] + "_Step_" + str(i)
            print("Interpolated Waypoint", i)
            for key in self.errors.keys():
                self.errors[key] = np.inf
            self.go_to_waypoint(wp_params)


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
        elif string == "INTERP":
            return Action.INTERP

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