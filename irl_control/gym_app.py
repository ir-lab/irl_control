import numpy as np
from mujoco_py import load_model_from_path, MjSim
from mujoco_py.mjviewer import MjViewer
import mujoco_py
import irl_control
from irl_control import Device, Robot
from typing import Dict
import time
import os
import yaml
import gym
from irl_control.utils import Target
from irl_control import OSC
from irl_control.device import DeviceState
import calendar 
from hashids import Hashids
from proto_tools import proto_logger
from gail.policyopt import Trajectory, TrajBatch
from pathlib import Path
from enum import Enum

IRL_DATA_DIR = Path(os.environ.get('HOME') + '/irl_control_container/data')
EXPERT_TRAJ_DIR = IRL_DATA_DIR / 'expert_trajectories'

GRIP_IDX_RIGHT = 7
GRIP_IDX_LEFT = 14

class ObservationType(Enum):
    JNT = "robot_joints"
    OBJ = "action_objects"

class ActionType(Enum):
    POS = "position"
    DPOS = "velocity"

class GymBimanual(gym.Env):
    def __init__(self,
                 robot_config_file,
                 scene_file,
                 osc_device_pairs=None,
                 data_collect_hz=100, 
                 render_scene=False,
                 record=False,
                 manual_base_ctrl=False,
                 obs: ObservationType=ObservationType.OBJ,
                 act: ActionType=ActionType.DPOS):
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        obs_dim = 26 if self.OBS_TYPE == ObservationType.OBJ else 25
        self.action_space = gym.spaces.Box(low=-2.0*np.ones(14, dtype=np.float32), high=2.0*np.ones(14, dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=-15*np.ones(obs_dim, dtype=np.float32), high=15*np.ones(obs_dim, dtype=np.float32))

        main_dir = os.path.dirname(irl_control.__file__)
        scene_file_path = os.path.join(main_dir, "scenes", scene_file)
        robot_config_path = os.path.join(main_dir, "robot_configs", robot_config_file)
        with open(robot_config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.model = load_model_from_path(scene_file_path)
        self.sim = MjSim(self.model)
        self.__devices = np.array([Device(dev, self.model, self.sim, use_sim=True) for dev in self.config['devices']])
        self.__create_robot_devices(self.config['robots'], use_sim=True)
        self.timer_running = False
        self.robot = self.__get_robot(robot_name="DualUR5")
        
        if osc_device_pairs is None:
            # Specify the controller configuations that should be used for the corresponding devices
            osc_device_configs = [
                ('base', self.__get_controller_config('osc0')),
                ('ur5right', self.__get_controller_config('osc2')),
                ('ur5left', self.__get_controller_config('osc2'))
            ]
        else:
            osc_device_configs = [(dev_name, self.__get_controller_config(osc_name)) for dev_name, osc_name in osc_device_pairs]
            
        # Get the configuration for the nullspace controller
        nullspace_config = self.__get_controller_config('nullspace')
        self.controller = OSC(self.robot, self.sim, osc_device_configs, nullspace_config, admittance = True)

        self.hashids = Hashids()
        self.record = record
        self.render_scene = render_scene
        self.manual_base_ctrl = manual_base_ctrl
        self.picked_up = False
        
        self.data_collect_hz = data_collect_hz
        
        self.prev_time = time.time()
        if self.render_scene:
            self.viewer = MjViewer(self.sim)
            #self.viewer.
        else:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -30
        self.viewer.cam.distance = self.model.stat.extent*1.5
        
        self.gripper_force = 0.00
        self.__action_hist = []
        self.__observation_hist = []
        self.__reward_hist = []
        self.step_count = 0

        self.frames = []
        self.done = False
    
    def targets2actions(self, targets: Dict[str, Target]):
        actions = np.zeros(14, dtype=np.float32)
        if self.ACT_TYPE == ActionType.DPOS:
            state = self.robot.get_device_states()
            cur_pos_left = state['ur5left'][DeviceState.EE_XYZ]
            cur_pos_right = state['ur5right'][DeviceState.EE_XYZ]
        else:
            cur_pos_left = np.array([0,0,0])
            cur_pos_right = np.array([0,0,0])
        actions[:3] = targets['ur5left'].get_xyz() - cur_pos_left
        actions[3:6] = targets['ur5right'].get_xyz() - cur_pos_right
        actions[6:10] = targets['ur5left'].get_quat()
        actions[10:14] = targets['ur5right'].get_quat()
        # actions[14] = targets['base'].get_abg()[2]
        return actions

    def actions2targets(self, actions):
        targets: Dict[str, Target] = {
            'base' : Target(),
            'ur5left' : Target(),
            'ur5right' : Target()
        }
        if self.ACT_TYPE == ActionType.DPOS:
            state = self.robot.get_device_states()
            cur_pos_left = state['ur5left'][DeviceState.EE_XYZ]
            cur_pos_right = state['ur5right'][DeviceState.EE_XYZ]
        else:
            cur_pos_left = np.array([0,0,0])
            cur_pos_right = np.array([0,0,0])
        targets['ur5left'].set_xyz(actions[:3] + cur_pos_left)
        targets['ur5right'].set_xyz(actions[3:6] + cur_pos_right)
        
        targets['ur5left'].set_quat(actions[6:10])
        targets['ur5right'].set_quat(actions[10:14])
        
        # targets['base'].set_abg([0, 0, actions[14]])
        targets['base'].active = self.manual_base_ctrl
        
        return targets

    # def step(self, targets: Dict[str, Target]):
    def step(self, actions):
        targets = self.actions2targets(actions)
        # Generate an OSC signal to steer robot toward the targets
        ctrlr_output = self.controller.generate(targets)
        self.__send_forces(ctrlr_output, gripper_force=self.gripper_force)
        self.sim.step()
        if self.render_scene:
            self.viewer.render()
        obs = self.observe()
        reward = self.__reward()
        if not self.done : self.done = self.is_done()
        #done = self.is_done()
        self.__maybe_record_states(actions, obs, reward)
        return obs, reward, self.done, {}

    # def reset(self):
    #     # Reset the state of the environment to an initial state
    #     obs, _ = self.observe()
    #     return obs
    
    def render(self, close=False):
        self.viewer.render()
        # Render the environment to the screen

    def set_record(self, val):
        """
        Create the CSV to record the expert demonstrations
        """
        self.record = val
        if val == False and len(self.__observation_hist) > 0:
            hash = self.hashids.encode(calendar.timegm(time.gmtime()))
            obs_T_Do = np.asarray(self.__observation_hist) # assert obs_T_Do.shape == (len(obs), self.obs_space.storage_size)
            obsfeat_T_Df = np.ones((obs_T_Do.shape[0], 1))*np.nan # assert obsfeat_T_Df.shape[0] == len(obs)
            adist_T_Pa = np.ones((obs_T_Do.shape[0], 1))*np.nan # assert adist_T_Pa.ndim == 2 and adist_T_Pa.shape[0] == len(obs)
            a_T_Da = np.asarray(self.__action_hist) # assert a_T_Da.shape == (len(obs), self.action_space.storage_size)
            r_T = np.asarray(self.__reward_hist)  # assert r_T.shape == (len(obs),)
            tr = Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T)
            tb = TrajBatch.FromTrajs([tr])
            SUB_DIR_OBS = 'joint' if self.OBS_TYPE == ObservationType.JNT else 'object_t'
            SUB_DIR_ACT = 'pos' if self.ACT_TYPE == ActionType.POS else 'dpos'
            TRAJ_SUB_DIR = f"bimanual_{SUB_DIR_OBS}_{SUB_DIR_ACT}"
            fname = f"{EXPERT_TRAJ_DIR}/{TRAJ_SUB_DIR}/dual_insert_{hash}.proto"
            proto_logger.export_samples_from_expert(tb, [obs_T_Do.shape[0]], fname)
            self.recording = []
            self.__action_hist = []
            self.__observation_hist = []
            self.__reward_hist = []

    def __reward(self):
        return 0.1

    def __maybe_record_states(self, action, observation, reward, verbose=False):
        """
        Optionally record states if self.record is set to True
        """
        if self.record:
            interval = float(1./self.data_collect_hz)
            if (time.time() - self.prev_time) >= interval:
                self.prev_time = time.time()
                self.__observation_hist.append(observation)
                self.__action_hist.append(action)
                self.__reward_hist.append(reward)
        else:
            if verbose:
                print("[ record_states() ]: Record is set to False: Not recording!")

    def __send_forces(self, forces, gripper_force:float=None):
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
            for idx in [GRIP_IDX_RIGHT, GRIP_IDX_LEFT]:
                self.sim.data.ctrl[idx] = gripper_force
        
    def observe(self):
        state = self.robot.get_device_states()

        if self.OBS_TYPE == ObservationType.JNT:
            observations = np.concatenate(( 
                state['base'][DeviceState.Q_ACTUATED],                  # 0-1
                state['ur5left'][DeviceState.Q_ACTUATED],               # 2-7
                state['ur5right'][DeviceState.Q_ACTUATED],              # 8-13
                
                state['ur5left'][DeviceState.FORCE],                    # 14-16
                state['ur5left'][DeviceState.TORQUE],                   # 17-19
                
                state['ur5right'][DeviceState.FORCE],                   # 20-22
                state['ur5right'][DeviceState.TORQUE],                  # 23-25
            ), dtype=np.float32)

        elif self.OBS_TYPE == ObservationType.OBJ:
            observations = np.concatenate((
                self.sim.data.get_body_xpos("grommet_11mm"),            # 0-3
                self.sim.data.get_body_xpos("quad_peg"),                # 3-6

                self.sim.data.get_body_xquat("grommet_11mm"),           # 6-10
                self.sim.data.get_body_xquat("quad_peg"),               # 10-14

                state['ur5left'][DeviceState.FORCE],                    # 14-17
                state['ur5left'][DeviceState.TORQUE],                   # 17-20
                
                state['ur5right'][DeviceState.FORCE],                   # 20-23
                state['ur5right'][DeviceState.TORQUE],                  # 23-26
            ))

        return observations
    
    # def get_bip_state(self):
    #     state = self.robot.get_device_states()
    #     state_full = np.concatenate(( 
    #         state['base'][DeviceState.Q_ACTUATED],                  # 0-1
    #         state['ur5left'][DeviceState.Q_ACTUATED],               # 2-7
    #         state['ur5right'][DeviceState.Q_ACTUATED],              # 8-13
            
    #         state['ur5left'][DeviceState.FORCE],                    # 14-16
    #         state['ur5left'][DeviceState.TORQUE],                   # 17-19
            
    #         state['ur5right'][DeviceState.FORCE],                   # 20-22
    #         state['ur5right'][DeviceState.TORQUE],                  # 23-25

    #         state['ur5left'][DeviceState.EE_XYZ],                   # 26-28
    #         state['ur5right'][DeviceState.EE_XYZ],                  # 29-31
            
    #         self.__fix_rot(state['ur5left'][DeviceState.EE_QUAT]),  # 32-35
    #         self.__fix_rot(state['ur5right'][DeviceState.EE_QUAT])  # 36-39
    #     ), dtype=np.float32)
        
    #     return state_full

    def __fix_rot(self, rot):
        """
        Ensure the w term is positive
        """
        rot = np.asarray(rot)
        if rot[0] < 0:
            rot *= -1.0
        return rot.tolist()


    def sleep_for(self, sleep_time: float):
        assert self.timer_running == False
        self.timer_running = True
        time.sleep(sleep_time)
        self.timer_running = False

    def __create_robot_devices(self, robot_yml: Dict, use_sim: bool):
        robots = np.array([])
        all_robot_device_idxs = np.array([], dtype=np.int32)
        for rbt in robot_yml:
            robot_device_idxs = rbt['device_ids']
            all_robot_device_idxs = np.hstack([all_robot_device_idxs, robot_device_idxs])
            robot = Robot(self.__devices[robot_device_idxs], rbt['name'], self.sim, use_sim)
            robots = np.append(robots, robot)
        
        all_idxs = np.arange(len(self.__devices))
        keep_idxs = np.setdiff1d(all_idxs, all_robot_device_idxs)
        self.__devices = np.hstack([self.__devices[keep_idxs], robots])
    
        
    def __get_robot(self, robot_name: str) -> Robot:
        for device in self.__devices:
            if type(device) == Robot:
                if device.name == robot_name:
                    return device
    
    def __get_controller_config(self, name: str) -> Dict:
        ctrlr_conf = self.config['controller_configs']
        for entry in ctrlr_conf:
            if entry['name'] == name:
                return entry
    
    def set_free_joint_qpos(self, free_joint_name, quat=None, pos=None):
        jnt_id = self.sim.model.joint_name2id(free_joint_name)
        offset = self.sim.model.jnt_qposadr[jnt_id]
        if quat is not None:
            quat_idxs = np.arange(offset+3, offset+7) # Grab the quaternion idxs
            self.sim.data.qpos[quat_idxs] = quat
        if pos is not None:
            pos_idxs = np.arange(offset, offset+3)
            self.sim.data.qpos[pos_idxs] = pos