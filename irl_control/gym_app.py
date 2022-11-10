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

IRL_DATA_DIR = Path(os.environ.get('HOME') + '/irl_control_container/data')
EXPERT_TRAJ_DIR = IRL_DATA_DIR / 'expert_trajectories' / 'bimanual'

GRIP_IDX_RIGHT = 7
GRIP_IDX_LEFT = 14

class GymBimanual(gym.Env):
    def __init__(self, robot_config_file, scene_file, osc_device_pairs = None):
        
        self.action_space = gym.spaces.Box(low=-2.0*np.ones(14), high=2.0*np.ones(14))
        self.observation_space = gym.spaces.Box(low=-15*np.ones(25), high=15*np.ones(25))
        
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
        self.record = False
        self.render_scene = True

        self.prev_time = time.time()
        if self.render_scene:
            self.viewer = MjViewer(self.sim)
        else:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -30
        self.viewer.cam.distance = self.model.stat.extent*1.5
        
        self.gripper_force = 0.00
        self.__action_hist = []
        self.__observation_hist = []
        self.__reward_hist = []
    
    def targets2actions(self, targets: Dict[str, Target]):
        actions = np.zeros(14)
        actions[:3] = targets['ur5left'].get_xyz()
        actions[3:6] = targets['ur5right'].get_xyz()
        actions[6:10] = targets['ur5left'].get_quat()
        actions[10:14] = targets['ur5right'].get_quat()
        return actions

    def step(self, targets: Dict[str, Target]):
        # Generate an OSC signal to steer robot toward the targets
        ctrlr_output = self.controller.generate(targets)
        self.__send_forces(ctrlr_output, gripper_force=self.gripper_force)
        self.sim.step()
        if self.render_scene:
            self.render()
        
        actions = self.targets2actions(targets)
        obs = self.__observe()
        reward = self.__reward()
        done = self.__is_done()
        self.__maybe_record_states(actions, obs, reward)
        return obs, reward, done

    # def reset(self):
    #     # Reset the state of the environment to an initial state
    #     obs, _ = self.__observe()
    #     return obs
    
    def render(self, close=False):
        self.viewer.render()
        # Render the environment to the screen
        ...

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
            fname = f"{EXPERT_TRAJ_DIR}/dual_insert_{hash}.proto"
            proto_logger.export_samples_from_expert(tb, [obs_T_Do.shape[0]], fname)
            self.recording = []

    def __reward(self):
        return -1

    def __is_done(self):
        return False

    def __maybe_record_states(self, action, observation, reward, verbose=False):
        """
        Optionally record states if self.record is set to True
        """
        if self.record:
            interval = float(1./60)
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
        
    def __observe(self):
        state = self.robot.get_device_states()
        observations = np.concatenate(( 
            state['base'][DeviceState.Q_ACTUATED],                  # 0-1
            state['ur5left'][DeviceState.Q_ACTUATED],               # 2-7
            state['ur5right'][DeviceState.Q_ACTUATED],              # 8-13
            
            state['ur5left'][DeviceState.FORCE],                    # 14-16
            state['ur5left'][DeviceState.TORQUE],                   # 17-19
            
            state['ur5right'][DeviceState.FORCE],                   # 20-22
            state['ur5right'][DeviceState.TORQUE],                  # 23-25
        ))

        return observations
    
    def get_bip_state(self):
        state = self.robot.get_device_states()
        state_full = np.concatenate(( 
            state['base'][DeviceState.Q_ACTUATED],                  # 0-1
            state['ur5left'][DeviceState.Q_ACTUATED],               # 2-7
            state['ur5right'][DeviceState.Q_ACTUATED],              # 8-13
            
            state['ur5left'][DeviceState.FORCE],                    # 14-16
            state['ur5left'][DeviceState.TORQUE],                   # 17-19
            
            state['ur5right'][DeviceState.FORCE],                   # 20-22
            state['ur5right'][DeviceState.TORQUE],                  # 23-25

            state['ur5left'][DeviceState.EE_XYZ],                   # 26-28
            state['ur5right'][DeviceState.EE_XYZ],                  # 29-31
            
            self.__fix_rot(state['ur5left'][DeviceState.EE_QUAT]),  # 32-35
            self.__fix_rot(state['ur5right'][DeviceState.EE_QUAT])  # 36-39
        ))
        
        return state_full

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