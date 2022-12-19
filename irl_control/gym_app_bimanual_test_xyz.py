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
import datetime
import cv2

IRL_DATA_DIR = Path(os.environ.get('HOME') + '/irl_control_container/data')
EXPERT_TRAJ_DIR = IRL_DATA_DIR / 'expert_trajectories' / 'bimanual_test_xyz_storage'

GRIP_IDX_RIGHT = 7
GRIP_IDX_LEFT = 14

class GymBimanualTestAppXYZ(gym.Env):
    def __init__(self, robot_config_file, scene_file, osc_device_pairs=None, data_collect_hz=100, 
                 render_scene=False, record=False, manual_base_ctrl=False):
        
        self.action_space = gym.spaces.Box(low=0*np.ones(3, dtype=np.float32), high=2.0*np.ones(3, dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=0*np.ones(3, dtype=np.float32), high=2.0*np.ones(3, dtype=np.float32))
        
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
        else:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -30
        self.viewer.cam.distance = self.model.stat.extent*2
        
        self.gripper_force = 0.00
        self.__action_hist = []
        self.__observation_hist = []
        self.__reward_hist = []
        self.step_count = 0
    
        self.base_idxs = [0]
        self.ur5right_idxs = [1, 2, 3, 4, 5, 6]
        self.ur5left_idxs = [8, 9, 10, 11, 12, 13]
        self.__goal_pos_r = None
        self.num_steps = 0
        self.received_reward = False
        self.action_count = 0
        if not self.render_scene:
            self.date_suffix = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.image_path = IRL_DATA_DIR / 'camera' / f"bimanual_xyz_{self.date_suffix}"
            os.mkdir(self.image_path)
            self.image_id = 0
    
    def targets2actions(self, targets: Dict[str, Target]):
        actions = np.zeros(3, dtype=np.float32)
        actions[:3] = targets['ur5right'].get_xyz()
        # actions[3:6] = targets['ur5right'].get_xyz()
        # actions[14] = targets['base'].get_abg()[2]
        return actions

    def actions2targets(self, actions):
        targets: Dict[str, Target] = {
            'ur5left' : Target(),
            'ur5right' : Target()
        }

        targets['ur5left'].set_xyz([-0.30148561,  0.46516144,  0.40242017])
        # if self.action_count % 10 == 0:
        #     targets['ur5right'].set_xyz(self.get_goal_pos_r())
        #     self.last_action = actions
        # else:
        targets['ur5right'].set_xyz(actions)
        
        self.sim.data.set_mocap_pos('target_red', targets['ur5right'].get_xyz())
        self.action_count += 1
        # if round(np.random.rand()*100) % 3 == 0:
        #     targets['ur5right'].set_xyz(actions)
        # else:
        #     targets['ur5right'].set_xyz(self.get_goal_pos_r())
        return targets
    

    def step(self, actions):
        assert len(actions) == 3
        targets = self.actions2targets(actions)
        self.sim.data.set_mocap_pos('target_blue', self.get_goal_pos_r())
        # print(targets['ur5right'].get_xyz())
        # Generate an OSC signal to steer robot toward the targets
        ctrlr_output = self.controller.generate(targets)
        self.__send_forces(ctrlr_output, gripper_force=self.gripper_force)
        sim_failed = False
        try:
            self.sim.step()
            if self.render_scene:
                self.viewer.render()
            else:
                if self.image_id % 200 == 0:
                    self.viewer.render(240, 240, -1)
                    data = np.asarray(self.viewer.read_pixels(240, 240, depth=False)[::-1, :, :], dtype=np.uint8)
                    if data is not None:
                        cv2.imwrite(str(self.image_path / f"bm_{self.image_id/100}.png"), cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
                self.image_id += 1
        except:
            sim_failed = True
        
        # actions = self.targets2actions(targets)
        obs = self.__observe()
        reward = self.__reward()
        done = self.__is_done() if not sim_failed else True
        self.__maybe_record_states(actions, obs, reward)
        return obs, reward, done, {}

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
        assert self.__goal_pos_r is not None
        ee_r = self.robot.sub_devices_dict['ur5right'].get_state(DeviceState.EE_XYZ)
        error = np.linalg.norm(ee_r - self.get_goal_pos_r())
        return -1*np.log(2*error)

    def set_goal_pos_r(self, pos_r):
        self.__goal_pos_r = pos_r
    
    def get_goal_pos_r(self):
        return self.__goal_pos_r

    def __is_done(self):
        assert self.__goal_pos_r is not None
        ee_r = self.robot.sub_devices_dict['ur5right'].get_state(DeviceState.EE_XYZ)
        error = np.linalg.norm(ee_r - self.get_goal_pos_r())
        hands_down = ee_r[2] < 0.1
        if (error < 0.02):
            return True
        else:
            return False

    def __maybe_record_states(self, action, observation, reward, verbose=False):
        """
        Optionally record states if self.record is set to True
        """
        self.__observation_hist.append(observation)
        self.__action_hist.append(action)
        self.__reward_hist.append(reward)
        # if self.record:
        #     interval = float(1./self.data_collect_hz)
        #     if (time.time() - self.prev_time) >= interval:
        #         self.prev_time = time.time()
        #         self.__observation_hist.append(observation)
        #         self.__action_hist.append(action)
        #         self.__reward_hist.append(reward)
        # else:
        #     if verbose:
        #         print("[ record_states() ]: Record is set to False: Not recording!")

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
            # state['base'][DeviceState.Q_ACTUATED], # 1
            # state['ur5left'][DeviceState.EE_XYZ],  # 2-4
            state['ur5right'][DeviceState.EE_XYZ], # 5-7

        ), dtype=np.float32)

        return observations
    
    def get_bip_state(self):
        raise NotImplementedError

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