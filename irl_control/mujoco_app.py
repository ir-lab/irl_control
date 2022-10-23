import numpy as np
from mujoco_py import load_model_from_path, MjSim
import irl_control
from irl_control import Device, Robot
from typing import Dict
import time
import os
import yaml

class MujocoApp():
    def __init__(self, robot_config_file : str = None, scene_file : str = None):
        main_dir = os.path.dirname(irl_control.__file__)
        scene_file_path = os.path.join(main_dir, "scenes", scene_file)
        robot_config_path = os.path.join(main_dir, "robot_configs", robot_config_file)
        with open(robot_config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.model = load_model_from_path(scene_file_path)
        self.sim = MjSim(self.model)
        self.devices = np.array([Device(dev, self.model, self.sim) for dev in self.config['devices']])
        self.create_robot_devices(self.config['robots'])
        self.controller_configs = self.config['controller_configs']
        self.timer_running = False

    def create_robot_devices(self, robot_yml: Dict):
        robots = np.array([])
        all_robot_device_idxs = np.array([], dtype=np.int32)
        for rbt in robot_yml:
            robot_device_idxs = rbt['device_ids']
            all_robot_device_idxs = np.hstack([all_robot_device_idxs, robot_device_idxs])
            robot = Robot(self.devices[robot_device_idxs], rbt['name'], self.sim)
            robots = np.append(robots, robot)
        
        all_idxs = np.arange(len(self.devices))
        keep_idxs = np.setdiff1d(all_idxs, all_robot_device_idxs)
        self.devices = np.hstack([self.devices[keep_idxs], robots])
    
    def sleep_for(self, sleep_time: float):
        assert self.timer_running == False
        self.timer_running = True
        time.sleep(sleep_time)
        self.timer_running = False
        
    def get_robot(self, robot_name: str) -> Robot:
        for device in self.devices:
            if type(device) == Robot:
                if device.name == robot_name:
                    return device
    
    def get_controller_config(self, name: str) -> Dict:
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