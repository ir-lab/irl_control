from irl_control.device import DeviceState
import numpy as np
import mujoco_py as mjp
import time
from irl_control.device import Device, DeviceState
from enum import Enum
from threading import Lock
from typing import Dict, Any, List
import copy

class RobotState(Enum):
    M = 'INERTIA'
    DQ = 'DQ'
    J = 'JACOBIAN'

class Robot():
    def __init__(self, sub_devices: List[Device], robot_name, sim, use_sim, collect_hz=1000):
        self.sim = sim
        self.__use_sim = use_sim
        self.sub_devices = sub_devices
        self.sub_devices_dict: Dict[str, Device] = dict()
        for dev in self.sub_devices:
            self.sub_devices_dict[dev.name] = dev

        self.name = robot_name
        self.num_scene_joints = self.sim.model.nv
        self.M_vec = np.zeros(self.num_scene_joints**2)
        self.joint_ids_all = np.array([], dtype=np.int32)
        for dev in self.sub_devices:
            self.joint_ids_all = np.hstack([self.joint_ids_all, dev.joint_ids_all])
        self.joint_ids_all = np.sort(np.unique(self.joint_ids_all))
        self.num_joints_total = len(self.joint_ids_all)
        self.running = False
        self.__state_locks: Dict[RobotState, Lock] = dict([(key, Lock()) for key in RobotState])
        self.__state_var_map: Dict[RobotState, function] = {
            RobotState.M : lambda : self.__get_M(),
            RobotState.DQ : lambda : self.__get_dq(),
            RobotState.J : lambda : self.__get_jacobian()
        }
        self.__state: Dict[RobotState, Any] = dict()
        self.data_collect_hz = collect_hz

    
    def __get_jacobian(self):
        """
            Return the Jacobians for all of the devices,
            so that OSC can stack them according to provided the target entries
        """
        Js = dict()
        J_idxs = dict()
        start_idx = 0
        for name, device in self.sub_devices_dict.items():
            J_sub = device.get_state(DeviceState.J)
            J_idxs[name] = np.arange(start_idx, start_idx + J_sub.shape[0])
            start_idx += J_sub.shape[0]
            J_sub = J_sub[:, self.joint_ids_all]
            Js[name] = J_sub
        return Js, J_idxs
    
    def __get_dq(self):
        # dq = self.sim.data.qvel[self.joint_ids_all]
        dq = np.zeros(self.joint_ids_all.shape)
        for dev in self.sub_devices:
            dq[dev.get_all_joint_ids()] = dev.get_state(DeviceState.DQ)
        return dq


    def __get_M(self):
        mjp.cymj._mj_fullM(self.sim.model, self.M_vec, self.sim.data.qM)
        M = self.M_vec.reshape(self.num_scene_joints, self.num_scene_joints)
        M = M[np.ix_(self.joint_ids_all, self.joint_ids_all)]
        return M

    def get_state(self, state_var: RobotState):
        if self.__use_sim:
            func = self.__state_var_map[state_var]
            state = copy.copy(func())
        else:
            self.__state_locks[state_var].acquire()
            state = copy.copy(self.__state[state_var])
            self.__state_locks[state_var].release()
        return state

    def __set_state(self, state_var: RobotState):
        assert self.__use_sim is False
        self.__state_locks[state_var].acquire()
        func = self.__state_var_map[state_var]
        value = func()
        self.__state[state_var] = copy.copy(value) # Make sure to copy (or else reference will stick to Dict value)
        self.__state_locks[state_var].release()

    def is_running(self):
        return self.running
    
    def is_using_sim(self):
        return self.__use_sim

    def __update_state(self):
        assert self.__use_sim is False
        for var in RobotState:
            self.__set_state(var)
    
    def start(self):
        assert self.running is False and self.__use_sim is False
        self.running = True
        interval = float(1.0/float(self.data_collect_hz))
        prev_time = time.time()
        while self.running:
            for dev in self.sub_devices:
                dev.update_state()
            self.__update_state()
            curr_time = time.time()
            diff = curr_time - prev_time
            delay = max(interval - diff, 0)
            time.sleep(delay)
            prev_time = curr_time
    
    def stop(self):
        assert self.running is True and self.__use_sim is False
        self.running = False

    def get_device(self, device_name: str) -> Device:
        return self.sub_devices_dict[device_name]

    def get_all_states(self):
        """
        Get's the state of all the devices connected plus the robot states
        """
        state = {}
        for device_name, device in self.sub_devices_dict.items():
            state[device_name] = device.get_all_states()
        
        for key in RobotState:
            state[key] = self.get_state(key)
        
        return state
    
    def get_device_states(self):
        """
        Get's the state of all the devices connected
        """
        state = {}
        for device_name, device in self.sub_devices_dict.items():
            state[device_name] = device.get_all_states()
        return state