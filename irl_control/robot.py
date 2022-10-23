from irl_control.device import DeviceState
import numpy as np
import mujoco_py as mjp
import time
from irl_control.device import Device, DeviceState
import threading
from enum import Enum
from threading import Lock
from typing import Dict, Any
import copy

class RobotState(Enum):
    M = 'INERTIA'
    DQ = 'DQ'
    J = 'JACOBIAN'

class Robot():
    def __init__(self, sub_devices, robot_name, sim):
        self.sim = sim
        self.sub_devices = sub_devices
        self.sub_devices_dict = dict()
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
        self.data_collect_hz = 50

    
    def __get_jacobian(self):
        controlled_devices = self.sub_devices_dict.keys()
        Js = dict()
        J_idxs = dict()
        start_idx = 0
        for name in controlled_devices:
            J_sub = self.sub_devices_dict[name].get_state(DeviceState.J)
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

    def get_state(self, robot_state: RobotState):
        self.__state_locks[robot_state].acquire()
        state = self.__state[robot_state]
        self.__state_locks[robot_state].release()
        return state
        
    def __set_state(self, state_var: RobotState):
        self.__state_locks[state_var].acquire()
        func = self.__state_var_map[state_var]
        value = func()
        self.__state[state_var] = copy.copy(value) # Make sure to copy (or else reference will stick to Dict value)
        self.__state_locks[state_var].release()

    def is_running(self):
        return self.running
    
    # def collect_robot_state(self):
    #     while self.running:
    #         for var in RobotState:
    #             self.__set_state(var)
    #         time.sleep( float(1.0/float(self.data_collect_hz)) )

    # def collect_device_state(self, device):
    #     while self.running:
    #         device.update_state()
    #         time.sleep( float(1.0/float(self.data_collect_hz)) )

    def __update_state(self):
        for var in RobotState:
            self.__set_state(var)
    
    def start(self):
        self.running = True
        # device_data_threads = [threading.Thread(target=self.collect_device_state, args=[dev]) for dev in self.sub_devices]
        # robot_data_thread = threading.Thread(target=self.collect_robot_state)
        # for thread in device_data_threads:
        #     thread.start()
        # robot_data_thread.start()
        while self.running:
            for dev in self.sub_devices:
                dev.update_state()
            self.__update_state()
            time.sleep( float(1.0/float(self.data_collect_hz)) )
    
    def stop(self):
        self.running = False

    def get_device(self, device_name: str) -> Device:
        return self.sub_devices_dict[device_name]

    def get_device_states(self):
        """
        Get's the state of all the devices connected
        """
        state = {}
        for device_name, device in self.sub_devices_dict.items():
            state[device_name] = device.get_all_states()
        return state