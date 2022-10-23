import numpy as np
import mujoco_py as mjp
import time
from irl_control import Device

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
    
    def get_jacobian(self, controlled_devices):
        J = np.array([])
        J_idxs = dict()
        start_idx = 0
        for name in controlled_devices:
            J_sub = self.sub_devices_dict[name].get_jacobian()
            J_idxs[name] = np.arange(start_idx, start_idx + J_sub.shape[0])
            start_idx += J_sub.shape[0]
            J_sub = J_sub[:, self.joint_ids_all]
            J = np.vstack([J, J_sub]) if J.size else J_sub
        
        return J, J_idxs

    def get_jointangles(self):
        q = self.sim.data.qpos[self.joint_ids_all]
        return q

    def set_jointangles(self,q):
        self.sim.data.qpos[self.joint_ids_all] = q
    
    def get_M(self):
        mjp.cymj._mj_fullM(self.sim.model, self.M_vec, self.sim.data.qM)
        M = self.M_vec.reshape(self.num_scene_joints, self.num_scene_joints)
        M = M[np.ix_(self.joint_ids_all, self.joint_ids_all)]
        return M

    def get_rotation(self,name):
        mjp.cymj._mju_quat2Mat(self._R9,self.sim.data.get_body_xquat(name))
        R = self._R9.reshape((3,3))
        return R

    def get_dq(self):
        dq = self.sim.data.qvel[self.joint_ids_all]
        return dq
    
    def start(self):
        self.running = True
        while self.running:
            for dev in self.sub_devices:
                dev.update_state()
            time.sleep(0.0001)

    def stop(self):
        self.running = False

    def get_device(self, device_name: str) -> Device:
        return self.sub_devices_dict[device_name]

    def getState(self):
        """
        Get's the state of all the devices connected
        """
        targets = ["q", "dq", "ee_xyz", "ee_quat"]
        state = {}
        state["devices"] = self.sub_devices_dict.keys()
        for device in state["devices"]:
            for target in targets:
                joint_ids = self.sub_devices_dict[device].get_actuator_joint_ids()
                state[target + "_" + device] = self.get_device(device).get_state(target, joint_ids)
            state["force_" + device] = self.get_device(device).get_force()
            state["torque_" + device] = self.get_device(device).get_torque()
        return state