import numpy as np
from typing import Any, List
from transforms3d.euler import quat2euler, euler2quat

class Target():
    """
        The Target class holds a target vector for both orientation (quaternion) and position (xyz)
        NOTE: Quat is stored as w, x, y, z 
    """
    def __init__(self, xyz_abg : List = np.zeros(6), xyz_abg_vel : List = np.zeros(6)):
        assert len(xyz_abg) == 6 and len(xyz_abg_vel) == 6
        self.__xyz = np.array(xyz_abg)[:3]
        self.__xyz_vel = np.array(xyz_abg_vel)[:3]
        self.__quat = np.array(euler2quat(*xyz_abg[3:]))
        self.__quat_vel = np.array(euler2quat(*xyz_abg_vel[3:]))
    
    def get_xyz(self):
        return self.__xyz

    def get_xyz_vel(self):
        return self.__xyz_vel
    
    def get_quat(self):
        return self.__quat

    def get_quat_vel(self):
        return np.asarray(self.__quat_vel)
    
    def get_abg(self):
        return np.asarray(quat2euler(self.__quat))
    
    def get_abg_vel(self):
        return np.asarray(quat2euler(self.__quat_vel))
    
    def set_xyz(self, xyz):
        assert len(xyz) == 3
        self.__xyz = np.asarray(xyz)
    
    def set_xyz_vel(self, xyz_vel):
        assert len(xyz_vel) == 3
        self.__xyz_vel = np.asarray(xyz_vel)
    
    def set_quat(self, quat):
        assert len(quat) == 4
        self.__quat = np.asarray(quat)

    def set_quat_vel(self, quat_vel):
        assert len(quat_vel) == 4
        self.__quat_vel = np.asarray(quat_vel)

    def set_abg(self, abg):
        assert len(abg) == 3
        self.__quat = np.asarray(euler2quat(*abg))

    def set_abg_vel(self, abg_vel):
        assert len(abg_vel) == 3
        self.__quat_vel = np.asarray(euler2quat(*abg_vel))

    def set_all_quat(self, xyz, quat):
        assert len(xyz) == 3 and len(quat) == 4
        self.__xyz = np.asarray(xyz)
        self.__quat = np.asarray(quat)

    def set_all_abg(self, xyz, abg):
        assert len(xyz) == 3 and len(abg) == 3
        self.__xyz = np.asarray(xyz)
        self.__quat = np.asarray(euler2quat(*abg))

class ControllerConfig():
    def __init__(self, ctrlr_dict):
        self.ctrlr_dict = ctrlr_dict
    
    def __getitem__(self, __name: str) -> Any:
         return self.ctrlr_dict[__name]
    
    def get_params(self, keys):
        return [self.ctrlr_dict[key] for key in keys]

    def __setitem__(self, __name: str, __value: Any) -> None:
        self.ctrlr_dict[__name] = __value