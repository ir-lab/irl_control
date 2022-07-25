import numpy as np
from typing import Any, List

class Target():
    """
        The Target class includes a target vector 
        (x,y,z,a,b,g) with (a = roll, b = pitch, g = yaw) and a vector of 
        their velocities (x', y', z', a', b', g'). Expects a list (or ndarray) 
        of these values to initialize the class.
    """
    def __init__(self, xyz_abg : List = np.zeros(6), xyz_abg_vel : List = np.zeros(6)):
        self.xyz = np.array(xyz_abg)[:3]
        self.abg = np.array(xyz_abg)[3:]
        self.xyz_vel = np.array(xyz_abg_vel)[:3]
        self.abg_vel = np.array(xyz_abg_vel)[3:]

class ControllerConfig():
    def __init__(self, ctrlr_dict):
        self.ctrlr_dict = ctrlr_dict
    
    def __getitem__(self, __name: str) -> Any:
         return self.ctrlr_dict[__name]
    
    def get_params(self, keys):
        return [self.ctrlr_dict[key] for key in keys]

    def __setitem__(self, __name: str, __value: Any) -> None:
        self.ctrlr_dict[__name] = __value