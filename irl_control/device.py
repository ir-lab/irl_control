import numpy as np
from threading import Lock
from typing import Dict, Any
from enum import Enum
import copy

class DeviceState(Enum):
    Q = 'Q'
    Q_ACTUATED = 'Q_ACTUATED'
    DQ = 'DQ'
    DQ_ACTUATED = 'DQ_ACTUATED'
    DDQ = 'DDQ'
    EE_XYZ = 'EE_XYZ'
    EE_XYZ_VEL = 'EE_XYZ_VEL'
    EE_QUAT = 'EE_QUAT'
    FORCE = 'FORCE'
    TORQUE = 'TORQUE'
    J = 'JACOBIAN'

class Device():
    """
    The Device class encapsulates the device parameters specified in the yaml file
    that is passed to MujocoApp. It collects data from the simulator, obtaining the 
    desired device states.
    """
    def __init__(self, device_yml: Dict, model, sim, use_sim: bool):
        self.sim = sim
        self.__use_sim = use_sim
        # Assign all of the yaml parameters
        self.name = device_yml['name']
        self.max_vel = device_yml.get('max_vel')
        #self.max_vel = device_yml['max_vel']
        self.EE = device_yml['EE']
        self.ctrlr_dof_xyz = device_yml['ctrlr_dof_xyz']
        self.ctrlr_dof_abg = device_yml['ctrlr_dof_abg']
        self.ctrlr_dof = np.hstack([self.ctrlr_dof_xyz, self.ctrlr_dof_abg])
        self.start_angles = np.array(device_yml['start_angles'])
        self.num_gripper_joints = device_yml['num_gripper_joints']
        
        # Check if the user specifies a start body for the while loop to terminte at
        try:
            start_body = model.body_name2id(device_yml['start_body'])
        except:
            start_body = 0
        
        # Reference: ABR Control
        # Get the joint ids, using the specified EE / start body 
        # start with the end-effector (EE) and work back to the world body
        body_id = model.body_name2id(self.EE)
        joint_ids = []
        joint_names = []
        while model.body_parentid[body_id] != 0 and model.body_parentid[body_id] != start_body:
            jntadrs_start = model.body_jntadr[body_id]
            tmp_ids = []
            tmp_names = []
            for ii in range(model.body_jntnum[body_id]):
                tmp_ids.append(jntadrs_start + ii)
                tmp_names.append(model.joint_id2name(tmp_ids[-1]))
            joint_ids += tmp_ids[::-1]
            joint_names += tmp_names[::-1]
            body_id = model.body_parentid[body_id]
        # flip the list so it starts with the base of the arm / first joint
        self.joint_names = joint_names[::-1]
        self.joint_ids = np.array(joint_ids[::-1])
        
        gripper_start_idx = self.joint_ids[-1] + 1
        self.gripper_ids = np.arange(gripper_start_idx, 
                                gripper_start_idx + self.num_gripper_joints)
        self.joint_ids_all = np.hstack([self.joint_ids, self.gripper_ids])

        # Find the actuator and control indices
        actuator_trnids = model.actuator_trnid[:,0]
        self.ctrl_idxs = np.intersect1d(actuator_trnids, self.joint_ids_all, return_indices=True)[1]
        self.actuator_trnids = actuator_trnids[self.ctrl_idxs]

        if self.name == "ur5right" or self.name == "ur5left":
            self.sim.data.qpos[self.joint_ids] = np.copy(self.start_angles)
        elif self.name == "base":
            self.sim.data.qpos[self.joint_ids] = np.copy(self.start_angles)
        self.sim.forward()

        # Check that the 
        if np.sum(np.hstack([self.ctrlr_dof_xyz, self.ctrlr_dof_abg])) > len(self.joint_ids):
            print("Fewer DOF than specified")
        
        # Initialize dicts to keep track of the state variables and locks
        self.__state_var_map: Dict[DeviceState, function] = {
            DeviceState.Q : lambda : self.sim.data.qpos[self.joint_ids_all],
            DeviceState.Q_ACTUATED : lambda : self.sim.data.qpos[self.joint_ids],
            DeviceState.DQ : lambda : self.sim.data.qvel[self.joint_ids_all],
            DeviceState.DQ_ACTUATED : lambda : self.sim.data.qvel[self.joint_ids],
            DeviceState.DDQ : lambda : self.sim.data.qacc[self.joint_ids_all],
            DeviceState.EE_XYZ : lambda : self.sim.data.get_body_xpos(self.EE),
            DeviceState.EE_XYZ_VEL : lambda : self.sim.data.get_body_xvelp(self.EE),
            DeviceState.EE_QUAT : lambda : self.sim.data.get_body_xquat(self.EE),
            DeviceState.FORCE : lambda : self.__get_force(),
            DeviceState.TORQUE : lambda : self.__get_torque(),
            DeviceState.J : lambda : self.__get_jacobian()
        }
        
        self.__state: Dict[DeviceState, Any] = dict()
        self.__state_locks: Dict[DeviceState, Lock] = dict([(key, Lock()) for key in DeviceState])
        
        # These are the that keys we should use when returning data from get_all_states()
        self.concise_state_vars = [
            DeviceState.Q_ACTUATED, 
            DeviceState.DQ_ACTUATED, 
            DeviceState.EE_XYZ, 
            DeviceState.EE_XYZ_VEL, 
            DeviceState.EE_QUAT,
            DeviceState.FORCE,
            DeviceState.TORQUE
        ]

    def __get_jacobian(self, full=False):
        """
        NOTE: Returns either:
        1) The full jacobian (of the Device, using its EE), if full==True 
        2) The full jacobian evaluated at the controlled DoF, if full==False 
        The parameter, full=False, is added in case we decide for the get methods 
        to take in arguments (currently not supported).
        """
        J = np.array([])
        # Get the jacobian for the x,y,z components
        J = self.sim.data.get_body_jacp(self.EE)
        J = J.reshape(3, -1)
        # Get the jacobian for the a,b,g components
        Jr = self.sim.data.get_body_jacr(self.EE)
        Jr = Jr.reshape(3, -1)
        J = np.vstack([J, Jr]) if J.size else Jr
        if full == False:
            J = J[self.ctrlr_dof]
        return J

    def __get_R(self):
        """
        Get rotation matrix for device's ft_frame
        """
        if self.name == "ur5right":
            return self.sim.data.get_site_xmat("ft_frame_ur5right")
        if self.name == "ur5left":
            return self.sim.data.get_site_xmat("ft_frame_ur5left")

    def __get_force(self):
        """
        Get the external forces, used (for admittance control) acting upon
        the gripper sensors
        """
        if self.name == "ur5right":
            force = np.matmul(self.__get_R(), self.sim.data.sensordata[0:3])
            return force
        if self.name == "ur5left":
            force = np.matmul(self.__get_R(), self.sim.data.sensordata[6:9])
            return force
        else:
            return np.zeros(3)
            
    def __get_torque(self):
        """
        Get the external torques, used (for admittance control) acting upon
        the gripper sensors
        """
        if self.name == "ur5right":
            force = np.matmul(self.__get_R(), self.sim.data.sensordata[3:6])
            return force
        if self.name == "ur5left":
            force = np.matmul(self.__get_R(), self.sim.data.sensordata[9:12])
            return force
        else:
            return np.zeros(3)

    def __set_state(self, state_var: DeviceState):
        """
        Set the state of the device corresponding to the key value (if exists)    
        """
        assert self.__use_sim is False
        self.__state_locks[state_var].acquire()
        var_func = self.__state_var_map[state_var]
        var_value = var_func()
        self.__state[state_var] = copy.copy(var_value) # Make sure to copy (or else reference will stick to Dict value)
        self.__state_locks[state_var].release()

    def get_state(self, state_var: DeviceState):
        """
        Get the state of the device corresponding to the key value (if exists)
        """
        if self.__use_sim:
            func = self.__state_var_map[state_var]
            state = copy.copy(func())
        else:
            self.__state_locks[state_var].acquire()
            state = copy.copy(self.__state[state_var])
            self.__state_locks[state_var].release()
        return state
    
    def get_all_states(self):
        return dict([(key, self.get_state(key)) for key in self.concise_state_vars])
    
    def update_state(self):
        """
        This should running in a thread: Robot.start()
        """
        assert self.__use_sim is False
        for var in DeviceState:
            self.__set_state(var)
        
    def get_all_joint_ids(self):
        return self.joint_ids_all
    
    def get_actuator_joint_ids(self):
        return self.joint_ids
    
    def get_gripper_joint_ids(self):
        return self.gripper_ids