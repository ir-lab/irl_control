import numpy as np
from threading import Lock
from typing import List, Dict

class Device():
    """
        The Device class encapsulates the device parameters specified in the yaml file
        that is passed to MujocoApp. It collects data from the simulator, obtaining the 
        desired device states.
    """
    def __init__(self, device_yml: Dict, model, sim):
        self.sim = sim
        # Assign all of the yaml parameters
        self.name = device_yml['name']
        self.max_vel = device_yml['max_vel']
        self.EE = device_yml['EE']
        self.ctrlr_dof_xyz = device_yml['ctrlr_dof_xyz']
        self.ctrlr_dof_abg = device_yml['ctrlr_dof_abg']
        self.ctrlr_dof = np.hstack([self.ctrlr_dof_xyz, self.ctrlr_dof_abg])
        self.start_angles = np.array(device_yml['start_angles'])
        self.num_gripper_joints = device_yml['num_gripper_joints']
        
        # Initialize dicts to keep track of the state variables and locks
        self.state_vars = ['q', 'dq', 'dqq', 'ee_xyz', 'ee_quat']
        self.state = dict()
        self.locks = dict()
        for var in self.state_vars:
            self.locks[var] = Lock()
        
        # Check if the user specifies a start body for the while loop to terminte at
        try:
            start_body = model.body_name2id(device_yml['start_body'])
        except:
            start_body = 0
        
        # Get the joint ids, using the specified EE / start body 
        # Reference: ABR Control
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

        # Check that the 
        if np.sum(np.hstack([self.ctrlr_dof_xyz, self.ctrlr_dof_abg])) > len(self.joint_ids):
            print("Fewer DOF than specified")

    def get_jacobian(self, full=False):
        """
            Returns either:
            1) The full jacobian (of the Device, using its EE), if full==True 
            2) The full jacobian evaluated at the controlled DoF, if full==False 
            Depeding on the 'full' parameter's value
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

    def _get_R(self):
        if self.name == "ur5right":
            return self.sim.data.get_site_xmat("ft_frame_ur5right")
        if self.name == "ur5left":
            return self.sim.data.get_site_xmat("ft_frame_ur5left")

    def get_force(self):
        if self.name == "ur5right":
            force = np.matmul(self._get_R(),self.sim.data.sensordata[0:3])
            return force
        if self.name == "ur5left":
            force = np.matmul(self._get_R(),self.sim.data.sensordata[6:9])
            return force
        else:
            return np.zeros(3)
            
    def get_torque(self):
        if self.name == "ur5right":
            force = np.matmul(self._get_R(),self.sim.data.sensordata[3:6])
            return force
        if self.name == "ur5left":
            force = np.matmul(self._get_R(),self.sim.data.sensordata[9:12])
            return force
        else:
            return np.zeros(3)
    
    def get_state(self, state_var: str, joint_ids: list = []):
        """
            Get the state of the device corresponding to the key value (if exists)
        """
        if len(joint_ids) == 0:
            joint_ids = self.joint_ids_all
        if state_var not in self.locks.keys():
            return None
        self.locks[state_var].acquire()
        value = None
        if state_var == 'q':
            value = self.sim.data.qpos[joint_ids]
        elif state_var == "dq":
            value = self.sim.data.qvel[joint_ids]
        elif state_var == "ddq":
            value = self.sim.data.qacc[joint_ids]
        elif state_var == 'ee_xyz':
            value = self.sim.data.get_body_xpos(self.EE)
        elif state_var == 'ee_quat':
            value = self.sim.data.get_body_xquat(self.EE)
        self.state[state_var] = value
        self.locks[state_var].release()
        return value
    
    def update_state(self):
        """
            This should running in a thread: Robot.start()
        """
        for var in self.state_vars:
            self.get_state(var)

    def get_all_joint_ids(self):
        return self.joint_ids_all
    
    def get_actuator_joint_ids(self):
        return self.joint_ids
    
    def get_gripper_joint_ids(self):
        return self.gripper_ids