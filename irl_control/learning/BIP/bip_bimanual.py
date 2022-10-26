from irl_control.learning.BIP.intprim.filter.spatiotemporal import EnsembleKalmanFilter
from irl_control.learning.BIP.intprim.filter import KalmanFilter
from irl_control.learning.BIP.intprim.basis.gaussian_model import GaussianModel
from irl_control.learning.BIP.intprim.basis.polynomial_model import PolynomialModel
from irl_control.learning.BIP.intprim.basis.mixture_model import MixtureModel
from irl_control.learning.BIP.intprim.bayesian_interaction_primitives import BayesianInteractionPrimitive
from irl_control.learning.BIP.intprim.basis.selection import Selection
from irl_control.learning.BIP.intprim.filter.align.dtw import fastdtw
from irl_control.learning.BIP.intprim.util.visualization import plot_distribution

import numpy as np
import glob
import copy
import os

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

SAVE_PATH = "./data/BIP/model.ip"

class IntprimStream():
    def __init__(self, enkf=True):
        self._enkf = enkf
        if enkf:
            print("Interaction Primitives using ENKF")
        else:
            print("Interaction Primitives using ProMP")

        print("Initializing Interaction Primitves")
        trajectories = []
        for fn in glob.glob("./data/BIP/*.csv"):
            data = np.loadtxt(fn, delimiter=',')
            trajectories.append(data[::5,:].T)

        basis_scale = 1.0
        var_scale   = 1.0/basis_scale
        j_nfunc     = np.round(5 * basis_scale).astype(dtype=np.int32)
        j_var       = 0.1 * var_scale
        tcp_nfunc   = np.round(7 * basis_scale).astype(dtype=np.int32)
        tcp_var     = 0.25 * var_scale
        ft_nfunc    = np.round(7 * basis_scale).astype(dtype=np.int32)

        dof_names = ["B0", "B1", "L1", "L2", "L3", "L4", "L5", "L6", "R1", "R2", "R3", "R4", "R5", "R6", 
            "FL_X", "FL_Y", "FL_Z", "TL_X", "TL_Y", "TL_Z", "FR_X", "FR_Y", "FR_Z", "TR_X", "TR_Y", "TR_Z",
            "TCPL_X", "TCPL_Y", "TCPL_Z", "TCPR_X", "TCPR_Y", "TCPR_Z",
            "QL_W", "QL_X", "QL_Y", "QL_Z", "QR_W", "QR_X", "QR_Y", "QR_Z"]
        basis_models = []
        # Joints
        basis_models.append(GaussianModel(j_nfunc, j_var, ["B0", "B1", "L1", "L2", "L3", "L4", "L5", "L6", "R1", "R2", "R3", "R4", "R5", "R6"]))
        # F/T
        basis_models.append(PolynomialModel(ft_nfunc, ["FL_X", "FL_Y", "FL_Z", "TL_X", "TL_Y", "TL_Z", "FR_X", "FR_Y", "FR_Z", "TR_X", "TR_Y", "TR_Z"]))
        # TCP
        basis_models.append(GaussianModel(tcp_nfunc, tcp_var, ["TCPL_X", "TCPL_Y", "TCPL_Z", "TCPR_X", "TCPR_Y", "TCPR_Z", "QL_W", "QL_X", "QL_Y", "QL_Z", "QR_W", "QR_X", "QR_Y", "QR_Z"]))

        # Joint basis model
        self.basis_model = MixtureModel(basis_models)

        selection = Selection(dof_names)
        self.ip_model = BayesianInteractionPrimitive(self.basis_model)

        for trj in trajectories:
            self.ip_model.add_demonstration(trj)
            selection.add_demonstration(trj)

        phase_velocity_mean, phase_velocity_var = self.get_phase_stats(trajectories)
        mean, cov = self.ip_model.get_basis_weight_parameters()

        if self._enkf:
            self.filter = EnsembleKalmanFilter(
                basis_model = self.basis_model,
                initial_phase_mean = [0.00, phase_velocity_mean],
                initial_phase_var = [1e-4, phase_velocity_var],
                proc_var = 1e-3, # This has been 1e-3,
                initial_ensemble = self.ip_model.basis_weights)   
        else:
            self.filter = KalmanFilter(
                basis_model = self.basis_model,
                mean_basis_weights = mean,
                cov_basis_weights = cov,
                align_func = fastdtw,
                iterative_alignment = False)
        self.ip_model.set_filter(copy.deepcopy(self.filter)) 

        noise_diag = selection.get_model_mse(self.basis_model, np.array(range(len(dof_names)), dtype =np.int32))
        noise_diag *= 100
        self.noise = np.diag(noise_diag)

        if os.path.exists(SAVE_PATH):
            print("Loading Interaction Primitives from: " + SAVE_PATH)
            self.ip_model.import_data(SAVE_PATH)
        else:
            print("Saving Interaction Primitives to: " + SAVE_PATH)
            self.ip_model.export_data(SAVE_PATH)

        self.last_phase = 0.0
        self.step = 0

        self._phase_history = []
        self._joint_history = []

        self._phase_step = np.random.uniform(1000,1600)
        print("Phase Step:", self._phase_step)

        self._is_done = False

        # mean, upper_bound, lower_bound = self.ip_model.get_probability_distribution()
        # plot_distribution(dof_names, mean, upper_bound, lower_bound)
        # plt.show()

    def get_phase_stats(self, training_trajectories):
        phase_velocities = []

        for trajectory in training_trajectories:
            phase_velocities.append(1.0 / trajectory.shape[1])

        return np.mean(phase_velocities), np.var(phase_velocities)

    def update_stream(self, feedback):
        # Run the interaction primitives only occasionally
        ret_phs = None
        ret_pred = None
        observation = feedback
        if self.step % 100 == 0:
            # observation = np.concatenate(( # 6 6 3 3 1 1 3 4
            #                 feedback["q"], feedback["dq"], feedback["force"], feedback["torque"],
            #                 [feedback["q_qpos"]], [feedback["q_qvel"]], feedback["tcp_pos"], feedback["tcp_rot"]))
            # observation = np.take(observation, [0,1,2,3,4,5, 12,13,14,15,16,17, 20,21,22])
            observation = observation.reshape((1,-1))

            starting_phase = None
            if not self._enkf:
                starting_phase = self.last_phase

            inferred_trajectory, phase, phase_velocity, mean, var = self.ip_model.generate_probable_trajectory_recursive(
                observation.T, self.noise, np.array(range(observation.shape[1]), dtype = np.int32),
                num_samples=100, phase_lookahead=0.07, return_variance=True, starting_phase=starting_phase, be_fancy=self._enkf)
            if self._enkf:
                self.last_phase = phase
            else:
                self.last_phase = min(self.last_phase + 1.0/self._phase_step, 1.0)

            # self._target_pos = inferred_trajectory.T[0,12:15]

            # logging.info("Predicted phase {:.3f} for pose {}".format(self.last_phase, self._target_pos))
            if self._enkf:
                self._is_done = phase >= 0.93 # 0.985
                self._phase_history.append([phase, var[0,0], var[1,1]])
                self._joint_history.append(inferred_trajectory.T[0,:])

                if self._is_done:
                    fig = plt.figure()
                    self._phase_history = np.asarray(self._phase_history)
                    for i in range(self._phase_history.shape[1]):
                        plt.plot(np.asarray(range(len(self._phase_history))) * 5, self._phase_history[:,i])
                    plt.xlabel("Steps")
                    plt.ylabel("Phase")
                    # plt.show()
                    # fn = os.path.join("/Users/simon/Desktop/Data/motion/picture_sequence.csv")
                    # This is for phase
                    # np.savetxt(fn , self._phase_history, delimiter=',')
                    # This is for motion
                    # np.savetxt(fn , [p.tolist()+j.tolist() for p,j in zip(self._phase_history,self._joint_history)], delimiter=',')
            else:
                self._is_done = self.last_phase >= 0.999
            ret_phs = self.last_phase
            ret_pred = inferred_trajectory.T[0,:] 
        self.step += 1
        return ret_phs, ret_pred

    def is_done(self):
        return self._is_done