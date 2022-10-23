#!python
##
#   This module defines the BayesianInteractionPrimitive class, which is the user-facing class for deploying IP/BIP/eBIP.
#
#   @author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
import irl_control.learning.BIP.intprim.constants as gip_const
import numpy as np
import pickle
import sklearn.preprocessing
from irl_control.learning.BIP.intprim.filter.ewma import ewma_vectorized
import matplotlib.pyplot as plt
import copy

##
#   The BayesianInteractionPrimitive class is responsible for training an Interaction Primitive model from demonstrations as well as performing run-time inference.
#   Support for importing and exporting a trained model as well
#
class BayesianInteractionPrimitive(object):
    ##
    #   The initialization method for BayesianInteractionPrimitives.
    #
    #   @param basis_model The basis model corresponding to this state space.
    #   @param scaling_groups If provided, used to indicate which degrees of freedom should be scaled as a group.
    #
    def __init__(self, basis_model, scaling_groups = None):
        self.basis_model = basis_model
        self.scaling_groups = scaling_groups

        self.basis_weights = np.array([], dtype = gip_const.DTYPE)
        self.filter = None
        self.prior_fitted = False

        self.scalers = []
        self.init_scalers()
        self.phase_tracker = []
        self.plot = False
        if self.plot:
            self.fig = plt.figure(figsize = (10, 5))
        self.resample_trj = None
        self.resample_cnt = 0
        self.yet_another_counter = 0
        self.time_inversion = 0.0
        self.resample_factor = 0.0

    ##
    #   Exports the internal state information from this model.
    #   Allows one to export a trained model and import it again without requiring training.
    #
    #   @param file_name The name of the export file.
    #
    def export_data(self, file_name):
        print("Exporting data to: " + str(file_name))

        data_struct = {
            "basis_weights" : self.basis_weights,
            "scaling_groups" : self.scaling_groups,
            "scalers" : self.scalers
        }

        with open(file_name, 'wb') as out_file:
            pickle.dump(data_struct, out_file, pickle.HIGHEST_PROTOCOL)

    ##
    #   Imports the internal state information from an export file.
    #   Allows one to import a trained model without requiring training.
    #
    #   @param file_name The name of the import file.
    #
    def import_data(self, file_name):
        print("Importing data from: " + str(file_name))

        with open(file_name, 'rb') as in_file:
            data_struct = pickle.load(in_file)

        self.basis_weights = np.array(data_struct["basis_weights"])

        try:
            self.scaling_groups = data_struct["scaling_groups"]
            self.scalers = data_struct["scalers"]
        except KeyError:
            print("No scalers found during import!")

    ##
    #   Internal method which initializes data scalers.
    #
    def init_scalers(self):
        if(self.scaling_groups is not None):
            for group in self.scaling_groups:
                self.scalers.append(sklearn.preprocessing.MinMaxScaler())

    ##
    #   Iteratively fits data scalers.
    #   This must be called on all training demonstrations before fitting, if scaling is used.
    #
    #   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
    #
    def compute_standardization(self, trajectory):
        if(len(self.scalers) > 0):
            # N x M matrix, N is degrees of freedom, M is number of time steps
            if(type(trajectory) != np.ndarray):
                raise TypeError("Trajectory must be a numpy array.")

            if(len(trajectory) != self.basis_model.num_observed_dof):
                raise ValueError("Trajectory contains an invalid number of degrees of freedom.")

            if(self.scaling_groups is not None):
                for group, scaler in zip(self.scaling_groups, self.scalers):
                    scaler.partial_fit(trajectory[group, :].reshape(-1, 1))
        else:
            print("Skipping basis standardization...")

    ##
    #   Iteratively adds a demonstration to the model.
    #   The demonstration is decomposed into the latent space and the weights are stored internally.
    #
    #   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
    #
    def add_demonstration(self, trajectory):
        # N x M matrix, N is degrees of freedom, M is number of time steps
        if(type(trajectory) != np.ndarray):
            raise TypeError("Trajectory must be a numpy array.")

        if(len(trajectory) != self.basis_model.num_observed_dof):
            raise ValueError("Trajectory contains an invalid number of degrees of freedom. Got " + str(len(trajectory)) + " but expected " + str(self.basis_model.num_observed_dof))

        demonstration_weights = self.basis_transform(trajectory)

        if(self.basis_weights.shape[0] == 0):
            self.basis_weights = np.hstack([self.basis_weights, demonstration_weights])
        else:
            self.basis_weights = np.vstack([self.basis_weights, demonstration_weights])

    ##
    #   Gets the mean trajectory of all trained demonstrations.
    #
    #   @param num_samples The length of the generated mean trajectory
    #
    #   @return mean_trajectory Matrix of dimension D x num_samples containing the mean trajectory.
    #
    def get_mean_trajectory(self, num_samples = gip_const.DEFAULT_NUM_SAMPLES):
        mean, var = self.get_basis_weight_parameters()

        domain = np.linspace(0, 1, num_samples, dtype = gip_const.DTYPE)

        return self.basis_inverse_transform(domain, mean)

    ##
    #   Gets the approximated trajectory for the given demonstration.
    #   This is obtained by transforming the demonstration to the latent space and then projecting it back to measurement space.
    #
    #   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
    #   @param num_samples The length of the generated approximate trajectory
    #
    #   @return approximate_trajectory Matrix of dimension D x num_samples containing the approximate trajectory.
    #
    def get_approximate_trajectory(self, trajectory, num_samples = gip_const.DEFAULT_NUM_SAMPLES, deriv = False):
        # N x M matrix, N is degrees of freedom, M is number of time steps
        if(type(trajectory) != np.ndarray):
            raise TypeError("Trajectory must be a numpy array.")

        if(len(trajectory) != self.basis_model.num_observed_dof):
            raise ValueError("Trajectory contains an invalid number of degrees of freedom.")

        basis_weights = self.basis_transform(trajectory)

        domain = np.linspace(0, 1, num_samples, dtype = gip_const.DTYPE)

        return self.basis_inverse_transform(domain, basis_weights, deriv)

    ##
    #   Gets the approximated trajectory derivative for the given demonstration.
    #   This is obtained by transforming the demonstration to a latent space composed of the basis function derivatives and then projecting it back to measurement space.
    #
    #   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
    #   @param num_samples The length of the generated approximate trajectory
    #
    #   @return approximate_trajectory Matrix of dimension D x num_samples containing the approximate trajectory.
    #
    def get_approximate_trajectory_derivative(self, trajectory, num_samples = gip_const.DEFAULT_NUM_SAMPLES):
        return get_approximate_trajectory(trajectory, num_samples, deriv = True)

    ##
    #   Gets the probability distribution of the trained demonstrations.
    #
    #   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
    #   @param num_samples The length of the generated distribution.
    #
    #   @return mean Matrix of dimension D x num_samples containing the mean of the distribution for every degree of freedom.
    #   @return upper_bound Matrix of dimension D x num_samples containing the mean + std of the distribution for every degree of freedom.
    #   @return lower_bound Matrix of dimension D x num_samples containing the mean - std of the distribution for every degree of freedom.
    #
    def get_probability_distribution(self, num_samples = gip_const.DEFAULT_NUM_SAMPLES):
        trajectory = np.zeros((self.basis_model.num_observed_dof, num_samples))
        upper_bound = np.zeros((self.basis_model.num_observed_dof, num_samples))
        lower_bound = np.zeros((self.basis_model.num_observed_dof, num_samples))

        domain = np.linspace(0, 1, num_samples, dtype = gip_const.DTYPE)
        for idx in range(num_samples):
            # In rare instances, projecting the covariance matrix can produce negative variance values in the diagonals of the projected matrix.
            # Therefore, we instead project each demonstration and manually calculate the empirical mean/covariance.
            projected_states = []
            for dem_idx in range(self.basis_weights.shape[0]):
                projected_states.append(self.basis_model.apply_coefficients(domain[idx], self.basis_weights[dem_idx, :]))
            projected_states = np.array(projected_states)

            dist_mean = np.mean(projected_states, axis = 0)
            dist_var = np.cov(projected_states.T)

            if(self.scaling_groups is not None):
                var_scale = np.ones(dist_mean.shape)
                for group, scaler in zip(self.scaling_groups, self.scalers):
                    var_scale[group] = 1.0 / scaler.scale_
                    dist_mean[group] = scaler.inverse_transform(dist_mean[group].reshape(-1, 1)).flatten()

                var_scale = np.diag(var_scale)
                dist_var = np.dot(var_scale, dist_var).dot(var_scale.T)

            trajectory[:, idx] = dist_mean

            for dof_index in range(0, self.basis_model.num_observed_dof):
                std_dev = dist_var[dof_index][dof_index] ** 0.5

                upper_bound[dof_index, idx] = dist_mean[dof_index] + std_dev
                lower_bound[dof_index, idx] = dist_mean[dof_index] - std_dev

        return trajectory, upper_bound, lower_bound

    ##
    #   Transforms the given trajectory from measurement space into the latent basis space.
    #
    #   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
    #
    #   @return transformed_state Vector of dimension B containing the transformed trajectory.
    #
    def basis_transform(self, trajectory):
        if(self.scaling_groups is not None):
            scaled_trajectory = np.zeros(trajectory.shape)

            for group, scaler in zip(self.scaling_groups, self.scalers):
                scaled_trajectory[group, :] = scaler.transform(trajectory[group, :].reshape(-1, 1)).reshape(trajectory[group, :].shape)

            trajectory = scaled_trajectory

        domain = np.linspace(0, 1, len(trajectory[0]), dtype = gip_const.DTYPE)
        return self.basis_model.fit_basis_functions_linear_closed_form(domain, trajectory.T)

    ##
    #   Transforms the given basis space weights to measurement space for the given phase values.
    #
    #   @param x Vector of dimension T containing the phase values that the basis space weights should be projected at.
    #   @param weights Vector of dimension B containing the basis space weights.
    #   @param deriv True if the basis weights should be transformed with basis function derivatives, False for normal basis functions.
    #
    #   @return transformed_trajectory Matrix of dimension D x T containing the transformed trajectory.
    #
    def basis_inverse_transform(self, x, weights, deriv = False):
        trajectory = np.zeros((self.basis_model.num_observed_dof, x.shape[0]), dtype = gip_const.DTYPE)

        for idx in range(x.shape[0]):
            trajectory[:, idx] = self.basis_model.apply_coefficients(x[idx], weights, deriv)

        if(self.scaling_groups is not None):
            for group, scaler in zip(self.scaling_groups, self.scalers):
                trajectory[group, :] = scaler.inverse_transform(trajectory[group, :].reshape(-1, 1)).reshape(trajectory[group, :].shape)

        return trajectory

    ##
    #   Gets the mean and covariance for the trained demonstrations.
    #
    #   @return mean Vector of dimension B containing the sample mean of the trained basis weights.
    #   @return var Matrix of dimension B x B containing the sample covariance of the trained basis weights.
    #
    def get_basis_weight_parameters(self):
        mean = np.mean(self.basis_weights, axis = 0)

        if(self.basis_weights.shape[0] > 1):
            var = np.cov(self.basis_weights, rowvar = False)
        else:
            var = None

        return mean, var

    ##
    #   Performs inference over the given trajectory and returns the most probable future trajectory.
    #   This is a recursive call and the internal state of the currently set filter will be updated, so only new observations should be passed to this method each time it is called.
    #
    #   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
    #   @param observation_noise Matrix of dimension D x D containing the observation noise.
    #   @param active_dofs Vector of dimension \f$ D_o \f$ containing measurement space indices of the observed degrees of freedom. Note that the measurements will also contain unobserved degrees of freedom, but their values should not be used for inference.
    #   @param num_samples The length of the generated trajectory. If set to 1, only a single value at the current or specified phase (see starting_phase) is generated.
    #   @param starting_phase The phase value to start generating the trajectory from. If none, will start generating from the currently estimated phase value.
    #   @param return_variance True if the phase system mean and variance should be returned. False otherwise.
    #   @param phase_lookahead If phase_lookahead is none, this is an offset that will be applied to the currently estimated phase used to generate the trajectory.
    #
    #   @return new_trajectory Matrix of dimension D x num_samples containing the inferred trajectory.
    #   @return phase Scalar value containing the inferred phase.
    #   @return mean Vector of dimension B (or N+1+B if return_variance is True) containing the mean of the inferred state.
    #   @return var Matrix of dimension B x B (or N+1+B x N+1+B if return_variance is True) containing the covariance of the inferred state.
    #
    def generate_probable_trajectory_recursive(self, trajectory, observation_noise, active_dofs, num_samples = gip_const.DEFAULT_NUM_SAMPLES, starting_phase = None, return_variance = False, phase_lookahead = 0.0, be_fancy=True):
        if(self.scaling_groups is not None):
            scaled_trajectory = np.zeros(trajectory.shape)

            for group, scaler in zip(self.scaling_groups, self.scalers):
                scaled_trajectory[group, :] = scaler.transform(trajectory[group, :].reshape(-1, 1)).reshape(trajectory[group, :].shape)

            trajectory = scaled_trajectory

        phase, mean, var = self.filter.localize(trajectory.T, observation_noise, active_dofs, return_phase_variance = return_variance)
        self.phase_tracker.append(abs(mean[0]))

        # Grab 5 values before even trying to correct (better for filter)
        max_factor = 1e5
        max_time_factor = -0.1
        stuck_cutoff = 0.00005
        phase_velocity = self.phase_tracker[-1]
        # factor = 0.0
        resample_n = 200
        self.time_inversion *= 0.995
        self.resample_factor *= 0.975
        if self.time_inversion < 0.01 and False:
            phase_velocity = self.get_velocity()
            self.time_inversion = 0
            # When do we considder it stuck?
            if phase_velocity < stuck_cutoff:
                self.time_inversion = 0.2
                self.resample_factor = 0
                logging.info("Resampling Trajectory at phase {:.3f}".format(phase))

        target_phase = starting_phase
        if(starting_phase is None):
            target_phase = phase + phase_lookahead - self.time_inversion
            if(target_phase > 1.0):
                target_phase = 1.0
            if(target_phase < 0.0):
                target_phase = 0.0

        domain = np.linspace(target_phase, 1.0, num_samples, dtype = gip_const.DTYPE)
        if be_fancy:
            new_mean = np.random.multivariate_normal(mean[self.filter.system_size:], var[self.filter.system_size:, self.filter.system_size:] * self.resample_factor)
            new_trajectory = self.basis_inverse_transform(domain, new_mean)

            # min_force = 1000
            # if self.resample_cnt > 0:
            #     tmp_trajectory = np.copy(new_trajectory)        
            #     for _ in range(3):
            #         projected_force = np.mean(np.take(tmp_trajectory, [-2,-3, -5, -6], axis=1))
            #         if projected_force < min_force:
            #             min_force = projected_force
            #             new_trajectory = np.copy(tmp_trajectory)
            #         new_mean = np.random.multivariate_normal(mean[self.filter.system_size:], var[self.filter.system_size:, self.filter.system_size:] * self.resample_factor)
            #         tmp_trajectory = self.basis_inverse_transform(domain, new_mean)
        else:
            new_trajectory = self.basis_inverse_transform(domain, mean)
     
        if self.plot:
            axis = ['Phase', 'Velocity', 'Variance', 'Resampling']
            values = [phase, phase_velocity * 1.0/0.05, var[1,1] * 1.0/0.0002, self.resample_factor * 1.0/max_factor]
            plt.cla()
            plt.bar(axis, values, color ='maroon', width = 0.4)
            plt.ylim([0, 1])
            plt.xlabel("Observed Metric")
            plt.ylabel("Normalized Values")
            plt.title("Interaction Primitives Live Tracker")
            plt.pause(0.0001)

        return new_trajectory, phase, self.get_velocity(), mean, var

    ##
    #   Sets the given filter as the current filter.
    #   This is used to initialize and reset the filter between interactions.
    #
    #   @param filter The filter to set.
    #
    def set_filter(self, filter):
        self.filter = filter
        self.phase_tracker = []
        self.resample_cnt = 0
        self.resample_trj = None
        self.yet_another_counter = 0
        self.time_inversion = 0.0

    def get_velocity(self):
        if len(self.phase_tracker) < 100:
            return 1.0       
        return np.abs(np.mean(np.diff(self.phase_tracker[-100:])))
        return np.abs(ewma_vectorized(np.diff(self.phase_tracker[5:]), alpha=0.2)[-1])