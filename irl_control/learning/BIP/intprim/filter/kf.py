##
#   This module defines a spatial filter based off of the standard Kalman filter.
#
#   @author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
import numpy as np
import scipy.linalg

import irl_control.learning.BIP.intprim.constants as gip_const
import irl_control.learning.BIP.intprim.filter.align as align
import irl_control.learning.BIP.intprim.filter.linear_system as linear_system

##
#   The KalmanFilter class localizes an interaction in space via the extended Kalman filter.
#   Temporal localization is accomplished explicitly with an external time-alignment algorithm such as Dynamic Time Warping.
#   This class is a recursive filter, meaning it maintains state information between successive calls to localize().
#   As a spatial filter, the KalmanFilter's internal state consists of only B dimensions modeling the latent space of the interaction.
#   This class corresponds to a standard Probabilistic Movement Primitive.
#
#   References:\n
#   Campbell, J., Stepputtis, S., & Ben Amor, H. (2019). Probabilistic Multimodal Modeling for Human-Robot Interaction Tasks.\n
#   Campbell, J., & Amor, H. B. (2017). Bayesian interaction primitives: A slam approach to human-robot interaction. In Conference on Robot Learning (pp. 379-387).\n
#
class KalmanFilter(linear_system.LinearSystem):
    ##
    #   The initialization method for the ExtendedKalmanFilter. Responsible for initializing the state.
    #
    #   @param basis_model The basis model corresponding to this state space.
    #   @param mean_basis_weights Vector of dimension D containing the initial state for the basis weights.
    #   @param cov_basis_weights Matrix of dimension D x D containing the initial covariance matrix for the basis weights.
    #   @param align_func The time alignment algorithm to use.
    #   @param iterative_alignment True to start alignment from the currently estimated phase, False to perform alignment over the entire trajectory every time.
    #   @param num_align_samples The number of samples to use when aligning trajectories.
    #   @param cyclical Indicates whether this is a cyclical primitive. If True, the internal phase state will cycle back to 0 once it exceeds 1, allowing for continuous inference of periodic interactions.
    #
    def __init__(self,
                 basis_model,
                 mean_basis_weights,
                 cov_basis_weights,
                 align_func = align.dtw.fastdtw,
                 iterative_alignment = False,
                 num_align_samples = 100,
                 cyclical = False):
        super(KalmanFilter, self).__init__(basis_model)

        self.iterative_alignment = iterative_alignment

        # Initial phase is 0 while the landmarks are the mean basis weights of the demonstrations.
        self.state_mean = np.zeros(self.state_dimension, dtype = gip_const.DTYPE)
        self.state_mean[:] = mean_basis_weights.astype(gip_const.DTYPE)

        # Covariance starts at 0 for phase since we assume trajectories start at the initial point
        # The covariance for the basis weights is the same as computed from demonstrations
        self.state_cov = np.zeros((self.state_dimension, self.state_dimension), dtype = gip_const.DTYPE)

        # Assume discrete white noise model for the phase/phase velocity.
        self.state_cov[:, :] = cov_basis_weights

        self.identity_cov = np.eye(self.state_cov.shape[0], dtype = gip_const.DTYPE)

        self.last_phase_estimate = 0.0
        self.align_func = align_func
        self.num_align_samples = num_align_samples
        self.align_trajectory = np.zeros((self.num_align_samples, self.measurement_dimension), dtype = gip_const.DTYPE)
        self.previous_observations = None

    ##
    #   Perform time alignment on the given observations.
    #   This will call the given time alignment function and attempt to temporally align the observations to the trajectory given by the currently estimated state.
    #   If the iterative_alignment flag is True, then time alignment will only be performed from the currently estimated phase to the end of the trajectory.
    #   While this is more computationally efficient, it causes errors to be accumulated and thus is not always a good choice.
    #   If the iterative_alignment flag is False, then time alignment will always occur from 0 phase. By default, the flag is False.
    #
    #   @param observed_trajectory Matrix of dimension T x D containing observations, where T is the number of timesteps that have been observed since the last call to localize() and D is the dimension of the measurement space.
    #   @param active_dofs Vector of dimension \f$ D_o \f$ containing measurement space indices of the observed degrees of freedom. Note that the measurements will also contain unobserved degrees of freedom, but their values should not be used for inference.
    #
    #   @returns Scalar value containing the estimated phase.
    def align_observations(self, observed_trajectory, active_dofs):
        # Create a sequence from the stored basis weights.
        if(self.iterative_alignment):
            domain = np.linspace(self.last_phase_estimate, 1.0, self.num_align_samples, dtype = gip_const.DTYPE)
        else:
            domain = np.linspace(0.0, 1.0, self.num_align_samples, dtype = gip_const.DTYPE)

        for idx in range(domain.shape[0]):
            basis_matrix = self.basis_model.get_block_diagonal_basis_matrix(domain[idx : idx + 1])
            self.align_trajectory[idx, :] = np.dot(basis_matrix.T, self.state_mean).flatten()

        if(self.iterative_alignment):
            end_idx = self.align_func(observed_trajectory[:, active_dofs], self.align_trajectory[:, active_dofs])
        else:
            if(self.previous_observations is None):
                self.previous_observations = np.array(observed_trajectory, copy = True)
            else:
                self.previous_observations = np.vstack((self.previous_observations, observed_trajectory))
            end_idx = self.align_func(self.previous_observations[:, active_dofs], self.align_trajectory[:, active_dofs])


        if(self.iterative_alignment):
            return self.last_phase_estimate + (float(end_idx) / float(self.num_align_samples))
        else:
            return float(end_idx) / float(self.num_align_samples)

    ##
    #   This method performs localization of space (basis weights).
    #   This is a recursive call, which means it updates the internal state estimate based on the given observations.
    #   For each observation given, two steps are performed recursively:
    #   First, the current state is propagated forward in time in what is known as the prediction step.
    #   $$ \\begin{align}
    #   \\boldsymbol{\\mu}_{t|t-1} &=
    #   \\boldsymbol{G}$
    #   \\boldsymbol{\\mu}_{t-1|t-1},\\
    #   %
    #   \\boldsymbol{\\Sigma}_{t|t-1} &= \\boldsymbol{G} \\boldsymbol{\\Sigma}_{t-1|t-1} \\boldsymbol{G}^{T} +
    #   \\boldsymbol{Q}_t,
    #   \\end{align} $$
    #   In practice, the prediction does nothing for spatial filters because basis weights are time invariant.
    #
    #   Next, we integrate the observations into the current state in the update step.
    #   $$ \\begin{align}
    #   \\boldsymbol{K}_t &= \\boldsymbol{\\Sigma}_{t|t-1} \\boldsymbol{H}_t^{T} (\\boldsymbol{H}_t \\boldsymbol{\\Sigma}_{t|t-1} \\boldsymbol{H}_t^{T} + \\boldsymbol{R}_t)^{-1},\\
    #   %
    #   \\boldsymbol{\\mu}_{t|t} &= \\boldsymbol{\\mu}_{t|t-1} + \\boldsymbol{K}_t(\\boldsymbol{y}_t - h(\\boldsymbol{\\mu}_{t|t-1})),\\
    #   %
    #   \\boldsymbol{\\Sigma}_{t|t} &= (I - \\boldsymbol{K}_t \\boldsymbol{H}_t)\\boldsymbol{\\Sigma}_{t|t-1},
    #   \\end{align} $$
    #
    #   Lastly, the mean and covariance of the state are returned.
    #   At the end of both the prediction and update steps the internal phase value is clipped such that it falls within the range [0, 1].
    #
    #   @param measurement Matrix of dimension T x D containing observations, where T is the number of timesteps that have been observed since the last call to localize() and D is the dimension of the measurement space.
    #   @param measurement_noise Matrix of dimension D x D containing the measurement noise for the given set of measurements.
    #   @param active_dofs Vector of dimension \f$ D_o \f$ containing measurement space indices of the observed degrees of freedom. Note that the measurements will also contain unobserved degrees of freedom, but their values should not be used for inference.
    #   @param return_phase_variance This is not used for this filter but is maintained for API compatibility with the others.
    #
    #   @returns Scalar value containing the inferred phase, Vector of dimension D containing inferred mean, Matrix of dimension D x D.
    def localize(self, measurement, measurement_noise, active_dofs, return_phase_variance = False):
        transition_model = self.get_transition_model()

        ending_phase = self.align_observations(measurement, active_dofs)

        phase_domain = np.linspace(self.last_phase_estimate, ending_phase, num = measurement.shape[0], endpoint = True)

        for measurement_idx in range(measurement.shape[0]):
            # Make forward prediction
            self.state_mean = np.dot(transition_model, self.state_mean)
            self.state_cov = np.dot(transition_model, self.state_cov).dot(transition_model.T) + self.get_process_noise()

            measurement_model = self.get_measurement_model(phase_domain[measurement_idx : measurement_idx + 1])
            predicted_measurement = np.dot(measurement_model, self.state_mean)

            kalman_gain = np.dot(self.state_cov, measurement_model.T)
            kalman_gain = kalman_gain.dot(scipy.linalg.inv(np.dot(measurement_model, self.state_cov).dot(measurement_model.T) + measurement_noise))

            self.state_mean += np.dot(kalman_gain, measurement[measurement_idx] - predicted_measurement)

            self.state_cov = (self.identity_cov - np.dot(kalman_gain, measurement_model)).dot(self.state_cov)

        self.last_phase_estimate = ending_phase

        # Return phase and updated weights and covariance
        return self.last_phase_estimate, self.state_mean, self.state_cov
