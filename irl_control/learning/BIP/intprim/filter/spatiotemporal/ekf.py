##
#   This module defines a spatiotemporal filter based off of the extended Kalman filter.
#
#   @author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
import numpy as np
import scipy.linalg

import irl_control.learning.BIP.intprim.constants as gip_const
import irl_control.learning.BIP.intprim.filter.spatiotemporal.nonlinear_system as nonlinear_system

##
#   The ExtendedKalmanFilter class localizes an interaction in time and space via the extended Kalman filter.
#   This class is a recursive filter, meaning it maintains state information between successive calls to localize().
#   As with the other spatiotemporal filters, the ExtendedKalmanFilter's internal state consists of (N+1) dimensions modeling
#   the N-th order phase system plus B dimensions modeling the latent space of the interaction for a total size of N+1+B.
#   This class corresponds to a Bayesian Interaction Primitive.
#
#   References:\n
#   Campbell, J., Stepputtis, S., & Ben Amor, H. (2019). Probabilistic Multimodal Modeling for Human-Robot Interaction Tasks.\n
#   Campbell, J., & Amor, H. B. (2017). Bayesian interaction primitives: A slam approach to human-robot interaction. In Conference on Robot Learning (pp. 379-387).\n
#
class ExtendedKalmanFilter(nonlinear_system.NonLinearSystem):
    ##
    #   The initialization method for the ExtendedKalmanFilter. Responsible for initializing the state.
    #
    #   @param basis_model The basis model corresponding to this state space.
    #   @param initial_phase_mean Vector of dimension N corresponding to the initial mean of the constant velocity phase system. If this is a 0th order system, should contain only [phase_mean]. If this is 1st order, should contain [phase_mean, phase_velocity]. If this is 2nd order, should contain [phase_mean, phase_velocity, phase_acceleration]. Anything above 2nd order is not supported.
    #   @param initial_phase_var Vector of dimension N corresponding to the diagonal of the initial covariance of the constant velocity phase system. If this is a 0th order system, should contain only [phase_var]. If this is 1st order, should contain [phase_var, phase_velocity_var]. If this is 2nd order, should contain [phase_var, phase_velocity_var, phase_acceleration_var]. Anything above 2nd order is not supported.
    #   @param proc_var The process noise of the constant velocity phase system. This is a scalar value corresponding to the variance of a piecewise white noise model.
    #   @param mean_basis_weights Vector of dimension D containing the initial state for the basis weights.
    #   @param cov_basis_weights Matrix of dimension D x D containing the initial covariance matrix for the basis weights.
    #   @param time_delta The amount of time that elapses between time steps. This serves as a scaling factor to the constant velocity phase system. In most cases, this should be set to 1.0.
    #   @param cyclical Indicates whether this is a cyclical primitive. If True, the internal phase state will cycle back to 0 once it exceeds 1, allowing for continuous inference of periodic interactions.
    #
    def __init__(self,
                 basis_model,
                 initial_phase_mean,
                 initial_phase_var,
                 proc_var,
                 mean_basis_weights,
                 cov_basis_weights,
                 time_delta = 1.0,
                 cyclical = False):
        super(ExtendedKalmanFilter, self).__init__(basis_model, proc_var, time_delta, len(initial_phase_mean) - 1)

        self.cyclical = cyclical

        initial_phase_mean = np.array(initial_phase_mean, dtype = gip_const.DTYPE)
        initial_phase_var = np.diag(initial_phase_var).astype(gip_const.DTYPE)

        # Initial phase is 0 while the landmarks are the mean basis weights of the demonstrations.
        self.state_mean = np.zeros(self.state_dimension, dtype = gip_const.DTYPE)
        self.state_mean[:self.system_size] = initial_phase_mean[:self.system_size]
        self.state_mean[self.system_size:] = mean_basis_weights.astype(gip_const.DTYPE)

        # Covariance starts at 0 for phase since we assume trajectories start at the initial point
        # The covariance for the basis weights is the same as computed from demonstrations
        self.state_cov = np.zeros((self.state_dimension, self.state_dimension), dtype = gip_const.DTYPE)

        # Assume discrete white noise model for the phase/phase velocity.
        self.state_cov[:self.system_size, :self.system_size] = initial_phase_var[:self.system_size, :self.system_size]
        self.state_cov[self.system_size:, self.system_size:] = cov_basis_weights

        self.identity_cov = np.eye(self.state_cov.shape[0], dtype = gip_const.DTYPE)

    ##
    #   Gets the mean of the internal state.
    #
    #   @returns Vector of dimension N+1+B containing the state mean.
    #
    def get_mean(self):
        return self.state_mean

    ##
    #   Gets the covariance of the internal state.
    #
    #   @returns Matrix of dimension N+1+B x N+1+B.
    def get_covariance(self):
        return self.state_cov

    ##
    #   Calculates the projection of the internal mean and covariance to measurement space.
    #
    #   @param phase The phase value \f$ \phi \f$ to use for the projection.
    #   @param mean The mean from which to calculate the projected mean. If none, uses the internal state mean.
    #   @param cov The covariance from which to calculate the projected covariance. If none, uses the interal state covariance.
    #
    #   @returns Vector of dimension D containing the sample mean, matrix of dimension D x D containing the sample covariance.
    #
    def get_projected_mean_covariance(self, phase, mean = None, cov = None):
        if(mean is None):
            mean = self.get_mean()
        if(cov is None):
            cov = self.get_covariance()

        if(phase is None):
            phase = mean[0]

        temp_mean = np.array(mean, copy = True)
        temp_mean[0] = phase
        measurement_model = self.get_measurement_model(temp_mean)
        projected_mean = np.dot(measurement_model[:,self.system_size:], temp_mean[self.system_size:])
        projected_cov = np.dot(measurement_model, cov).dot(measurement_model.T)

        return projected_mean, projected_cov

    ##
    #   This method performs simultaneous localization of both time (phase) and space (basis weights).
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
    #   Note that in this case, we only apply the dot product to the first N+1 dimensions of the state. This is for computational efficiency as only the constant velocity phase system has a non-zero transition.
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
    #   @param return_phase_variance True if the mean/variance for the phase system should be returned in addition to the basis weights.
    #
    #   @returns Scalar value containing the inferred phase, Vector of dimension D (or N+1+D if return_phase_variance is True) containing inferred mean, Matrix of dimension D x D (or N+1+D x N+1+D if return_phase_variance is True).
    def localize(self, measurement, measurement_noise, active_dofs, return_phase_variance = False):
        transition_model = self.get_transition_model()

        nonactive_dofs = np.setdiff1d(range(self.measurement_dimension), active_dofs)

        for measurement_idx in range(measurement.shape[0]):
            # Make forward prediction
            self.state_mean = np.dot(transition_model, self.state_mean)
            self.state_cov = np.dot(transition_model, self.state_cov).dot(transition_model.T) + self.get_process_noise()

            # Restrict final output to [0.0, 1.0] so it's a valid phase.
            if(self.state_mean[0] > 1.0):
                self.state_mean[0] = 1.0
            elif(self.state_mean[0] < 0.0):
                self.state_mean[0] = 0.0

            measurement_model = self.get_measurement_model(self.state_mean)
            predicted_measurement = np.dot(measurement_model[:,self.system_size:], self.basis_model.inverse_transform(self.state_mean[self.system_size:]))

            kalman_gain = np.dot(self.state_cov, measurement_model.T)
            kalman_gain = kalman_gain.dot(scipy.linalg.inv(np.dot(measurement_model, self.state_cov).dot(measurement_model.T) + measurement_noise))

            # Zero out the Kalman gain entries for the non-active DoFs. Since we aren't considering them we don't want them to affect the update process.
            kalman_gain[:, nonactive_dofs] = 0.0

            self.state_mean += np.dot(kalman_gain, measurement[measurement_idx] - predicted_measurement)

            self.state_cov = (self.identity_cov - np.dot(kalman_gain, measurement_model)).dot(self.state_cov)

            # If the interaction is cyclical and the mean is greater than 1.0, then set the mean phase to 0. Leave covariance and higher order moments alone.
            if(self.cyclical and self.state_mean[0] >= 1.0):
                self.state_mean[0] -= 1.0

        # Restrict final output to [0.0, 1.0] so it's a valid phase.
        if(self.state_mean[0] > 1.0):
            self.state_mean[0] = 1.0
        elif(self.state_mean[0] < 0.0):
            self.state_mean[0] = 0.0

        # Return phase and updated weights and covariance
        if(return_phase_variance is False):
            return self.state_mean[0], self.state_mean[self.system_size:], self.state_cov[self.system_size:, self.system_size:]
        else:
            return self.state_mean[0], self.state_mean[:], self.state_cov[:, :]
