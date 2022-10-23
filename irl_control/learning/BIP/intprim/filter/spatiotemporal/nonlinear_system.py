##
#   This module defines a generic NonLinearSystem class for use in nonlinear spatiotemporal filtering.
#
#   @author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
import irl_control.learning.BIP.intprim.constants as gip_const
import numpy as np

##
#   The NonLinearSystem class provides general methods which are required by all the nonlinear filtering methods (EKF, EnKF, PF).
#   This includes the generation of transition models, process noise models, and measurement models.
#
class NonLinearSystem(object):
    ##
    #   Initialization method for NonLinearSystem. Instantiates internal state matrices for computational efficiency.
    #
    #   @param basis_model The basis model corresponding to this state space.
    #   @param proc_var The process noise of the constant velocity phase system. This is a scalar value corresponding to the variance of a piecewise white noise model.
    #   @param time_delta The amount of time that elapses between time steps. This serves as a scaling factor to the constant velocity phase system. In most cases, this should be set to 1.0.
    #   @param system_order The order N of the constant velocity phase system.
    #
    def __init__(self, basis_model, proc_var, time_delta, system_order):
        if(system_order < 0 or system_order > 2):
            raise ValueError("Invalid system order.")

        self.system_order = system_order
        self.system_size = system_order + 1
        self.basis_model = basis_model
        self.state_dimension = self.system_size + basis_model.block_prototype.shape[0]
        self.measurement_dimension = basis_model.num_observed_dof
        self.proc_var = proc_var
        self.time_delta = time_delta

        # Discrete time white noise model for a constant velocity linear dynamical system.
        self.process_noise = None
        self.transition_model = None
        self.phase_process_noise = self.get_phase_process_noise()
        self.measurement_model_prototype = np.zeros((self.system_size + self.basis_model.block_prototype.shape[0], self.basis_model.num_observed_dof))

    ##
    #   Gets the transition model for spatiotemporal filtering.
    #   Only the entries corresponding to the phase system are non-zero, as the basis weights themselves are time invariant.
    #   This method uses lazy instantiation, so the transition matrix is created in memory upon first access.
    #
    #   @returns Matrix of dimension N+1+B x N+1+B containing the transition model.
    def get_transition_model(self):
        if(self.transition_model is None):
            self.transition_model = np.eye(self.state_dimension)

            if(self.system_order == 1):
                # Constant velocity model
                self.transition_model[0, 1] = self.time_delta
            elif(self.system_order == 2):
                # Constant acceleration model
                self.transition_model[0, 1] = self.time_delta
                self.transition_model[0, 2] = (self.time_delta ** 2.0) / 2.0
                self.transition_model[1, 2] = self.time_delta

        return self.transition_model

    ##
    #   Gets the process noise for spatiotemporal filtering.
    #   Only the entries corresponding to the phase system are non-zero, as the basis weights themselves are time invariant.
    #   This method uses lazy instantiation, so the transition matrix is created in memory upon first access.
    #
    #   @returns Matrix of dimension N+1+B x N+1+B containing the process noise.
    def get_process_noise(self):
        if(self.process_noise is None):
            self.process_noise = np.zeros((self.state_dimension, self.state_dimension))
            self.process_noise[:self.system_size, :self.system_size] = self.get_phase_process_noise()

        return self.process_noise

    ##
    #   Gets the process noise specifically for the constant velocity phase system.
    #   Uses a piecewise white noise model for up to a 2nd order system.
    #
    #   @returns Matrix of dimension N+1 x N+1 containing the phase system process noise.
    def get_phase_process_noise(self):
        # Continuous white noise model for 2nd order system
        # noise_model = np.array([
        #     [self.time_delta**5.0 / 20.0, self.time_delta**4.0 / 8.0, self.time_delta**3.0 / 6.0],
        #     [self.time_delta**4.0 / 8.0, self.time_delta**3.0 / 3.0, self.time_delta**2.0 / 2.0],
        #     [self.time_delta**3.0 / 6.0, self.time_delta**2.0 / 2.0, self.time_delta]
        # ], dtype = intprim.constants.DTYPE) * self.proc_var

        # Piecewise white noise for 2nd order system
        noise_model = np.array([
            [self.time_delta**4.0 / 4.0, self.time_delta**3.0 / 2.0, self.time_delta**2.0 / 2.0],
            [self.time_delta**3.0 / 2.0, self.time_delta**2.0, self.time_delta],
            [self.time_delta**2.0 / 2.0, self.time_delta, 1.0]
        ], dtype = gip_const.DTYPE) * self.proc_var

        return noise_model[:self.system_size, :self.system_size]

    ##
    #   Gets the measurement model for spatiotemporal filtering.
    #   This is the Jacobian associated with the partial differential of \f$ \boldsymbol{H}_t = \frac{\partial h(\boldsymbol{s}_t)}{\partial s_t} \f$.
    #   This is only used by the extended Kalman filter.
    #
    #   The first row is the partial derivative of the basis expansion of each DoF w.r.t. phase. Thus, via the sum rule and constant rule, this is simply the basis expansion of the derivative basis functions.
    #   The second row is the partial dervative of the basis expansions w.r.t. phase velocity. Since phase velocity is not present in the equations, this is 0.
    #   The third row on are the partial derivatives of the basis expansions w.r.t. the weights. Again, via the sum rule and constant rule, this is simply the regular basis functions.
    #
    #   @param x Vector of dimension N+1+B containing the basis state with which to obtain the measurement model.
    #
    #   @returns Matrix of dimension N+1+B x D containing the measurement model.
    def get_measurement_model(self, x):
        # The measurement model is a Jacobian matrix.
        self.basis_model.get_weighted_vector_derivative(x[0], x[self.system_size:], out_array = self.measurement_model_prototype, start_row = 0, start_col = 0)
        self.basis_model.get_block_diagonal_basis_matrix(x[0], out_array = self.measurement_model_prototype, start_row = self.system_size, start_col = 0)

        return self.measurement_model_prototype.T
