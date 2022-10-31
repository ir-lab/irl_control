##
#   This module defines a generic LinearSystem class for use in linear spatial filtering.
#
#   @author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
import numpy as np

##
#   The LinearSystem class provides general methods which are required by the KF linear filter.
#   This includes the generation of transition models, process noise models, and measurement models.
#
class LinearSystem(object):
    ##
    #   Initialization method for LinearSystem. Instantiates internal state matrices for computational efficiency.
    #
    #   @param basis_model The basis model corresponding to this state space.
    #
    def __init__(self, basis_model):
        self.basis_model = basis_model
        self.state_dimension = basis_model.block_prototype.shape[0]
        self.measurement_dimension = basis_model.num_observed_dof

        self.process_noise = None
        self.transition_model = None
        self.measurement_model_prototype = np.zeros((self.basis_model.block_prototype.shape[0], self.basis_model.block_prototype.shape[1]))

    ##
    #   Gets the transition model for spatial filtering.
    #   Functionally this is just an identity matrix since basis weights are time invariant.
    #
    #   @returns Matrix of dimension B x B containing the transition model.
    def get_transition_model(self):
        if(self.transition_model is None):
            self.transition_model = np.eye(self.state_dimension)

        return self.transition_model

    ##
    #   Gets the process noise for spatial filtering.
    #   Functionally this is just a zero matrix since basis weights are time invariant.
    #
    #   @returns Matrix of dimension B x B containing the process noise.
    def get_process_noise(self):
        if(self.process_noise is None):
            self.process_noise = np.zeros((self.state_dimension, self.state_dimension))

        return self.process_noise

    ##
    #   Gets the measurement model for spatial filtering.
    #   This is the Jacobian associated with the partial differential of \f$ \boldsymbol{H}_t = \frac{\partial h(\boldsymbol{s}_t)}{\partial s_t} \f$.
    #   However, the partial derivative of a linear combination of basis weights and basis functions is simply the basis functions.
    #
    #   @returns Matrix of dimension B x D containing the measurement model.
    def get_measurement_model(self, x):
        self.basis_model.get_block_diagonal_basis_matrix(x[:1], out_array = self.measurement_model_prototype, start_row = 0, start_col = 0)

        return self.measurement_model_prototype.T
