##
#   This module defines the PolynomialModel class.
#
#   @author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
import irl_control.learning.BIP.intprim.basis.basis_model as basis_model
import irl_control.learning.BIP.intprim.constants as gip_const
import numpy as np

##
#   The PolynomialModel class implements a basis model consisting of Polynomial basis functions.
#
class PolynomialModel(basis_model.BasisModel):
    ##
    #   Initializer method for PolynomialModel.
    #
    #   @param degree int The number of Sigmoid basis functions which will be uniformly distributed throughout the space.
    #   @param observed_dof_names array-like, shape (num_observed_dof, ). The names of all observed degrees of freedom.
    #   @param start_phase float The starting value from which the basis functions are uniformly distributed. The centers are calculated with linspace(start_phase, end_phase, degree).
    #   @param end_phase float The ending value from which the basis functions are uniformly distributed. The centers are calculated with linspace(start_phase, end_phase, degree).
    #
    def __init__(self, degree, observed_dof_names, start_phase = 0.0, end_phase = 1.01):
        super(PolynomialModel, self).__init__(degree, observed_dof_names)

        self.scale = None

    ##
    #   Gets the basis function evaluations for the given phase value(s). Essentially a vectorized wrapper to call get_basis_values for each phase value given.
    #
    #   @param x array-like, shape(num_phase_values, ) or float. If array, a list of phase values for which to compute basis values. Otherwise, a single scalar phase value.
    #   @param degree int. Degree of this basis model.
    #
    #   @return array-like, shape(degree, num_phase_values) or array-like, shape(degree, ) if x is a scalar. The evaluated Polynomial basis functions for the given phase value.
    #
    def get_basis_functions(self, x, degree = None):
        f = lambda x, degree: np.array([(x - (1 - x))**d for d in range(degree)], dtype = gip_const.DTYPE)

        return f(x, self._degree)

    ##
    #   Gets the evaluations for the derivative of the basis functions for the given phase value(s).
    #   This is necessary for the computation of the Jacobian matrix in the EKF filter.
    #   Unlike get_basis_functions, this function does not (currently) implement a hash map internally and so re-computes values everytime.
    #   Since the basis decompositions are simple linear combinations, the partial derivative of the combination with respect to each weight is simply the partial derivative of a single basis function due to the sum rule.
    #
    #   This is the first order partial derivative with respect to x!
    #   It is used to compute the Jacobian matrix for filtering linear dynamical systems.
    #   Verified using wolfram alpha: d/dx a*(x-(1-x))^0 + b*(x-(1-x))^1 + c*(x-(1-x))^2 and d/dx a*(2x-1)^n
    #
    #   @param x array-like, shape(num_phase_values, ) or float. If array, a list of phase values for which to compute basis derivative values. Otherwise, a single scalar phase value.
    #   @param degree int. Degree of this basis model.
    #
    #   @return values array-like, shape(degree, num_phase_values) or array-like, shape(degree, ) if x is a scalar. The evaluated Polynomial basis function derivatives for the given phase value.
    def get_basis_function_derivatives(self, x, degree = None):
        f = lambda x, degree: np.array([(2*d)*(2*x-1)**(d-1) for d in range(degree)], dtype = gip_const.DTYPE)

        return f(x, self._degree)
