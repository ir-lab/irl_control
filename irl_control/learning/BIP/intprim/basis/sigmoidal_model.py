##
#   This module defines the SigmoidalModel class.
#
#   @author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
import irl_control.learning.BIP.intprim.basis.basis_model as basis_model
import irl_control.learning.BIP.intprim.constants as gip_const
import numpy as np

##
#   The SigmoidalModel class implements a basis model consisting of Sigmoid radial basis functions.
#
class SigmoidalModel(basis_model.BasisModel):
    ##
    #   Initializer method for SigmoidalModel.
    #
    #   @param degree int The number of Sigmoid basis functions which will be uniformly distributed throughout the space.
    #   @param scale float Controls the steepness of the curve.
    #   @param observed_dof_names array-like, shape (num_observed_dof, ). The names of all observed degrees of freedom.
    #   @param start_phase float The starting value from which the basis functions are uniformly distributed. The centers are calculated with linspace(start_phase, end_phase, degree).
    #   @param end_phase float The ending value from which the basis functions are uniformly distributed. The centers are calculated with linspace(start_phase, end_phase, degree).
    #
    def __init__(self, degree, scale, observed_dof_names, start_phase = 0.0, end_phase = 1.01):
        super(SigmoidalModel, self).__init__(degree, observed_dof_names)

        self.scale = scale
        self.centers = np.linspace(start_phase, end_phase, self._degree, dtype = gip_const.DTYPE)

        self.computed_basis_values = {}

        self.rounding_precision = 10.0 ** 4

    ##
    #   Computes the logistic sigmoid.
    #
    #   @param a float. The value to compute.
    #
    #   @returns result float. The result.
    #
    def log_sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))

    ##
    #   Computes the basis values for the given phase value. The basis values are simply the evaluation of a Sigmoid radial basis function at each center.
    #
    #   @param x float The phase value for which to compute the basis values.
    #
    #   @return array-like, shape(degree, ) The evaluated Sigmoid radial basis functions for the given phase value.
    def compute_basis_values(self, x):
        # For computational efficiency...

        return self.log_sigmoid((x - self.centers) / self.scale)

    ##
    #   Gets the basis function evaluations for the given phase value. If a phase value has previously been computed, returns the stored version. Otherwise computes it (via compute_basis_values) and stores it.
    #
    #   @param x float The phase value for which to evaluate the basis functions.
    #
    #   @return array-like, shape(degree, ) The evaluated Sigmoid radial basis functions for the given phase value.
    def get_basis_values(self, x):
        # key = round(x, 4)
        # Simple, optimized rounding function for pure speed.
        key = int(x * self.rounding_precision) / self.rounding_precision

        if(key in self.computed_basis_values):
            # self.num_successful_hashes += 1
            return self.computed_basis_values[key]
        else:
            values = self.compute_basis_values(x)
            self.computed_basis_values[key] = values
            return values

    ##
    #   Gets the basis function evaluations for the given phase value(s). Essentially a vectorized wrapper to call get_basis_values for each phase value given.
    #
    #   @param x array-like, shape(num_phase_values, ) or float. If array, a list of phase values for which to compute basis values. Otherwise, a single scalar phase value.
    #   @param degree int. Degree of this basis model.
    #
    #   @return array-like, shape(degree, num_phase_values) or array-like, shape(degree, ) if x is a scalar. The evaluated Sigmoid radial basis functions for the given phase value.
    #
    def get_basis_functions(self, x, degree = None):
        if(type(x) is np.ndarray):
            values = np.zeros((self.centers.shape[0], x.shape[0]))

            for value_idx in range(x.shape[0]):
                values[:, value_idx] = self.get_basis_values(x[value_idx])

            return values
        else:
            return self.get_basis_values(x)

    ##
    #   Gets the evaluations for the derivative of the basis functions for the given phase value(s).
    #   This is necessary for the computation of the Jacobian matrix in the EKF filter.
    #   Unlike get_basis_functions, this function does not (currently) implement a hash map internally and so re-computes values everytime.
    #   Since the basis decompositions are simple linear combinations, the partial derivative of the combination with respect to each weight is simply the partial derivative of a single basis function due to the sum rule.
    #
    #   This is the first order partial derivative with respect to x!
    #   It is used to compute the Jacobian matrix for filtering linear dynamical systems.
    #   Verified using wolfram alpha: d/dx a*(1 / (1 + e^(-(x - c)/s))) + b*(1 / (1 + e^(-(x - d)/s)))
    #
    #   @param x array-like, shape(num_phase_values, ) or float. If array, a list of phase values for which to compute basis derivative values. Otherwise, a single scalar phase value.
    #   @param degree int. Degree of this basis model.
    #
    #   @return values array-like, shape(degree, num_phase_values) or array-like, shape(degree, ) if x is a scalar. The evaluated Gaussian radial basis function derivatives for the given phase value.
    def get_basis_function_derivatives(self, x, degree = None):
        f = lambda x, c: np.exp(np.array([y - x for y in c], dtype = gip_const.DTYPE) / self.scale) / (self.scale * (np.exp(np.array([y - x for y in c], dtype = intprim.constants.DTYPE) / self.scale) + 1) ** 2)

        return f(x, self.centers)
