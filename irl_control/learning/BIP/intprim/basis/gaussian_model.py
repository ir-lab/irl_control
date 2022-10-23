##
#   This module defines the GaussianModel class.
#
#   @author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
import irl_control.learning.BIP.intprim.basis.basis_model as basis_model
import irl_control.learning.BIP.intprim.constants as gip_const
import numpy as np

##
#   The GaussianModel class implements a basis model consisting of Gaussian radial basis functions.
#
class GaussianModel(basis_model.BasisModel):
    ##
    #   Initializer method for GaussianModel.
    #
    #   @param degree int The number of Gaussian basis functions which will be uniformly distributed throughout the space.
    #   @param scale float The variance of each Gaussian function. Controls the width.
    #   @param observed_dof_names array-like, shape (num_observed_dof, ). The names of all observed degrees of freedom.
    #   @param start_phase float The starting value from which the basis functions are uniformly distributed. The centers are calculated with linspace(start_phase, end_phase, degree).
    #   @param end_phase float The ending value from which the basis functions are uniformly distributed. The centers are calculated with linspace(start_phase, end_phase, degree).
    #
    def __init__(self, degree, scale, observed_dof_names, start_phase = 0.0, end_phase = 1.01):
        super(GaussianModel, self).__init__(degree, observed_dof_names)

        # The variance of each Gaussian function. Controls the width.
        self.scale = scale
        # array-like, shape (degree, ) The center of each basis function uniformly distributed over the range [start_phase, end_phase]
        self.centers = np.linspace(start_phase, end_phase, self._degree, dtype = gip_const.DTYPE)

        # dict A hash map storing previously computed basis values. The key is the requested phase value rounded to rounding_precision, and the value is the corresponding computed basis function values for that phase.
        # The first time a phase value is received, the value is computed and stored in the map. On subsequent calls, the stored value is returned (if the phase is within 4 digits), skipping the computation.
        # This map greatly aids in computation times for basis values, as often computations are requested thousands of times for very similar phase values.
        self.computed_basis_values = {}

        # float The precision with which phase values will be stored in the computed_basis_values hash map. A smaller value indicates a larger granularity, which means fewer computations will be performed at the expense of inaccurate basis function computations.
        # The default value is 10.0 ** 4, which indicates that the phase values are rounded to 4 digits after the decimal.
        self.rounding_precision = 10.0 ** 4

    ##
    #   Computes the basis values for the given phase value. The basis values are simply the evaluation of a Gaussian radial basis function at each center.
    #
    #   @param x float The phase value for which to compute the basis values.
    #
    #   @return array-like, shape(degree, ) The evaluated Gaussian radial basis functions for the given phase value.
    def compute_basis_values(self, x):
        # For computational efficiency...
        return np.exp(-((x - self.centers) ** 2) / (2.0 * self.scale))

    ##
    #   Gets the basis function evaluations for the given phase value. If a phase value has previously been computed, returns the stored version. Otherwise computes it (via compute_basis_values) and stores it.
    #
    #   @param x float The phase value for which to evaluate the basis functions.
    #
    #   @return array-like, shape(degree, ) The evaluated Gaussian radial basis functions for the given phase value.
    def get_basis_values(self, x):
        # Simple, optimized rounding function for pure speed.
        key = int(x * self.rounding_precision) / self.rounding_precision

        if(key in self.computed_basis_values):
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
    #   @return array-like, shape(degree, num_phase_values) or array-like, shape(degree, ) if x is a scalar. The evaluated Gaussian radial basis functions for the given phase value.
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
    #   Verified using wolfram alpha: d/dx a*e^(-(x-c)^2/(2*s)) + b*e^(-(x-d)^2/(2*s))
    #
    #   @param x array-like, shape(num_phase_values, ) or float. If array, a list of phase values for which to compute basis derivative values. Otherwise, a single scalar phase value.
    #   @param degree int. Degree of this basis model.
    #
    #   @return values array-like, shape(degree, num_phase_values) or array-like, shape(degree, ) if x is a scalar. The evaluated Gaussian radial basis function derivatives for the given phase value.
    def get_basis_function_derivatives(self, x, degree = None):
        f = lambda x, c: (np.exp(-(np.array([x - y for y in c], dtype = gip_const.DTYPE) ** 2) / (2.0 * self.scale)) * np.array([x - y for y in c], dtype = intprim.constants.DTYPE)) / -self.scale

        return f(x, self.centers)
