##
#   This module defines the BasisModel base class, which defines general purpose functions used by all basis models.
#
#   @author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
import irl_control.learning.BIP.intprim.constants
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial
import scipy.linalg
import scipy.optimize
import scipy.sparse
import sklearn.preprocessing

##
#   The BasisModel class is a base level class defining general methods that are used by all implemented basis models.
#   This class corresponds to a shared basis space for 1 or more degrees of freedom.
#   That is, every DoF modeled by this basis space uses the same set of basis functions with the same set of parameters.
#
class BasisModel(object):
    ##
    #   Initialization method for the BasisModel class.
    #
    #   @param degree The degree of this basis model. Corresponds to the number of basis functions distributed throughout the domain.
    #   @param observed_dof_names The names for the observed degrees of freedom that use this basis space. The length is used to compute the number of observed degrees of freedom.
    #
    def __init__(self, degree, observed_dof_names):
        self.observed_dof_names = observed_dof_names
        self._degree = degree
        self.num_observed_dof = len(observed_dof_names)
        self._num_blocks = self.num_observed_dof
        self.block_prototype = scipy.linalg.block_diag(*np.tile(np.ones((degree, 1)), (1, self.num_observed_dof)).T).T

    ##
    #   Gets the block diagonal basis matrix for the given phase value(s).
    #   Used to transform vectors from the basis space to the measurement space.
    #
    #   @param x Scalar of vector of dimension T containing the phase values to use in the creation of the block diagonal matrix.
    #   @param out_array Matrix of dimension greater to or equal than (degree * num_observed_dof * T) x num_observed_dof in which the results are stored. If none, an internal matrix is used.
    #   @param start_row A row offset to apply to results in the block diagonal matrix.
    #   @param start_col A column offset to apply to results in the block diagonal matrix.
    #
    #   @return block_matrix Matrix of dimension greater to or equal than (degree * num_observed_dof * T) x num_observed_dof containing the block diagonal matrix.
    #
    def get_block_diagonal_basis_matrix(self, x, out_array = None, start_row = 0, start_col = 0):
        if(out_array is None):
            out_array = self.block_prototype

        basis_funcs = self.get_basis_functions(x)
        for block_index in range(self._num_blocks):
            out_array[start_row + block_index * self._degree : start_row + (block_index + 1) * self._degree, start_col + block_index : start_col + block_index + 1] = basis_funcs.reshape((-1,1))

        return out_array

    ##
    #   Gets the block diagonal basis derivative matrix for the given phase value(s).
    #   Used to transform vectors from the derivative basis space to the measurement space.
    #
    #   @param x Scalar containing the phase value to use in the creation of the block diagonal matrix.
    #   @param out_array Matrix of dimension greater to or equal than (degree * num_observed_dof) x num_observed_dof in which the results are stored. If none, an internal matrix is used.
    #   @param start_row A row offset to apply to results in the block diagonal matrix.
    #   @param start_col A column offset to apply to results in the block diagonal matrix.
    #
    #   @return block_matrix Matrix of dimension greater to or equal than (degree * num_observed_dof) x num_observed_dof containing the block diagonal matrix.
    #
    def get_block_diagonal_basis_matrix_derivative(self, x, out_array = None, start_row = 0, start_col = 0):
        if(out_array is None):
            out_array = self.block_prototype

        basis_funcs = self.get_basis_function_derivatives(x)
        for block_index in range(self._num_blocks):
            out_array[start_row + block_index * self._degree : start_row + (block_index + 1) * self._degree, start_col + block_index : start_col + block_index + 1] = basis_funcs

        return out_array

    ##
    #   Gets the weighted vector derivatives corresponding to this basis model for the given basis state.
    #
    #   @param x Scalar containing the phase value to use in the creation of the block diagonal matrix.
    #   @param weights Vector of dimension degree * num_observed_dof containing the weights for this basis model.
    #   @param out_array Matrix of dimension greater to or equal than 1 x degree in which the results are stored. If none, an internal matrix is used.
    #   @param start_row A row offset to apply to results in the block diagonal matrix.
    #   @param start_col A column offset to apply to results in the block diagonal matrix.
    #
    #   @return block_matrix Matrix of dimension greater to or equal than (degree * num_observed_dof) x num_observed_dof containing the block diagonal matrix.
    #
    def get_weighted_vector_derivative(self, x, weights, out_array = None, start_row = 0, start_col = 0):
        if(out_array is None):
            out_array = np.zeros((1, self._degree))

        out_row = start_row
        basis_func_derivs = self.get_basis_function_derivatives(x)

        # temp_weights = self.inverse_transform(weights)
        temp_weights = weights

        for degree in range(self._num_blocks):
            offset = degree * self._degree
            out_array[out_row, start_col + degree] = np.dot(basis_func_derivs.T, temp_weights[offset : offset + self._degree])

        return out_array

    ##
    #   Fits the given trajectory to this basis model via least squares.
    #
    #   @param x Vector of dimension T containing the phase values of the trajectory.
    #   @param y Matrix of dimension num_observed_dof x T containing the observations of the trajectory.
    #
    #   @param coefficients Vector of dimension degree * num_observed_dof containing the fitted basis weights.
    #
    def fit_basis_functions_linear_closed_form(self, x, y):
        basis_matrix = self.get_basis_functions(x).T

        # The following two methods are equivalent, but the scipy version is more robust. Both are calculating the OLS solution to Ax = B.
        #return np.linalg.solve(np.dot(basis_matrix.T, basis_matrix), np.dot(basis_matrix.T, y)).T.flatten()
        coefficients = scipy.linalg.lstsq(basis_matrix, y)[0].T.flatten()

        return coefficients

    ##
    #   Applies the given weights to this basis model. Projects a basis state to the measurement space.
    #
    #   @param x Scalar of vector of dimension T containing the phase values to project at.
    #   @param coefficients Vector of dimension degree * num_observed_dof containing the basis weights.
    #   @param deriv True to use basis function derivative, False to use regular basis functions.
    #
    #   @return Vector of dimension num_observed_dof or matrix of dimension num_observed_dof x T if multiple phase values are given.
    def apply_coefficients(self, x, coefficients, deriv = False):
        if(deriv):
            basis_funcs = self.get_basis_function_derivatives(x)
        else:
            basis_funcs = self.get_basis_functions(x)

        coefficients = coefficients.reshape((self._num_blocks, self._degree)).T

        result = np.dot(basis_funcs, coefficients)

        return result

    def plot(self):
        """Plots the unweighted linear basis model.
        """
        test_domain = np.linspace(0, 1, 100, dtype = intprim.constants.DTYPE)
        test_range = self.get_basis_functions(test_domain)

        fig = plt.figure()

        for basis_func in test_range:
            plt.plot(test_domain, basis_func)

        fig.suptitle('Basis Functions')

        plt.show(block = False)

    def plot_derivative(self):
        """Plots the unweighted linear basis model.
        """
        test_domain = np.linspace(0, 1, 100, dtype = intprim.constants.DTYPE)
        test_range = self.get_basis_function_derivatives(test_domain)

        fig = plt.figure()

        for basis_func in test_range:
            plt.plot(test_domain, basis_func)

        fig.suptitle('Basis Function Derivatives')

        plt.show(block = True)

    def plot_weighted(self, coefficients, coefficient_names):
        """Plots the weighted linear basis model.
        """
        test_domain = np.linspace(0, 1, 100, dtype = intprim.constants.DTYPE)
        test_range = self.get_basis_functions(test_domain)

        for coefficients_dimension, name in zip(coefficients, coefficient_names):
            fig = plt.figure()

            for basis_func, coefficient in zip(test_range, coefficients_dimension):
                plt.plot(test_domain, basis_func * coefficient)

            fig.suptitle('Basis Functions For Dimension ' + name)

        plt.show(block = True)

    def observed_to_state_indices(self, observed_indices):
        state_indices = []

        try:
            for observed_index in observed_indices:
                state_indices.extend(range(int(observed_index) * self._degree, (int(observed_index) + 1) * self._degree))
        except TypeError:
            state_indices.extend(range(int(observed_indices) * self._degree, (int(observed_indices) + 1) * self._degree))

        return np.array(state_indices, dtype = int)

    def observed_indices_related(self, observed_indices):
        return True
