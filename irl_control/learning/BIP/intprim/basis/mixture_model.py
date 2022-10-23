##
#   This module defines the MixtureModel class.
#
#   @author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
import irl_control.learning.BIP.intprim.basis.basis_model as basis_model
import irl_control.learning.BIP.intprim.constants as gip_const
import numpy as np
import scipy.linalg

##
#   The MixtureModel class is an extension of BasisModel to multiple basis spaces for multiple degrees of freedom.
#   This allows a user to define separate models for each degree of freedom. For example, in a 3 DoF model the first 2 DoFs might both belong to a GaussianModel while the 3rd belongs to a SigmoidalModel.
#
class MixtureModel(basis_model.BasisModel):
    ##
    #   Initialization method for the MixtureModel class.
    #
    #   @param basis_models array-like, shape(num_basis_models, ). Contains a list of (non-MixtureModel) basis models that are encapsulated in this model.
    #
    def __init__(self, basis_models):
        self.observed_dof_names = []
        self._degree = 0
        self.num_observed_dof = 0
        self._num_blocks = 0

        # A map of observed DoF index to basis model index
        self.observed_dof_to_model_map = {}
        # A map of observed DoF index to state index range (start, end)
        self.observed_dof_to_degree_map = {}

        for model in basis_models:
            self.observed_dof_names.extend(model.observed_dof_names)
            self.num_observed_dof += model.num_observed_dof
            self._num_blocks += model._num_blocks

        self.basis_models = basis_models

        self.model_degrees = []

        block_index = 0
        for model_index, model in enumerate(basis_models):
            for block in range(model.num_observed_dof):
                # A map of observed DoF index to basis model index
                self.observed_dof_to_model_map[block_index] = model_index
                # A map of observed DoF index to state index range (start, end)
                self.observed_dof_to_degree_map[block_index] = (self._degree, self._degree + model._degree)
                self._degree += model._degree
                self.model_degrees.append(np.ones(model._degree))
                block_index += 1

        self.block_prototype = scipy.linalg.block_diag(*self.model_degrees).T

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

        for model in self.basis_models:
            basis_funcs = model.get_basis_functions(x)

            for block_index in range(model.num_observed_dof):
                out_array[start_row + block_index * model._degree : start_row + (block_index + 1) * model._degree, start_col + block_index :  start_col + block_index + 1] = basis_funcs.reshape(-1, 1)
            start_row += model.num_observed_dof * model._degree
            start_col += model.num_observed_dof

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

        for model in self.basis_models:
            basis_funcs = model.get_basis_function_derivatives(x)

            for block_index in range(model.num_observed_dof):
                out_array[start_row + block_index * model._degree : start_row + (block_index + 1) * model._degree, start_col + block_index :  start_col + block_index + 1] = basis_funcs.reshape(-1, 1)
            start_row += model.num_observed_dof * model._degree
            start_col += model.num_observed_dof

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

        temp_weights = weights

        for model in self.basis_models:
            basis_func_derivs = model.get_basis_function_derivatives(x)

            for block_index in range(model.num_observed_dof):
                out_array[out_row, start_col + block_index :  start_col + block_index + 1] = np.dot(basis_func_derivs.T, temp_weights[start_row + block_index * model._degree : start_row + (block_index + 1) * model._degree])
            start_row += model.num_observed_dof * model._degree
            start_col += model.num_observed_dof

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
        weights = []
        start_degree = 0
        end_degree = 0
        for model in self.basis_models:
            end_degree += model.num_observed_dof

            weights.append(model.fit_basis_functions_linear_closed_form(x, y[:, start_degree : end_degree]))

            start_degree = end_degree

        return np.concatenate(weights)

    ##
    #   Applies the given weights to this basis model. Projects a basis state to the measurement space.
    #
    #   @param x Scalar of vector of dimension T containing the phase values to project at.
    #   @param coefficients Vector of dimension degree * num_observed_dof containing the basis weights.
    #   @param deriv True to use basis function derivative, False to use regular basis functions.
    #
    #   @return Vector of dimension num_observed_dof or matrix of dimension num_observed_dof x T if multiple phase values are given.
    def apply_coefficients(self, x, coefficients, deriv = False):
        result = np.zeros(self._num_blocks, dtype = gip_const.DTYPE)

        start_index_blocks = 0
        start_index_degrees = 0
        for model in self.basis_models:

            result[start_index_blocks : start_index_blocks + model.num_observed_dof] = model.apply_coefficients(x, coefficients[start_index_degrees : start_index_degrees + (model.num_observed_dof * model._degree)], deriv)
            start_index_blocks += model.num_observed_dof
            start_index_degrees += model.num_observed_dof * model._degree

        return result

    # Transforms indices from the observed space to the state space.
    def observed_to_state_indices(self, observed_indices):
        state_indices = []

        try:
            for observed_index in observed_indices:
                state_indices.extend(range(self.observed_dof_to_degree_map[observed_index][0], self.observed_dof_to_degree_map[observed_index][1]))
        except TypeError:
            state_indices.extend(range(self.observed_dof_to_degree_map[observed_indices][0], self.observed_dof_to_degree_map[observed_indices][1]))

        return np.array(state_indices)

    def observed_indices_related(self, observed_indices):
        try:
            block_index = None

            for observed_index in observed_indices:
                if(block_index is None):
                    block_index = self.observed_dof_to_model_map[observed_index]
                elif(self.observed_dof_to_model_map[observed_index] != block_index):
                    return False
        except TypeError:
            return False

        return True
