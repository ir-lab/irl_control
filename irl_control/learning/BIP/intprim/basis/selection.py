import irl_control.learning.BIP.intprim.basis.gaussian_model
import irl_control.learning.BIP.intprim.basis.polynomial_model
import irl_control.learning.BIP.intprim.basis.sigmoidal_model
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import sklearn.metrics

class Selection:
    def __init__(self, dof_names, scaling_groups = None):
        self.demonstration_list = []
        self.error_std = 0.05
        self.addtl_model_penalty = 1.0

        self.max_degree = 14
        self.min_degree = 5
        self.degree_step = 2
        self.max_scale = 0.2
        self.min_scale = 0.01
        self.scale_step = 0.03

        self.num_poly_models = 0
        self.num_gauss_models = 0
        self.num_sig_models = 0

        self.data_degree = len(dof_names)
        self.dof_names = dof_names
        self.scaling_groups = scaling_groups

        self.scalers = []
        self.init_scalers()

    def init_scalers(self):
        if(self.scaling_groups is not None):
            for group in self.scaling_groups:
                self.scalers.append(sklearn.preprocessing.MinMaxScaler())

    # The original intention of this function was to stack all of the demonstrations into one giant data matrix of dimension N x M', where M' is the sum of the number of time steps of all demonstrations.
    # However, this means that a single regression fit will calculate the error from trying to fit all demonstrations at once, which is not what we want.
    # So instead, we will keep demonstrations separate and fit them one at a time and average out the errors that result.
    def add_demonstration(self, data):
        # N x M matrix, N is degrees of freedom, M is number of time steps
        self.demonstration_list.append(data)

        if(self.scaling_groups is not None):
            for group, scaler in zip(self.scaling_groups, self.scalers):
                scaler.partial_fit(data[group, :].reshape(-1, 1))

    def create_models(self, dofs, start_phase, end_phase):
        models = []

        # Add polynomial models
        for degree in np.arange(self.min_degree, self.max_degree, self.degree_step):
            models.append(intprim.basis.polynomial_model.PolynomialModel(degree, self.dof_names[dofs], start_phase, end_phase))
        self.num_poly_models = len(models)

        # Add Gaussian models
        for degree in np.arange(self.min_degree, self.max_degree, self.degree_step):
            for scale in np.arange(self.min_scale, self.max_scale, self.scale_step):
                models.append(intprim.basis.gaussian_model.GaussianModel(degree, scale, self.dof_names[dofs], start_phase, end_phase))
        self.num_gauss_models = len(models) - self.num_poly_models

        # Add Sigmoidal models
        for degree in np.arange(self.min_degree, self.max_degree, self.degree_step):
            for scale in np.arange(self.min_scale, self.max_scale, self.scale_step):
                models.append(intprim.basis.sigmoidal_model.SigmoidalModel(degree, scale, self.dof_names[dofs], start_phase, end_phase))
        self.num_sig_models = len(models) - self.num_gauss_models - self.num_poly_models

        return models

    def get_model_mse(self, model, dofs, sample_start_phase = 0.0, sample_end_phase = 1.0):
        error_list = []
        for demonstration in self.demonstration_list:
            domain = np.linspace(0, 1, demonstration.shape[1])
            sample_indices = np.where(np.logical_and(domain >= sample_start_phase, domain <= sample_end_phase))[0]

            data = np.copy(demonstration)
            if(self.scaling_groups is not None):
                for group, scaler in zip(self.scaling_groups, self.scalers):
                    data[group, :] = scaler.transform(data[group, :].reshape(-1, 1)).reshape(data[group, :].shape)

            weights = model.fit_basis_functions_linear_closed_form(domain[sample_indices], data[np.ix_(dofs, sample_indices)].T)

            predicted = []
            for domain_index in sample_indices:
                predicted.append(model.apply_coefficients(domain[domain_index], weights))

            error_list.append(sklearn.metrics.mean_squared_error(data[np.ix_(dofs, sample_indices)].T, np.array(predicted), multioutput = "raw_values"))

        return np.mean(error_list, axis = 0)

    def get_information_criteria(self, dofs, basis_start_phase = 0.0, basis_end_phase = 1.0, sample_start_phase = 0.0, sample_end_phase = 1.0):
        aic = []
        bic = []

        self.models = self.create_models(dofs, basis_start_phase, basis_end_phase)

        for model in self.models:
            print("Fitting " + model.__class__.__name__ + ". Degree: " + str(model._degree) + ". Scale: " + str(model.scale))

            aic_list = []
            bic_list = []
            for demonstration in self.demonstration_list:
                domain = np.linspace(0, 1, demonstration.shape[1])
                sample_indices = np.where(np.logical_and(domain >= sample_start_phase, domain <= sample_end_phase))[0]

                data = np.copy(demonstration)
                if(self.scaling_groups is not None):
                    for group, scaler in zip(self.scaling_groups, self.scalers):
                        data[group, :] = scaler.transform(data[group, :].reshape(-1, 1)).reshape(data[group, :].shape)

                weights = model.fit_basis_functions_linear_closed_form(domain[sample_indices], data[np.ix_(dofs, sample_indices)].T)

                predicted = []
                for domain_index in sample_indices:
                    predicted.append(model.apply_coefficients(domain[domain_index], weights))

                log_likelihood = self.log_likelihood(data[np.ix_(dofs, sample_indices)].T, np.array(predicted))
                aic_value = 2.0 * (self.addtl_model_penalty * model._degree) - (2.0 * log_likelihood)
                bic_value = np.log(sample_indices.shape[0]) * (self.addtl_model_penalty * model._degree) - (2.0 * log_likelihood)
                aic_list.append(aic_value)
                bic_list.append(bic_value)

            aic.append(np.mean(aic_list))
            bic.append(np.mean(bic_list))

        return aic, bic

    def get_best_model(self, aic, bic):
        # Get "best" model which has the lowest AIC/BIC
        min_aic_idx = np.argmin(aic)
        min_bic_idx = np.argmin(bic)

        print("Best model according to AIC: Value: " + str(aic[min_aic_idx]) + ". Model: " + self.models[min_aic_idx].__class__.__name__ + ". Degree: " + str(self.models[min_aic_idx]._degree) + ". Scale: " + str(self.models[min_aic_idx].scale))
        print("Best model according to BIC: Value: " + str(aic[min_bic_idx]) + ". Model: " + self.models[min_bic_idx].__class__.__name__ + ". Degree: " + str(self.models[min_bic_idx]._degree) + ". Scale: " + str(self.models[min_bic_idx].scale))

        return self.models[min_aic_idx], self.models[min_bic_idx]

    def log_likelihood(self, measured, predicted):
        covariance = scipy.linalg.block_diag(*np.tile(self.error_std, (measured.shape[1], 1)))
        return np.sum(scipy.stats.multivariate_normal.logpdf(y, y_hat, covariance) for y, y_hat in zip(measured, predicted))

    def plot_information_criteria(self, ic, save_name = None):
        aspect_ratio = (self.max_scale - self.min_scale) /  (self.max_degree - self.min_degree)
        fig, ax = plt.subplots(figsize = (6,6))
        fig.suptitle("Polynomial model IC")
        im = ax.imshow(np.array(ic[:self.num_poly_models]).reshape(-1, 1),
            cmap="binary",
            interpolation="none",
            extent = [self.min_scale, self.max_scale, self.max_degree, self.min_degree])
        fig.colorbar(im)
        ax.set_xlabel("Scale")
        ax.set_ylabel("Degree")
        ax.set_aspect(aspect_ratio)
        if(save_name is not None):
            fig.savefig(save_name + "_poly.png")
        else:
            fig.show()

        fig, ax = plt.subplots(figsize = (6,6))
        fig.suptitle("Gaussian model IC")
        im = ax.imshow(np.array(ic[self.num_poly_models:self.num_poly_models + self.num_gauss_models]).reshape(-1, int(np.ceil((self.max_scale - self.min_scale) / self.scale_step))),
            cmap = "binary",
            interpolation = "none",
            extent = [self.min_scale, self.max_scale, self.max_degree, self.min_degree])
        fig.colorbar(im)
        ax.set_xlabel("Scale")
        ax.set_ylabel("Degree")
        ax.set_aspect(aspect_ratio)
        if(save_name is not None):
            fig.savefig(save_name + "_gauss.png")
        else:
            fig.show()

        fig, ax = plt.subplots(figsize = (6,6))
        fig.suptitle("Sigmoidal model IC")
        im = ax.imshow(np.array(ic[self.num_poly_models + self.num_gauss_models:]).reshape(-1, int(np.ceil((self.max_scale - self.min_scale) / self.scale_step))),
            cmap="binary",
            interpolation="none",
            extent = [self.min_scale, self.max_scale, self.max_degree, self.min_degree])
        fig.colorbar(im)
        ax.set_xlabel("Scale")
        ax.set_ylabel("Degree")
        ax.set_aspect(aspect_ratio)
        if(save_name is not None):
            fig.savefig(save_name + "_sig.png")
        else:
            fig.show()
