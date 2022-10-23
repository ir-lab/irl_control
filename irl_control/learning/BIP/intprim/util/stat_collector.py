import datetime
import numpy as np
import os
import scipy.stats
import sklearn.decomposition
import xml.etree.ElementTree as ET

class StatCollector():
    def __init__(self, bip_instance, generated_indices, observed_indices):
        self.generated_indices = generated_indices
        self.observed_indices = observed_indices

        self.timestep_clock = []
        self.observed_trajectory = []
        self.generated_trajectory = []
        self.ensembles = []
        self.means = []
        self.covariances = []

        self.phase_pdf_domain = [0.0, 1.0]
        self.phase_vel_pdf_domain = [-0.025, 0.025]
        self.gen_dof_pdf_domain = [-3.14, 3.14]
        self.dof_pdf_num_samples = 10

    def get_projected_ensemble(self, bip_instance):
        projected_ensembles = []
        projected_ellipses = []

        if(type(bip_instance.filter).__name__ == "EnsembleKalmanFilter"):
            full_ensemble_set = None

            for idx in range(len(self.ensembles)):
                if(idx == 0):
                    full_ensemble_set = self.ensembles[idx][bip_instance.filter.system_order:, :].T

                full_ensemble_set = np.vstack([full_ensemble_set, self.ensembles[idx][bip_instance.filter.system_order:, :].T])

            pca = sklearn.decomposition.PCA(n_components = 2)
            pca.fit(full_ensemble_set)

            for idx in range(len(self.ensembles)):
                ensemble = pca.transform(self.ensembles[idx][bip_instance.filter.system_order:, :].T)

                projected_ensembles.append(ensemble)

        return projected_ensembles

    def get_dof_pdfs(self, bip_instance, target_phase):
        # Time steps x Generated DoFs
        generated_means = []
        generated_covs = []
        sampled_means = []
        sampled_covs = []

        num_samples = self.dof_pdf_num_samples

        for idx in range(len(self.ensembles)):
            generated_means.append([])
            generated_covs.append([])

            # Generate for the target phase
            if(type(bip_instance.filter).__name__ == "EnsembleKalmanFilter"):
                mean, covariance = bip_instance.filter.get_projected_mean_covariance(target_phase, self.ensembles[idx])
            elif(type(bip_instance.filter).__name__ == "ExtendedKalmanFilter"):
                mean, covariance = bip_instance.filter.get_projected_mean_covariance(target_phase, self.means[idx], self.covariances[idx])

            for generated_idx in self.generated_indices:
                # Don't need to add 2 because the returned projected mean/cov doesn't include phase or phase velocity.
                gen_mean = mean[generated_idx]
                gen_var = covariance[generated_idx, generated_idx]

                generated_means[idx].append(gen_mean)
                generated_covs[idx].append(gen_var)

        return generated_means, generated_covs, sampled_means, sampled_covs

    def get_phase_pdfs(self, bip_instance):
        phase_means = []
        vel_means = []
        phase_covs = []
        vel_covs = []

        for idx in range(len(self.ensembles)):
            if(type(bip_instance.filter).__name__ == "EnsembleKalmanFilter"):
                mean = bip_instance.filter.get_ensemble_mean(self.ensembles[idx])
                cov = bip_instance.filter.get_ensemble_covariance(self.ensembles[idx])
            elif(type(bip_instance.filter).__name__ == "ExtendedKalmanFilter"):
                mean = self.means[idx]
                cov = self.covariances[idx]

            phase_means.append(mean[0])
            phase_covs.append(cov[0, 0])

            if(bip_instance.filter.system_order > 0):
                vel_means.append(mean[1])
                vel_covs.append(cov[1, 1])
            else:
                vel_means.append(0.0)
                vel_covs.append(0.0)

        return phase_means, vel_means, phase_covs, vel_covs

    def get_covariances(self, bip_instance):
        covs = []

        # This covariance will be skewed unless normalized to 1.
        for idx in range(len(self.ensembles)):
            if(type(bip_instance.filter).__name__ == "EnsembleKalmanFilter"):
                cov = bip_instance.filter.get_ensemble_covariance(self.ensembles[idx])
            elif(type(bip_instance.filter).__name__ == "ExtendedKalmanFilter"):
                cov = self.covariances[idx]

            covs.append(cov[bip_instance.filter.system_size:, bip_instance.filter.system_size:])

        return covs

    def collect(self, bip_instance, observed_trajectory, generated_trajectory, timestamp):
        self.timestep_clock.append(timestamp)
        self.observed_trajectory.append(observed_trajectory)
        self.generated_trajectory.append(generated_trajectory)
        if(type(bip_instance.filter).__name__ == "EnsembleKalmanFilter"):
            self.ensembles.append(np.array(bip_instance.filter.ensemble, copy = True))
            self.means.append(None)
            self.covariances.append(None)
        elif(type(bip_instance.filter).__name__ == "ExtendedKalmanFilter"):
            self.ensembles.append(None)
            self.means.append(np.array(bip_instance.filter.get_mean()))
            self.covariances.append(np.array(bip_instance.filter.get_covariance()))


    def export(self, bip_instance, export_dir_path, debug_bag_file, response_length, use_spt, spt_phase):
        ensembles = self.get_projected_ensemble(bip_instance)

        target_phase = None
        if(use_spt and spt_phase != "current"):
            target_phase = spt_phase

        generated_means, generated_covs, sampled_means, sampled_covs = self.get_dof_pdfs(bip_instance, target_phase)

        phase_means, vel_means, phase_covs, vel_covs = self.get_phase_pdfs(bip_instance)

        dof_covs = self.get_covariances(bip_instance)

        file_name = os.path.join(export_dir_path, "stat_collection_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.xml"))

        stats = ET.Element("stats")

        bag_info = ET.SubElement(stats, "bag_name")
        bag_info.text = debug_bag_file

        dof = ET.SubElement(stats, "dof")
        dof_names = ET.SubElement(dof, "dof_names")
        dof_names.text = ",".join(bip_instance.basis_model.observed_dof_names)

        obs_indices = ET.SubElement(dof, "observed_indices")
        obs_indices.text = ",".join([str(int(value)) for value in self.observed_indices])

        gen_indices = ET.SubElement(dof, "generated_indices")
        gen_indices.text = ",".join([str(int(value)) for value in self.generated_indices])

        ensemble_names = ET.SubElement(dof, "ensemble_names")
        ensemble_names.text = "Component 1,Component 2"

        pdf_names = ET.SubElement(dof, "pdf_names")
        pdf_names.text = "Phase,Phase Velocity," + ",".join(np.array(bip_instance.basis_model.observed_dof_names)[self.generated_indices])

        pdf_info = ET.SubElement(dof, "pdf_info")
        for feature_name in pdf_names.text.split(","):
            feature_node = ET.SubElement(pdf_info, "feature")
            feature_node.set("name", feature_name)
            feature_node.set("num_samples", str(self.dof_pdf_num_samples))
            if(feature_name == "Phase"):
                feature_node.set("min_range", str(self.phase_pdf_domain[0]))
                feature_node.set("max_range", str(self.phase_pdf_domain[1]))
            elif(feature_name == "Phase Velocity"):
                feature_node.set("min_range", str(self.phase_vel_pdf_domain[0]))
                feature_node.set("max_range", str(self.phase_vel_pdf_domain[1]))
            else:
                feature_node.set("min_range", str(self.gen_dof_pdf_domain[0]))
                feature_node.set("max_range", str(self.gen_dof_pdf_domain[1]))

        misc = ET.SubElement(dof, "response_length")
        misc.text = str(response_length)

        spt = ET.SubElement(dof, "single_point_trajectory")
        spt.text = str(use_spt)
        spt.set("phase", str(spt_phase))

        state_info = ET.SubElement(dof, "latent_state_indices")

        for obs_feature_idx in np.union1d(self.observed_indices, self.generated_indices):
            obs_state_indices = bip_instance.basis_model.observed_to_state_indices(obs_feature_idx)

            feature_node = ET.SubElement(state_info, "feature")
            feature_node.set("name", bip_instance.basis_model.observed_dof_names[obs_feature_idx])
            feature_node.text = ",".join([str(value) for value in obs_state_indices])


        max_timestep = str(self.timestep_clock[-1])
        misc2 = ET.SubElement(dof, "max_oracle_clock")
        misc2.text = str(max_timestep)

        min_timestep = str(self.timestep_clock[1] - (self.timestep_clock[2] - self.timestep_clock[1]))
        misc3 = ET.SubElement(dof, "min_oracle_clock")
        misc3.text = str(min_timestep)


        timesteps = ET.SubElement(stats, "timesteps")

        for timestep_idx in range(len(self.timestep_clock)):
            timestep_node = ET.SubElement(timesteps, "timestep")
            if(self.timestep_clock[timestep_idx] is None):
                # If the timestep is None, that's because it's the initial timestep which was measured far before hand. So just use the next valid time step.
                timestep_node.set("clock", str(self.timestep_clock[timestep_idx + 1] - (self.timestep_clock[timestep_idx + 2] - self.timestep_clock[timestep_idx + 1])))
            else:
                timestep_node.set("clock", str(self.timestep_clock[timestep_idx]))
            observed_node = ET.SubElement(timestep_node, "observed")

            for feature_idx in range(self.observed_trajectory[timestep_idx].shape[1]):
                feature_node = ET.SubElement(observed_node, "feature")
                feature_node.set("name", bip_instance.basis_model.observed_dof_names[feature_idx])
                feature_node.text = ",".join(["%.3f" % value for value in self.observed_trajectory[timestep_idx][:, feature_idx]])

            generated_node = ET.SubElement(timestep_node, "generated")

            for feature_idx in range(self.generated_trajectory[timestep_idx].shape[1]):
                feature_node = ET.SubElement(generated_node, "feature")
                feature_node.set("name", bip_instance.basis_model.observed_dof_names[feature_idx])
                feature_node.text = ",".join(["%.3f" % value for value in self.generated_trajectory[timestep_idx][:, feature_idx]])

            oracle_node = ET.SubElement(timestep_node, "oracle")


            # Calculate 2D transformation of the ensemble and export it to a node.
            ensemble_node = ET.SubElement(timestep_node, "ensemble")
            ensemble_comp1_node = ET.SubElement(ensemble_node, "feature")
            ensemble_comp1_node.set("name", "Component 1")
            if(len(ensembles) > 0):
                ensemble_comp1_node.text = ",".join(["%.3f" % value for value in ensembles[timestep_idx][:, 0]])
            else:
                ensemble_comp1_node.text = ""
            ensemble_comp2_node = ET.SubElement(ensemble_node, "feature")
            ensemble_comp2_node.set("name", "Component 2")
            if(len(ensembles) > 0):
                ensemble_comp2_node.text = ",".join(["%.3f" % value for value in ensembles[timestep_idx][:, 1]])
            else:
                ensemble_comp2_node.text = ""

            # Calculate the PDF for phase and phase velocity and export it to a node.
            pdf_node = ET.SubElement(timestep_node, "pdf")
            phase_node = ET.SubElement(pdf_node, "feature")
            phase_node.set("name", "Phase")
            phase_node.set("mean", str(phase_means[timestep_idx]))
            phase_node.set("cov", str(phase_covs[timestep_idx]))

            velocity_node = ET.SubElement(pdf_node, "feature")
            velocity_node.set("name", "Phase Velocity")
            velocity_node.set("mean", str(vel_means[timestep_idx]))
            velocity_node.set("cov", str(vel_covs[timestep_idx]))

            for generated_idx in range(len(self.generated_indices)):
                feature_node = ET.SubElement(pdf_node, "feature")
                feature_node.set("name", bip_instance.basis_model.observed_dof_names[self.generated_indices[generated_idx]])
                feature_node.set("mean", str(generated_means[timestep_idx][generated_idx]))
                feature_node.set("cov", str(generated_covs[timestep_idx][generated_idx]))


            covariance_node = ET.SubElement(timestep_node, "covariance")
            for obs_feature_idx in np.union1d(self.observed_indices, self.generated_indices):
                obs_state_indices = bip_instance.basis_model.observed_to_state_indices(obs_feature_idx)

                observed_node = ET.SubElement(covariance_node, "outer_cov")
                observed_node.set("name", bip_instance.basis_model.observed_dof_names[obs_feature_idx])

                for gen_feature_idx in np.union1d(self.observed_indices, self.generated_indices)[obs_feature_idx:]:
                    gen_state_indices = bip_instance.basis_model.observed_to_state_indices(gen_feature_idx)

                    cov_values = dof_covs[timestep_idx][obs_state_indices[0] : obs_state_indices[-1] + 1, gen_state_indices[0] : gen_state_indices[-1] + 1]

                    related = bip_instance.basis_model.observed_indices_related([obs_feature_idx, gen_feature_idx])

                    # Each observed factor is a N x M matrix, where N is the number of DoFs for the observed degree and M is the number of DoFs for the generated degree
                    gen_node = ET.SubElement(observed_node, "inner_cov")
                    gen_node.set("name", bip_instance.basis_model.observed_dof_names[gen_feature_idx])
                    gen_node.set("stride", str(int(cov_values.shape[0])))
                    gen_node.set("same_mode", str(related))
                    gen_node.text = ",".join(["%.4f" % value for value in cov_values.flatten()])

        out_file = open(file_name, "wb")
        out_file.write(ET.tostring(stats))

        print("File " + str(file_name) + " successfully exported.")
