import matplotlib.pyplot as plt
import numpy as np

# Import the library.
import third_party.googleintprim as intprim

# Set a seed for reproducibility
np.random.seed(213413414)

# Define some parameters used when generating synthetic data.
num_train_trajectories = 100
train_translation_mean = 0.0
train_translation_std = 5.0
train_noise_std = 0.01
train_length_mean = 95
train_length_std = 30

# Generate some synthetic handwriting trajectories.
training_trajectories = intprim.examples.create_2d_handwriting_data(
    num_train_trajectories,
    train_translation_mean,
    train_translation_std,
    train_noise_std,
    train_length_mean,
    train_length_std)

# Plot the results.
plt.figure()
for trajectory in training_trajectories:
    plt.plot(trajectory[0], trajectory[1])
plt.show()



# Define the data axis names.
dof_names = np.array(["X (Agent 1)", "Y (Agent 2)"])

# Decompose the handwriting trajectories to a basis space with 8 uniformly distributed Gaussian functions and a variance of 0.1.
basis_model = intprim.basis.GaussianModel(8, 0.1, dof_names)

# Initialize a BIP instance.
primitive = intprim.BayesianInteractionPrimitive(basis_model)

# Train the model.
for trajectory in training_trajectories:
    primitive.add_demonstration(trajectory)

# Plot the distribution of the trained model.
mean, upper_bound, lower_bound = primitive.get_probability_distribution()
intprim.util.visualization.plot_distribution(dof_names, mean, upper_bound, lower_bound)



# Set an observation noise for the demonstrations.
observation_noise = np.diag([10000.0, train_noise_std ** 2])



# Compute the phase mean and phase velocities from the demonstrations.
phase_velocity_mean, phase_velocity_var = intprim.examples.get_phase_stats(training_trajectories)

# Define a filter to use. Here we use an ensemble Kalman filter
filter = intprim.filter.spatiotemporal.EnsembleKalmanFilter(
    basis_model = basis_model,
    initial_phase_mean = [0.0, phase_velocity_mean],
    initial_phase_var = [1e-4, phase_velocity_var],
    proc_var = 1e-8,
    initial_ensemble = primitive.basis_weights)




num_test_trajectories = 1
test_translation_mean = 5.0
test_translation_std = 1e-5
test_noise_std = 0.01
test_length_mean = 45
test_length_std = 1e-5

# Create test trajectories.
test_trajectories = intprim.examples.create_2d_handwriting_data(num_test_trajectories, test_translation_mean, test_translation_std, test_noise_std, test_length_mean, test_length_std)

# Evaluate the trajectories.
intprim.examples.evaluate_trajectories(primitive, filter, test_trajectories, observation_noise)
