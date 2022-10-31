import copy
import irl_control.learning.BIP.intprim as intprim
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import numpy.random
import sklearn.metrics

try:
    import IPython.display
except:
    pass

animation_plots = []

def create_2d_handwriting_data(num_trajectories, translation_mean, translation_std, noise_std, length_mean, length_std):
    # A single example of a handwriting trajectory
    xdata = np.array([
        2.52147861,  2.68261873,  2.84009521,  2.99269205,  3.13926385,
        3.27876056,  3.41025573,  3.5329778 ,  3.64634321,  3.74998937,
        3.8438048 ,  3.92795314,  4.00288777,  4.0693539 ,  4.12837543,
        4.18122498,  4.22937664,  4.27444203,  4.31809201,  4.36196737,
        4.40758299,  4.4562309 ,  4.50888808,  4.56613502,  4.62809093,
        4.69437067,  4.76406782,  4.83576665,  4.90758435,  4.97724312,
        5.04216954,  5.099617  ,  5.14680484,  5.18106677,  5.1999997 ,
        5.20160394,  5.18440564,  5.14755368,  5.09088427,  5.01494897,
        4.92100438,  4.8109641 ,  4.68731662,  4.55301474,  4.41134412,
        4.26577973,  4.11983926,  3.97694226,  3.84028296,  3.71272292,
        3.59670796,  3.4942117 ,  3.4067061 ,  3.33515726,  3.28004369,
        3.24139282,  3.21883106,  3.21164261,  3.21883242,  3.23918946,
        3.27134723,  3.31383944,  3.36515007,  3.42375745,  3.48817336,
        3.55697803,  3.62885243,  3.70260907,  3.77722187,  3.85185522,
        3.92589153,  3.99895578,  4.07093474,  4.14198835,  4.21255021,
        4.2833145 ,  4.35520693,  4.42933819,  4.50693958,  4.5892814 ,
        4.67757669,  4.7728736 ,  4.87594169,  4.98715824,  5.10640159,
        5.23295916,  5.36545793,  5.50182437,  5.63928031,  5.7743792 ,
        5.90308534,  6.02089593,  6.12300271,  6.20448725,  6.26054043,
        6.28669463,  6.27905489,  6.23451447,  6.15094057,  6.02731681
    ])
    ydata = np.array([
        2.60877965,  2.76485925,  2.91587601,  3.06074461,  3.19850088,
        3.32832259,  3.44955237,  3.56172269,  3.66458245,  3.75812375,
        3.84260667,  3.9185795 ,  3.98689125,  4.04869382,  4.10543106,
        4.1588132 ,  4.21077576,  4.26342334,  4.31895999,  4.37960871,
        4.44752397,  4.52470161,  4.61289081,  4.71351323,  4.82759375,
        4.95570667,  5.09794052,  5.25388323,  5.42262803,  5.60279957,
        5.79259769,  5.98985598,  6.19211079,  6.39667626,  6.60072087,
        6.80134129,  6.99563046,  7.18073763,  7.35391969,  7.51258424,
        7.6543261 ,  7.77695956,  7.87854902,  7.95744025,  8.0122939 ,
        8.0421214 ,  8.0463223 ,  8.0247204 ,  7.97759496,  7.90570262,
        7.81028529,  7.69306011,  7.55618819,  7.40222104,  7.23402506,
        7.05468668,  6.86740265,  6.67536129,  6.48162182,  6.28899902,
        6.09996034,  5.916542  ,  5.74028898,  5.57222266,  5.41283782,
        5.26212897,  5.11964415,  4.98456294,  4.85579367,  4.73208409,
        4.61213865,  4.49473531,  4.37883468,  4.26367447,  4.14884334,
        4.0343288 ,  3.9205359 ,  3.80827461,  3.69871613,  3.59332021,
        3.49373739,  3.40169213,  3.31885379,  3.24670384,  3.18640788,
        3.13870115,  3.10379544,  3.08131435,  3.07026211,  3.06902906,
        3.07543489,  3.08680804,  3.10009753,  3.11201102,  3.11917145,
        3.1182827 ,  3.10629444,  3.08055594,  3.03894936,  2.97999426
    ])

    new_data = []

    basis_model = intprim.basis.GaussianModel(8, 0.1, ["X", "Y"])

    # From this single example, create noisy demonstrations.
    # Approximate the original data with a basis model so that we can sub/super sample it to create
    # trajectories of different lengths while maintaining the same shape.

    # Add 30 demonstrations which are generated from the writing sample
    for demo in range(num_trajectories):
        # Randomly generate a new length
        demonstration_length = int(np.round(np.random.normal(length_mean, length_std)))

        # Fit the single demonstration to the pre-defined basis model
        domain = np.linspace(0, 1, xdata.shape[0], dtype = intprim.constants.DTYPE)
        weights = basis_model.fit_basis_functions_linear_closed_form(domain, np.array([xdata, ydata]).T).T

        # Resample a new trajectory from the basis model with the desired length
        new_interaction = np.zeros((2, demonstration_length))
        domain = np.linspace(0, 1, demonstration_length, dtype = intprim.constants.DTYPE)
        for idx in range(demonstration_length):
            new_interaction[:, idx] = basis_model.apply_coefficients(domain[idx], weights)

        # Apply a random translation
        new_interaction = (new_interaction.T + np.random.normal(translation_mean, translation_std)).T
        new_interaction = np.random.normal(new_interaction, noise_std)

        new_data.append(new_interaction)

    return new_data

def train_model(primitive, training_trajectories):
    for trajectory in training_trajectories:
        primitive.compute_standardization(trajectory)

    for trajectory in training_trajectories:
        primitive.add_demonstration(trajectory)

    return primitive

def get_phase_stats(training_trajectories):
    phase_velocities = []

    for trajectory in training_trajectories:
        phase_velocities.append(1.0 / trajectory.shape[1])

    return np.mean(phase_velocities), np.var(phase_velocities)

def get_observation_noise(basis_selector, basis_model, training_trajectories, bias):
    for trajectory in training_trajectories:
        basis_selector.add_demonstration(trajectory)

    error = basis_selector.get_model_mse(basis_model, np.array(range(training_trajectories[0].shape[0])), 0.0, 1.0)

    observation_noise = np.diag(error) * bias
    observation_noise[0, 0] = 10000

    return observation_noise

def animate_results(generated_data, observed_data, mean_data):
    fig = plt.figure()

    ax = plt.axes(xlim=(-5, 15), ylim=(-5, 15))

    # plot_lines = [plt.plot([], [])[0] for _ in range(3)]

    plot_lines = [
        plt.plot([], [], "--", color = "#ff6a6a", label = "Generated", linewidth = 2.0)[0],
        plt.plot([], [], color = "#6ba3ff", label = "Observed", linewidth = 2.0)[0],
        plt.plot([], [], color = "#85d87f", label = "Mean")[0]
    ]

    fig.suptitle('Probable trajectory')

    def init():
        plot_lines[0].set_data([], [])
        plot_lines[1].set_data([], [])
        plot_lines[2].set_data(mean_data[0], mean_data[1])

        return plot_lines

    def animate(i):
        plot_lines[0].set_data(generated_data[i][0], generated_data[i][1])
        plot_lines[1].set_data(observed_data[i][0], observed_data[i][1])

        return plot_lines

    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func = init,
                                   frames = len(generated_data), interval = 500, blit = True)

    animation_plots.append(anim)
    plt.legend(loc = "upper left")
    plt.show()

def evaluate_trajectories(primitive, filter, test_trajectories, observation_noise, delay_prob = 0.0, delay_ratio = 0.0):
    for test_trajectory in test_trajectories:
        test_trajectory_partial = np.array(test_trajectory, copy = True)
        test_trajectory_partial[0, :] = 0.0

        new_filter = copy.deepcopy(filter)

        primitive.set_filter(new_filter)

        # all_gen_trajectories = []
        # all_test_trajectories = []
        mean_trajectory = primitive.get_mean_trajectory()

        mean_mse = 0.0
        phase_mae = 0.0
        mse_count = 0
        prev_observed_index = 0
        for observed_index in range(8, test_trajectory.shape[1], 8):
            gen_trajectory, phase, mean, var = primitive.generate_probable_trajectory_recursive(test_trajectory_partial[:, prev_observed_index:observed_index], observation_noise, np.array([1]), num_samples = test_trajectory_partial.shape[1] - observed_index)

            mse = sklearn.metrics.mean_squared_error(test_trajectory[:, observed_index:], gen_trajectory)
            mean_mse += mse
            mse_count += 1

            phase_mae += np.abs((float(observed_index) / test_trajectory.shape[1]) - phase)

            if(delay_prob > 0.0 and np.random.binomial(1, delay_prob) == 1):
                length = int(delay_ratio * test_trajectory.shape[1])
                # Repeat the last observation for delay_ratio times.
                delay_trajectory = np.tile(test_trajectory[:, observed_index - 1], (length, 1)).T

                gen_trajectory, phase, mean, var = primitive.generate_probable_trajectory_recursive(delay_trajectory, observation_noise, np.array([1]), num_samples = test_trajectory_partial.shape[1] - observed_index)
                mse = sklearn.metrics.mean_squared_error(test_trajectory[:, observed_index:], gen_trajectory)
                mean_mse += mse
                mse_count += 1
                phase_mae += np.abs((float(observed_index) / test_trajectory.shape[1]) - phase)

                # Plot the phase/phase velocity PDF for each time step? Want to show it for temporal non-linearity.

            intprim.util.visualization.plot_partial_trajectory(gen_trajectory, test_trajectory[:, :observed_index], mean_trajectory)
            # all_gen_trajectories.append(gen_trajectory)
            # all_test_trajectories.append(test_trajectory[:, :observed_index])

            prev_observed_index = observed_index

        print("Mean DoF MSE: " + str(mean_mse / mse_count) + ". Phase MAE: " + str(phase_mae / mse_count))

        # animate_results(all_gen_trajectories, all_test_trajectories, mean_trajectory)

