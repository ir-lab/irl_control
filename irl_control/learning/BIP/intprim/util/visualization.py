import matplotlib.pyplot as plt
import numpy as np

# Displays the probability that the current trajectory matches the stored trajectores at every instant in time.
def plot_distribution(dof_names, mean, upper_bound, lower_bound):
    """Plots a given probability distribution.
    """
    figures_per_plot = np.min([4, mean.shape[0]])

    for index in range(mean.shape[0]):
        if(index % figures_per_plot == 0):
            fig = plt.figure()

        new_plot = plt.subplot(figures_per_plot, 1, (index % figures_per_plot) + 1)
        domain = np.linspace(0, 1, mean.shape[1])

        new_plot.fill_between(domain, upper_bound[index], lower_bound[index], color = '#ccf5ff')
        new_plot.plot(domain, mean[index], color = '#000000')
        new_plot.set_title('Trajectory distribution for degree ' + dof_names[index])

        fig.tight_layout()

    plt.show(block = False)

def plot_trajectory(dof_names, trajectory, observed_trajectory, mean_trajectory = None):
    """Plots a given trajectory.
    """
    fig = plt.figure()

    plt.plot(trajectory[0], trajectory[1])
    plt.plot(observed_trajectory[0], observed_trajectory[1])
    if(mean_trajectory is not None):
        plt.plot(mean_trajectory[0], mean_trajectory[1])

    fig.suptitle('Probable trajectory')

    fig = plt.figure()
    for index, degree in enumerate(trajectory):
        new_plot = plt.subplot(len(trajectory), 1, index + 1)

        domain = np.linspace(0, 1, len(trajectory[index]))
        new_plot.plot(domain, trajectory[index], label = "Inferred")

        domain = np.linspace(0, 1, len(observed_trajectory[index]))
        new_plot.plot(domain, observed_trajectory[index], label = "Observed")

        if(mean_trajectory is not None):
            domain = np.linspace(0, 1, len(mean_trajectory[index]))
            new_plot.plot(domain, mean_trajectory[index], label = "Mean")

        new_plot.set_title('Trajectory for degree ' + dof_names[index])
        new_plot.legend()

    plt.show()

def plot_partial_trajectory(trajectory, partial_observed_trajectory, mean_trajectory = None):
    """Plots a trajectory and a partially observed trajectory.
    """
    fig = plt.figure()

    plt.plot(partial_observed_trajectory[0], partial_observed_trajectory[1], color = "#6ba3ff", label = "Observed", linewidth = 3.0)
    plt.plot(trajectory[0], trajectory[1], "--", color = "#ff6a6a", label = "Inferred", linewidth = 2.0)
    if(mean_trajectory is not None):
        plt.plot(mean_trajectory[0], mean_trajectory[1], color = "#85d87f", label = "Mean")

    fig.suptitle('Probable trajectory')
    plt.legend()

    plt.text(0.01, 0.7, "Observed samples: " + str(partial_observed_trajectory.shape[1]), transform = fig.axes[0].transAxes)

    plt.show()

def plot_approximation(dof_names, trajectory, approx_trajectory, approx_trajectory_deriv):
    """Plots a trajectory and its approximation.
    """
    domain = np.linspace(0, 1, len(trajectory[0]))
    approx_domain = np.linspace(0, 1, len(approx_trajectory[0]))

    for dof in range(len(trajectory)):
        plt.figure()
        new_plot = plt.subplot(3, 1, 1)
        new_plot.plot(domain, trajectory[dof])
        new_plot.set_title('Original ' + dof_names[dof] + ' Data')

        new_plot = plt.subplot(3, 1, 2)
        # The trailing [0] is the dimension of the the state. In this case only plot position.
        new_plot.plot(approx_domain, approx_trajectory[dof])
        new_plot.set_title('Approximated ' + dof_names[dof] + ' Data')

        new_plot = plt.subplot(3, 1, 3)
        # The trailing [0] is the dimension of the the state. In this case only plot position.
        new_plot.plot(approx_domain, approx_trajectory_deriv[dof])
        new_plot.set_title('Approximated ' + dof_names[dof] + ' Derivative')

    plt.show()

def plot_weights(weight_matrix):
    plt.figure()
    plt.imshow(weight_matrix, cmap = "gray", interpolation = "none")
    plt.colorbar()
    plt.show()
