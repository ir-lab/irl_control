import numpy as np
import scipy.spatial.distance
import sklearn.metrics.pairwise

def compute_cost(x, y, dist_func = "euclidean"):
    r, c = len(x), len(y)

    distance = np.zeros((r, c))
    distance[0, 1:] = np.inf
    distance[1:, 0] = np.inf

    cost = scipy.spatial.distance.cdist(x, y, "euclidean")

    n_star = np.argmin(cost[:, c - 1])

    return n_star

def fastdtw(observed_trajectory, predicted_trajectory):
    # If a trajectory DOF is all zeros, then we are considering it to be unobserved and will not use it for phase matching

    min_cost_index = compute_cost(predicted_trajectory, observed_trajectory)

    #phase_value = float(min_cost_index) / float(pro_mp.NUM_SAMPLES)

    #return phase_value
    return min_cost_index
