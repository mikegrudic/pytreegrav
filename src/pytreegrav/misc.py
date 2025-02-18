from numba import njit
import numpy as np
from numpy import zeros, sqrt


@njit(fastmath=True)
def random_rotation(seed):
    """Returns a random rotation matrix reproducibly, given a random seed

    Parameters
    ----------
    seed: int
    Random seed

    Returns
    -------
    rotation_matrix: array_like
    3x3 array of random rotation matrix entries
    """

    rotation_matrix = zeros((3, 3))
    np.random.seed(seed)
    # generate x axis
    costheta = np.random.uniform(-1, 1)
    sintheta = sqrt(max(1 - costheta * costheta, 0))
    phi = 2 * np.pi * np.random.uniform()
    rotation_matrix[0] = sintheta * np.cos(phi), sintheta * np.sin(phi), costheta

    # generate independent y axis and orthogonalize
    costheta = np.random.uniform(-1, 1)
    sintheta = sqrt(max(1 - costheta * costheta, 0))
    phi = 2 * np.pi * np.random.uniform()
    rotation_matrix[1] = sintheta * np.cos(phi), sintheta * np.sin(phi), costheta

    sum = 0
    for k in range(3):  # dot product
        sum += rotation_matrix[0, k] * rotation_matrix[1, k]
    for k in range(3):  # deproject
        rotation_matrix[1, k] -= sum * rotation_matrix[0, k]
    sum = 0
    for k in range(3):  # normalize
        sum += rotation_matrix[1, k] * rotation_matrix[1, k]
    sum = sqrt(sum)
    for k in range(3):
        rotation_matrix[1, k] /= sum

    # now z axis is the cross product
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        rotation_matrix[2, i] = (
            rotation_matrix[0, j] * rotation_matrix[1, k] - rotation_matrix[1, j] * rotation_matrix[0, k]
        )

    return rotation_matrix
