import numpy as np

from scipy.linalg import expm, inv


def zero_order_hold(Amatrix: np.ndarray, Bmatrix: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    This executes the zero order discretization of a continuous state space system.

    Parameters
    ----------
    Amatrix
        The continuous time state matrix.
    Bmatrix
        The continuous time input matrix

    Returns
    -------
    Fmatrix
        The discrete time state matrix.
    Gmatrix
        The discrete time input matrix.
    """
    Fmatrix = expm(Amatrix * dt)
    Gmatrix = inv(Amatrix) @ (Fmatrix - np.eye(Amatrix.shape[0]))  @ Bmatrix
    return Fmatrix, Gmatrix