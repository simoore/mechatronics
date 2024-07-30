import numpy as np

def calculate_least_squared_solution(Hmatrix: np.ndarray, Ymatrix: np.ndarray) -> np.ndarray:
    """
    The computes the linear least squares solution for y = Hx + v

    Parameters
    ----------
    Hmatrix 
        This is the model matrix.
    Ymatrix
        The measurement vector.

    Returns
    -------
    Xmatrix
        The solution.
    """
    Xmatrix = np.linalg.inv(Hmatrix.T @ Hmatrix) @ Hmatrix.T @ Ymatrix
    return Xmatrix


def calculate_weighted_least_squared(
        Hmatrix: np.ndarray, 
        Rmatrix: np.ndarray, 
        Ymatrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    This calculates the weighted least squares which places a weighting inversly proportional to the variance of 
    each measurement.

    Parameters
    ----------
    Hmatrix 
        This is the model matrix.
    Ymatrix
        The measurement vector.
    Rmatrix
        The covariance of the measurements. This should be diagonal.

    Returns
    -------
    Xmatrix
        The solution.
    """
    Rinv = np.linalg.inv(Rmatrix)
    Pmatrix = np.linalg.inv(Hmatrix.T @ Rinv @ Hmatrix)
    Xmatrix = Pmatrix @ Hmatrix.T @ Rinv @ Ymatrix
    return Xmatrix, Pmatrix


def recursive_least_squares(
        Xmatrix_prev: np.ndarray, 
        Pmatrix_prev: np.ndarray,
        Hmatrix: np.ndarray, 
        Rmatrix: np.ndarray, 
        Ymatrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    This calculates the recursive least squares.

    Parameters
    ----------
    Hmatrix 
        This is the model matrix that maps the solution to the measurement.
    Ymatrix
        The measurement at a single timestep.
    Rmatrix
        The covariance of the measurement noise.

    Returns
    -------
    Xmatrix
        The solution.
    Pmatrix
        The uncertainty of the solution.
    """
    gain = Pmatrix_prev @ Hmatrix.T @ np.linalg.inv(Hmatrix @ Pmatrix_prev @ Hmatrix.T + Rmatrix)
    Xmatrix = Xmatrix_prev + gain @ (Ymatrix - Hmatrix @ Xmatrix_prev)
    KH = gain @ Hmatrix
    I_KH = np.eye(KH.shape[0]) - KH
    Pmatrix = I_KH @ Pmatrix_prev @ I_KH.T + gain @ Rmatrix @ gain.T
    return Xmatrix, Pmatrix