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