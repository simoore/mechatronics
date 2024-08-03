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


class KalmanFilter:
    """
    This implements a linear Kalman Filter. The process model we are considering is,

    x_k = F*x_k-1 + w_k-1
    z_k = H*x_k-1 + v_k-1

    where x is the state, F is the state transition matrix, w is the process noise, z is the measurement, H is the 
    measurement matrix, v is the measurement noise.
    """

    def __init__(
            self,
            Fmatrix: np.ndarray,
            Hmatrix: np.ndarray, 
            Qmatrix: np.ndarray,
            Rmatrix: np.ndarray,
            X0matrix: np.ndarray,
            P0matrix: np.ndarray,
            I0matrix: np.ndarray | None,
        ):
        """
        Sets the parameters and inital state of the filter.

        Parameters
        ----------
        init_on_measurement

        Fmatrix
            The state transition matrix.
        Hmatrix
            The measurement matrix.
        Qmatrix
            The covariance of the process model noise.
        Rmatrix
            The covariance of the measurement noise.
        X0matrix
            The initial state vector. If None, it will initialize on the first measurement.
        P0matrix
            The initial state covariance.
        I0matrix
            Converts the initial measurement to an initial state. If None, the matrix X0matrix is used for the initial
            state.
        """
        self._nstates = X0matrix.shape[0]
        self._noutputs = Hmatrix.shape[0]

        assert Fmatrix.shape == (self._nstates, self._nstates)
        assert Hmatrix.shape == (self._noutputs, self._nstates)
        assert Qmatrix.shape == (self._nstates, self._nstates)
        assert Rmatrix.shape == (self._noutputs, self._noutputs)
        assert X0matrix.shape == (self._nstates, 1)
        assert P0matrix.shape == (self._nstates, self._nstates)

        self._Fmatrix = Fmatrix
        self._Hmatrix = Hmatrix
        self._Qmatrix = Qmatrix
        self._Rmatrix = Rmatrix
        self._I0matrix = I0matrix

        self._state: np.ndarray = X0matrix if I0matrix is None else None
        self._cov: np.ndarray = P0matrix
        self._innovation: np.ndarray = None
        self._innovation_cov: np.ndarray = None


    def state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current estimated state and its uncertainty.
        """
        return self._state, self._cov
    

    def innovation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current innovation and its uncertainty.
        """
        return self._innovation, self._innovation_cov


    def prediction(self):
        """
        Executes the prediction step of the Kalman filter.
        """
        if self._state is not None:
            self._state = self._Fmatrix @ self._state 
            self._cov = self._Fmatrix @ self._cov @ self._Fmatrix.T + self._Qmatrix


    def update(self, measurement: np.ndarray):
        """
        Executes the update step of the Kalman filter when a new measurement is obtained.

        Parameters
        ----------
        measurement
            The new measurement.
        """
        assert measurement.shape == (self._noutputs, 1)

        if self._state is not None:
            self._innovation = measurement - self._Hmatrix @ self._state
            self._innovation_cov = self._Hmatrix @ self._cov @ self._Hmatrix.T + self._Rmatrix
            gain = self._cov @ self._Hmatrix.T @ np.linalg.inv(self._innovation_cov)
            self._state = self._state + gain @ self._innovation
            KH = gain @ self._Hmatrix
            self._cov = (np.eye(KH.shape[0]) - KH) @ self._cov
        else:
            self._state = self._I0matrix @ measurement