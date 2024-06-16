import control as ct
import dataclasses
import functools
import numpy as np

from typing import Iterable, Tuple


def block_diag(ms: Iterable[np.ndarray]) -> np.ndarray:
    """
    Creates a block diagonal matrix from all a set of matrices.

    Parameters
    ----------
    ms 
        A list of matricies to combine in a block diagonal matrix.

    Returns
    -------
    mat
        The resulting block diagonal matrix.
    """
    # Assert that we are actually dealing with maticies.
    for m in ms:
        assert m.ndim == 2

    # Compute the dimensions of the resulting matrix to pre-allocate memory.
    r, c = functools.reduce(lambda acc, e: (acc[0] + e.shape[0], acc[1] + e.shape[1]), ms, (0, 0))
    mat = np.zeros((r, c))
    
    # Copy the matrices into their place in the block diagonal matrix.
    i, j = 0, 0
    for m in ms:
        mat[i:(i + m.shape[0]), j:(j + m.shape[1])] = m
        i, j = i + m.shape[0], j + m.shape[1]
    
    return mat


@dataclasses.dataclass
class MPCParameters:
    
    hz: int
    """The horizon period."""
    
    Q: np.ndarray
    """The weigthing matrix for the error."""
    
    S: np.ndarray
    """The weighting matrix for the final error."""

    R: np.ndarray
    """The weighting matrix for the conrol action."""


def unconstrained_lti(sysd: ct.StateSpace, p: MPCParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is the computes the analytical solution to the MPC optimization problem for unconstrained LTI 
    systems.

    Parameters
    ----------
    sysd
        The discrete time system we are controlling.
    p
        The MPC control parameters.

    Returns
    -------
    Hbar, Fbar
        Two matrices from the matrix form of the quadratic cost function from MPC to compute the control law for
        MPC on an unconstrained LTI system.  
    """
    assert sysd.A.ndim == 2
    assert sysd.B.ndim == 2
    assert sysd.C.ndim == 2

    CQC = sysd.C.T @ p.Q @ sysd.C
    CSC = sysd.C.T @ p.S @ sysd.C
    QC = p.Q @ sysd.C
    SC = p.S @ sysd.C

    def multiply_shape(mat: np.ndarray, factor: int) -> Tuple[int, int]:
        return (mat.shape[0] * factor, mat.shape[1] * factor)

    Qbar = np.zeros(multiply_shape(CQC, p.hz))
    Tbar = np.zeros(multiply_shape(QC, p.hz))
    Rbar = np.zeros(multiply_shape(p.R, p.hz))
    Cbar = np.zeros(multiply_shape(sysd.B, p.hz))
    Abar = np.zeros((sysd.A.shape[0] * p.hz, sysd.A.shape[1]))

    def Qbar_slice(i: int) -> Tuple[slice, slice]:
        return (slice(CQC.shape[0] * i, CQC.shape[0] * (i + 1)), slice(CQC.shape[1] * i, CQC.shape[1] * (i + 1)))
    
    def Tbar_slice(i: int) -> Tuple[slice, slice]:
        return (slice(SC.shape[0] * i, SC.shape[0] * (i + 1)), slice(SC.shape[1] * i, SC.shape[1] * (i + 1)))
    
    def Rbar_slice(i: int) -> Tuple[slice, slice]:
        return (slice(p.R.shape[0] * i, p.R.shape[0] * (i + 1)), slice(p.R.shape[1] * i, p.R.shape[1] * (i + 1)))
    
    def Abar_slice(i: int) -> Tuple[slice, slice]:
        return (slice(sysd.A.shape[0] * i, sysd.A.shape[0] * (i + 1)) , slice(0, sysd.A.shape[1]))
    
    def Cbar_slice(i: int, j: int) -> Tuple[slice, slice]:
        return (slice(sysd.B.shape[0] * i, sysd.B.shape[0] * (i + 1)), 
            slice(sysd.B.shape[1] * j, sysd.B.shape[1] * (j + 1)))

    for i in range(0, p.hz):
        Qbar[Qbar_slice(i)] = CSC if i == p.hz - 1 else CQC
        Tbar[Tbar_slice(i)] = SC if i == p.hz - 1 else QC
        Rbar[Rbar_slice(i)] = p.R

        for j in range(0, p.hz):
            if j <= i:
                Cbar[Cbar_slice(i, j)] = np.linalg.matrix_power(sysd.A, i - j) @ sysd.B

        Abar[Abar_slice(i)] = np.linalg.matrix_power(sysd.A, i + 1)

    Hbar = Cbar.T @ Qbar @ Cbar + Rbar
    FbarT = np.vstack((Abar.T @ Qbar @ Cbar, -Tbar @ Cbar))

    return Hbar, FbarT.T