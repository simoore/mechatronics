import control as ct
import dataclasses
import functools
import numpy as np

from qpsolvers import solve_qp
from typing import Iterable


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


def unconstrained_lti(sysd: ct.StateSpace, p: MPCParameters) -> tuple[np.ndarray, np.ndarray]:
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

    def multiply_shape(mat: np.ndarray, factor: int) -> tuple[int, int]:
        return (mat.shape[0] * factor, mat.shape[1] * factor)

    Qbar = np.zeros(multiply_shape(CQC, p.hz))
    Tbar = np.zeros(multiply_shape(QC, p.hz))
    Rbar = np.zeros(multiply_shape(p.R, p.hz))
    Cbar = np.zeros(multiply_shape(sysd.B, p.hz))
    Abar = np.zeros((sysd.A.shape[0] * p.hz, sysd.A.shape[1]))

    def Qbar_slice(i: int) -> tuple[slice, slice]:
        return (slice(CQC.shape[0] * i, CQC.shape[0] * (i + 1)), slice(CQC.shape[1] * i, CQC.shape[1] * (i + 1)))
    
    def Tbar_slice(i: int) -> tuple[slice, slice]:
        return (slice(SC.shape[0] * i, SC.shape[0] * (i + 1)), slice(SC.shape[1] * i, SC.shape[1] * (i + 1)))
    
    def Rbar_slice(i: int) -> tuple[slice, slice]:
        return (slice(p.R.shape[0] * i, p.R.shape[0] * (i + 1)), slice(p.R.shape[1] * i, p.R.shape[1] * (i + 1)))
    
    def Abar_slice(i: int) -> tuple[slice, slice]:
        return (slice(sysd.A.shape[0] * i, sysd.A.shape[0] * (i + 1)) , slice(0, sysd.A.shape[1]))
    
    def Cbar_slice(i: int, j: int) -> tuple[slice, slice]:
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

    return Hbar, FbarT.T, Cbar, Abar


@dataclasses.dataclass
class Constraints:

    uub: np.ndarray
    """Upper bound on the inputs"""

    ulb: np.ndarray
    """Lower bound on the inputs"""

    xub: np.ndarray
    """Upper bound on states. Only for states selected by Cc matrix."""

    xlb: np.ndarray
    """Lower bound on states. Only for states selected by Cc matrix."""

    Cc: np.ndarray
    """This matrix selects the states the constraints are applied to."""


def constrained_lti(sysd: ct.StateSpace, p: MPCParameters, c: Constraints, x0: np.ndarray):
    """
    Here we produce the matrices of the quadratic problem we need to solve to produce and mpc solution. This includes
    the matrices of the cost function and the contraint matrices.

    Parameters
    ----------
    sysd
        The state space system we are controlling with MPC.
    p
        The MPC horizon and weighting matrices.
    c
        The contraints on the system input and state.

    Returns
    -------
    Hbar
        A matrix from the quadratic term of the cost function.
    Fbar
        A matrix from the linear term of the cost function.
    G
        The coefficient matrix from the linear inequality constraints.
    h
        The RHS vector from the linear inequality constraints.
    """
    assert x0.ndim == 2
    assert x0.shape[1] == 1

    # Technically since sysd changes on each iteration, Cbar and Abar should recompute sysd for each iteration. This
    # function assumes sysd is constant.
    Hbar, Fbar, Cbar, Abar = unconstrained_lti(sysd, p)

    # Components due to bounds on control action.
    uubar = np.tile(c.uub, (p.hz, 1))
    ulbar = np.tile(c.ulb, (p.hz, 1))
    G1_half = np.eye(p.hz * sysd.ninputs)
    G1 = np.vstack((G1_half, -G1_half))
    h1 = np.vstack((uubar, -ulbar))

    # Components due to bounds on state
    def Ccbar_slice(i: int) -> tuple[slice, slice]:
        return (slice(c.Cc.shape[0] * i, c.Cc.shape[0] * (i + 1)) , slice(c.Cc.shape[1] * i, c.Cc.shape[1] * (i + 1)))
    
    Ccbar = np.zeros((p.hz * c.Cc.shape[0], p.hz * c.Cc.shape[1]))
    for i in range(0, p.hz):
        Ccbar[Ccbar_slice(i)] = c.Cc

    xubar = np.tile(c.xub, (p.hz, 1))
    xlbar = np.tile(c.xlb, (p.hz, 1))
    G2_half = Ccbar @ Cbar
    h2_offset = Ccbar @ Abar @ x0
    G2 = np.vstack((G2_half, -G2_half))
    h2 = np.vstack((xubar - h2_offset, -xlbar + h2_offset))

    # Combine the two sections together
    G = np.vstack((G1, G2))
    h = np.vstack((h1, h2))

    return Hbar, Fbar, G, h


def solve_constrained_lti(
    Hbar: np.ndarray, 
    Fbar: np.ndarray, 
    G: np.ndarray, 
    h: np.ndarray, 
    x: np.ndarray, 
    rg: np.ndarray
):
    """
    This solves a quadratic program for the constrained optimization problem.

    Parameters
    ----------
    Hbar, Fbar, G, h
        Cost function and constraint matricies calclated by the function constrained_lti(..)
    x
        The current state of the system.
    rg
        The global reference vector.

    Returns
    -------
    ug
        The global input vector.
    """
    try:
        q = Fbar @ np.vstack((x, rg)) 
        ug = solve_qp(Hbar, q, G, h, solver="cvxopt")
        if ug is None:
            return None
        ug = ug[:, np.newaxis]
    except ValueError as e:
        print(e)
        raise e
    return ug