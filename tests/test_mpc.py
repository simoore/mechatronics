import numpy as np

from mechatronics.mpc import block_diag, unconstrained_lti, MPCParameters
from mechatronics.models import BicycleModelParameters, bicycle_model, augmented_bicycle_model


def test_block_diag():

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7], [8]])
    c = np.array([[9, 10, 11]])

    mat = block_diag((a, b, c))

    assert mat.shape == (5, 7)
    assert np.all(a == mat[0:2, 0:3])
    assert np.all(b == mat[2:4, 3:4])
    assert np.all(c == mat[4:5, 4:7])


def test_unconstrained_lti_mpc():

    hz = 3
    Q = np.array([[1, 0], [0, 1]])
    S = np.array([[1, 0], [0, 1]])
    R = np.array([[1]])       
    model_params = BicycleModelParameters(m=1500, Iz=3000, Caf=19000, Car=33000, lf=2, lr=3, Ts=0.02, x_dot=20.0)
    mpc_params = MPCParameters(hz, Q=Q, S=S, R=R)
    sysd = bicycle_model(model_params)
    sysd_aug = augmented_bicycle_model(sysd)

    Hbar, Fbar, Cbar, Abar = unconstrained_lti(sysd_aug, mpc_params)

    assert Hbar.shape == (hz * sysd_aug.ninputs, hz * sysd_aug.ninputs)
    assert Fbar.shape == (hz * sysd_aug.ninputs, sysd_aug.nstates + hz * sysd_aug.noutputs)
    assert Cbar.shape == (hz * sysd_aug.nstates, hz * sysd_aug.ninputs)
    assert Abar.shape == (hz * sysd_aug.nstates, sysd_aug.nstates)
