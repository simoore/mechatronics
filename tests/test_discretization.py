import numpy as np
from scipy.linalg import expm, inv

from mechatronics.discretization import zero_order_hold

def test_zoh():

    Amatrix = np.array([[1.1, 2.1], [1.0, 0.0]])
    Bmatrix = np.array([[3.1], [0.0]])
    dt = 0.1

    # This is another derivation I found, I want to check they are the same.
    Fmatrix = expm(Amatrix * dt)
    Gmatrix = Fmatrix @ (np.eye(Amatrix.shape[0]) - expm(-Amatrix * dt)) @ inv(Amatrix) @ Bmatrix

    F2, G2 = zero_order_hold(Amatrix, Bmatrix, dt)
    assert np.all(np.isclose(F2, Fmatrix))
    assert np.all(np.isclose(G2, Gmatrix))
