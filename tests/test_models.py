import numpy as np

from mechatronics.models import car_model, CarModelParameters


def test_car_model():

    model_params = CarModelParameters(m=1500, Iz=3000, Caf=38000, Car=66000, lf=2, lr=3, Ts=0.02, mju=0.02, g=9.81)
    x = np.array([[11.99389277, 0.1234, -0.00102285, 0.00542, 10.0, 80.0]]).T
    u = np.array([[0.23, 0.9]]).T

    Ad = np.array([
        [9.99672833e-01,  9.63061903e-03,  0.00000000e+00,  2.17292381e-02, 0.00000000e+00,  0.00000000e+00],
        [0.00000000e+00,  8.85498034e-01,  0.00000000e+00, -1.02028417e-01, 0.00000000e+00,  0.00000000e+00],
        [0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  2.00000000e-02, 0.00000000e+00,  0.00000000e+00],
        [0.00000000e+00,  6.89247194e-02,  0.00000000e+00,  5.87569381e-01, 0.00000000e+00,  0.00000000e+00],
        [1.99999895e-02,  2.04569964e-05,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00,  0.00000000e+00],
        [-2.04569964e-05, 1.99999895e-02,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  1.00000000e+00]
    ])
    Bd = np.array([[-0.11550861, 0.02], [0.49332431, 0.0], [0.0, 0.0], [0.49332431, 0.0], [0.0, 0.0], [0.0, 0.0]])
    Cd = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    Dd = np.zeros((4, 2))

    car = car_model(x, u, model_params, method="euler")

    assert Ad.shape == car.A.shape
    assert Bd.shape == car.B.shape
    assert Cd.shape == car.C.shape
    assert Dd.shape == car.D.shape
    assert np.all(np.isclose(Ad, car.A))
    assert np.all(np.isclose(Bd, car.B))
    assert np.all(np.isclose(Cd, car.C))
    assert np.all(np.isclose(Dd, car.D))
