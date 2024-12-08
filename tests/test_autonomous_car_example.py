import numpy as np

from mechatronics.apps.autonomous_car_examples import control_action, mpc_constraints
from mechatronics.models import CarModelParameters
from mechatronics.mpc import Constraints

def test_control_action():

    constraints = Constraints(
        uub=np.array([[1.0, 1.0]]).T,
        ulb=np.array([[-1.0, -1.0]]).T,
        Cc=None,
        xlb=np.array([[-1.0, -1.0, -1.0, -1.0]]).T,
        xub=np.array([[1.0, 1.0, 1.0, 1.0]]).T
    )
    u = np.array([[0.8, -0.3]]).T
    ug = np.ones((20, 1))

    unew = control_action(u, ug, constraints)

    assert unew.shape == (2, 1)
    assert np.all(np.isclose(unew, np.array([[1.0, 0.7]]).T))


def test_mpc_constraints():
    
    model_params = CarModelParameters(m=1500, Iz=3000, Caf=38000, Car=66000, lf=2, lr=3, Ts=0.02, mju=0.02, g=9.81)
    x = np.array([[15.0, 0.0, 0.0, 0.0, 0.0, 250.0]]).T
    u = np.array([[0.0, 0.0]]).T
    con = mpc_constraints(x, u, model_params)

    assert con.Cc.shape == (4, 8)
    assert con.ulb.shape == (2, 1)
    assert con.uub.shape == (2, 1)
    assert con.xlb.shape == (4, 1)
    assert con.xub.shape == (4, 1)

    assert con.ulb[0, 0] == -np.pi/300
    assert con.ulb[1, 0] == -0.1
    assert con.uub[0, 0] == np.pi/300
    assert con.uub[1, 0] == 0.1 
