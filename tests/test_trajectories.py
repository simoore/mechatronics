import numpy as np

from mechatronics.trajectories import straight_trajectory, StraightTrajectoryConfig
from mechatronics.trajectories import cubic_trajectory, CubicTrajectoryConfig


def test_straight_trajectory():

    cfg = StraightTrajectoryConfig(final_t=10.0, sampling_period=0.01, xdot=2.0, ycoord=-7.0)
    traj = straight_trajectory(cfg)

    expected_length = 1000
    assert len(traj.x) == expected_length
    assert len(traj.y) == expected_length
    assert len(traj.psi) == expected_length
    assert np.all(traj.x == cfg.xdot * traj.t)
    assert np.all(traj.y == -7.0)
    assert np.all(traj.psi == 0.0)


def test_cubic_trajectory():

    cfg = CubicTrajectoryConfig(final_t=10.0, sampling_period=0.01, xdot=2.0, initial_y=-7.0, final_y=7.0)
    traj = cubic_trajectory(cfg)

    expected_length = 1000
    assert len(traj.x) == expected_length
    assert len(traj.y) == expected_length
    assert len(traj.psi) == expected_length
    assert np.all(traj.x == cfg.xdot * traj.t)
    assert traj.y[0] == -7.0
    assert np.isclose(traj.y[-1], 7.0)
    assert np.isclose(traj.y.max(), 7.0)
    assert traj.y.min() == -7.0
