import dataclasses
import numpy as np


@dataclasses.dataclass
class BicycleModelTrajectory:

    t: np.ndarray
    """The time vector for this trajectory in s"""

    x: np.ndarray
    """The x-coordinates of the trjectory at each time point in m"""

    y: np.ndarray
    """The y-coordinates of the trjectory at each time point in m"""

    psi: np.ndarray
    """Yaw angles of the trajectory in radians."""


def compute_yaw(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the angle of an (x, y) trajectory which has been uniformly sampled.

    Parameters
    ----------
    x
        The x-coordinates of the trajectory.
    y
        The y-coordinates of the trajectory.

    Returns
    -------
    psi
        The angle of the trajectory.

    TODO: Consider unwarping the result.
    """
    assert x.ndim == 1
    assert y.ndim == 1
    assert len(x) == len(y)
    dy = np.gradient(y)
    dx = np.gradient(x)
    psi = np.arctan2(dy, dx)
    return psi


@dataclasses.dataclass
class StraightTrajectoryConfig:

    final_t: float
    """The trajectory is generated from 0.0 seconds til this time."""

    sampling_period: float
    """The desired sampling period of the trajectory is s."""
    
    xdot: float
    """The constant longitudinal velocity of the bicycle."""

    ycoord: float
    """The const y-coordinate of the trajectory."""


def straight_trajectory(cfg: StraightTrajectoryConfig) -> BicycleModelTrajectory:
    """
    This generates a constant in y-direction trajectory moving at a constant speed in the x-direction for the bicycle
    model.

    Parameters
    ----------
    cfg
        The parameters required to compute the trajectory.

    Returns
    -------
    trajectory
        The (time x, y, psi) trajectory.
    """
    time = np.arange(0.0, cfg.final_t, cfg.sampling_period)
    x = cfg.xdot * time
    y = cfg.ycoord * np.ones(len(time))
    return BicycleModelTrajectory(time, x, y, compute_yaw(x, y))


@dataclasses.dataclass
class CubicTrajectoryConfig:

    final_t: float
    """The trajectory is generated from 0.0 seconds til this time."""

    sampling_period: float
    """The desired sampling period of the trajectory is s."""
    
    xdot: float
    """The constant longitudinal velocity of the bicycle"""

    initial_y: float
    """The initial y-coordinate of the trajectory"""

    final_y: float
    """The final y-coordinate of the trajectory"""


def cubic_trajectory(cfg: CubicTrajectoryConfig) -> BicycleModelTrajectory:
    """
    This generates a cubic in y-direction trajectory moving at a constant speed in the x-direction for the bicycle
    model. This most accurately respresents the trajectory of a car performing a lane change.

    https://www.mdpi.com/2076-3417/12/19/9662

    Parameters
    ----------
    cfg
        The parameters required to compute the trajectory.

    Returns
    -------
    trajectory
        The (time, x, y, psi) trajectory.
    """
    time = np.arange(0.0, cfg.final_t, cfg.sampling_period)
    w = cfg.final_y - cfg.initial_y
    l = cfg.xdot * cfg.final_t
    x = cfg.xdot * time
    y = cfg.initial_y + 3 * w * (x / l)**2 - 2 * w * (x / l)**3 
    return BicycleModelTrajectory(time, x, y, compute_yaw(x, y))


@dataclasses.dataclass
class SineTrajectoryConfig:

    final_t: float
    """The trajectory is generated from 0.0 seconds til this time."""

    sampling_period: float
    """The desired sampling period of the trajectory is s."""
    
    xdot: float
    """The constant longitudinal velocity of the bicycle"""

    amplitude: float
    """The amplitude of the sin trajectory in the y-axis direction."""

    frequency: float
    """The frequency of the sin trajectory in the y-axis direction."""


def sine_trajectory(cfg: SineTrajectoryConfig) -> BicycleModelTrajectory:
    """
    This generates a sinusoidal y-direction trajectory as the car moves in a constant velocity in the x-direction 
    for the bicycle model.

    Parameters
    ----------
    cfg
        The parameters required to compute the trajectory.

    Returns
    -------
    trajectory
        The (t, x, y, psi) trajectory.
    """
    time = np.arange(0.0, cfg.final_t, cfg.sampling_period)
    x = cfg.xdot * time
    y = cfg.amplitude * np.sin(2 * np.pi * cfg.frequency * time)
    return BicycleModelTrajectory(time, x, y, compute_yaw(x, y))
