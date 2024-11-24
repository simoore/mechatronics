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


@dataclasses.dataclass
class CarTrajectory:

    t: np.ndarray
    """The time vector for this trajectory in s"""

    xg: np.ndarray
    """The global x-coordinates of the trajectory at each time point in m"""

    yg: np.ndarray
    """The global y-coordinates of the trajectory at each time point in m"""

    psi: np.ndarray
    """The heading trajectory at each time point in rad"""

    xdotb: np.ndarray
    """The longitudal velocity trajectory in the body frame in m/s"""

    ydotb: np.ndarray
    """The lateral velocity trajectory in the body frame in m/s"""


def compute_psi_xdotb_ydotb(ts: float, xg: np.ndarray, yg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the heading, and velocity reference for a given (x,y) trajectory.

    Parameters
    ----------
    ts
        The sampling period of `xg` and `yg`.
    xg
        The x-coordinate trajectory in the global frame.
    yg
        The y-coordinate trajectory in the global frame.

    Returns
    -------
    xdotb
        The longitudal velocity of the trajectory in the body frame.
    ydotb
        The lateral velocity of the trajectory in the body frame.
    psi
        The heading.
    """
    xdot = np.gradient(xg, ts)
    ydot = np.gradient(yg, ts)
    psi = np.unwrap(np.arctan2(ydot, xdot))
    xdotb = np.cos(psi) * xdot + np.sin(psi) * ydot
    ydotb = -np.sin(psi) * xdot + np.cos(psi) * ydot
    return xdotb, ydotb, psi
    

def car_trajectory_1(ts: float) -> CarTrajectory:
    """
    This creates a trajectory for an autonomous car to follow in a 2D plane.

    Parameters
    ----------
    ts
        The sampling period of the trajectory.
    
    Returns
    -------
    traj
        The trajectory for the car.
    """
    final_t = 60.0
    time = np.arange(0.0, final_t, ts)
    xg = 15 * time
    yg = 750 / 900**2 * xg**2 + 250
    xdotb, ydotb, psi = compute_psi_xdotb_ydotb(ts, xg, yg)
    return CarTrajectory(t=time, xg=xg, yg=yg, psi=psi, xdotb=xdotb, ydotb=ydotb)


def car_trajectory_2(ts: float) -> CarTrajectory:
    """
    This creates a trajectory for an autonomous car to follow in a 2D plane.

    Parameters
    ----------
    ts
        The sampling period of the trajectory.
    
    Returns
    -------
    traj
        The trajectory for the car.
    """
    twp = np.array([0.0, 40.0, 100.0, 140.0])
    time = np.arange(0.0, twp[-1], ts)

    t1 = time[(time >= twp[0]) & (time < twp[1])]
    x1 = 15 * t1
    y1 = 50 * np.sin(2 * np.pi * 0.75 / 40 * t1) + 250
    t2 = time[(time >= twp[1]) & (time < twp[2])]
    x2 = 300 * np.cos(2 * np.pi * 0.5 / 60 * (t2 - 40) - np.pi / 2) + 600
    y2 = 300 * np.sin(2 * np.pi * 0.5 / 60 * (t2 - 40) - np.pi / 2) + 500
    t3 = time[(time >= twp[2]) & (time < twp[3])]
    x3 = 600 - 15 * (t3 - 100)
    y3 = 50 * np.cos(2 * np.pi * 0.75 / 40 * (t3 - 100)) + 750

    xg = np.concatenate((x1, x2, x3), axis=0)
    yg = np.concatenate((y1, y2, y3), axis=0)

    xdotb, ydotb, psi = compute_psi_xdotb_ydotb(ts, xg, yg)
    return CarTrajectory(t=time, xg=xg, yg=yg, psi=psi, xdotb=xdotb, ydotb=ydotb) 


def compute_cubic_coefficients(t: tuple[float], x: tuple[float], dx: tuple[float]) -> np.ndarray:
    """
    Calculates the cubic coefficients given two points for a function of time and the time derivates at the two points.

    Parameters
    ----------
    t
        (t0, t1) time coordinates of the two end points of the cubic spline.
    x
        (x0, x1) the value of the cubic polynomical at the two time poitns.
    dx
        (dx0, dx1) the time derivate of the cubic polynomial at the two end points.

    Returns
    -------
    Avec
        The polynomical coefficients of the cubic spline. Avec = [a0, a1, a2, a3] 
        where x(t) = a0 + a1*t + a2*t^2 + a3*t^3
    """
    assert len(t) == 2
    assert len(x) == 2
    assert len(dx) == 2

    Mmat = np.array([[1, t[0], t[0]**2, t[0]**3], [1, t[1], t[1]**2, t[1]**3], [0, 1, 2*t[0], 3*t[0]**2], 
        [0, 1, 2*t[1], 3*t[1]**2]])
    Cvec = np.array([[x[0], x[1], dx[0], dx[1]]]).T
    Avec = np.linalg.solve(Mmat, Cvec)
    return np.squeeze(Avec)


def sample_cubic(time: np.ndarray, t: tuple[float], x: tuple[float], dx: tuple[float]) -> np.ndarray:
    """
    Both calculates the cubic polynomical of a trajectory and then samples it for a given time vector. That is we 
    know how long the trajectory runs for and we calculate the cubic polynomical for points in t = [t0,t1)

    Parameters
    ----------
    time
        The time vector for the full sampled trajectory.
    t
        (t0, t1) time coordinates of the two end points of the cubic spline.
    x
        (x0, x1) the value of the cubic polynomical at the two time poitns.
    dx
        (dx0, dx1) the time derivate of the cubic polynomial at the two end points.

    Returns
    -------
    x_sampled
        The sampled cubic polynomial.
    """
    Avec = compute_cubic_coefficients(t, x, dx)
    return np.polyval(np.flip(Avec), time[(time >= t[0]) & (time < t[1])])
    

def car_trajectory_3(ts: float, gain: float = 1.0) -> CarTrajectory:
    """
    This creates a trajectory for an autonomous car to follow in a 2D plane.

    Parameters
    ----------
    ts
        The sampling period of the trajectory.
    gain
        Scale the trajectory by this factor.
    
    Returns
    -------
    traj
        The trajectory for the car.
    """
    # The xg, yg, xdot, and ydot levels at the the beginning and ends of each cubic term.
    twp = np.array([0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154])
    xwp = np.array([0, 60, 110, 140, 160, 110, 40, 10, 40, 70, 110, 150]) * gain
    ywp = np.array([40, 20, 20, 60, 100, 140, 140, 80, 60, 60, 90, 90]) * gain
    xdotwp = np.array([2, 1, 1, 1, 0, -1, -1, 0, 1, 1, 1, 1]) * 3 * gain
    ydotwp = np.array([0, 0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 0]) * 3 * gain

    time = np.arange(0.0, twp[-1], ts)
    xg = []
    yg = []
    for i in range(1, len(twp)):
        xcub = sample_cubic(time, (twp[i-1], twp[i]), (xwp[i-1], xwp[i]), (xdotwp[i-1], xdotwp[i]))
        ycub = sample_cubic(time, (twp[i-1], twp[i]), (ywp[i-1], ywp[i]), (ydotwp[i-1], ydotwp[i]))
        xg = np.concatenate([xg, xcub])
        yg = np.concatenate([yg, ycub])
    xdotb, ydotb, psi = compute_psi_xdotb_ydotb(ts, xg, yg)

    return CarTrajectory(t=time, xg=xg, yg=yg, psi=psi, xdotb=xdotb, ydotb=ydotb) 
