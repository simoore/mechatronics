"""
This purpose of this application is to visually examine some of the trajectories in `mechatronics.trajectories`.
"""

import matplotlib.pyplot as plt
import mechatronics.trajectories as trajectories


def plot_bicycle_trajectory(traj: trajectories.BicycleModelTrajectory):

    _, axs = plt.subplots(nrows=4)
    axs[0].plot(traj.t, traj.x)
    axs[1].plot(traj.t, traj.y)
    axs[2].plot(traj.t, traj.psi)
    axs[3].plot(traj.x, traj.y)


def main():

    cfg = trajectories.StraightTrajectoryConfig(final_t=10, sampling_period=0.01, xdot=2.0, ycoord=-7.0)
    traj = trajectories.straight_trajectory(cfg=cfg)
    plot_bicycle_trajectory(traj)
    
    cfg = trajectories.CubicTrajectoryConfig(final_t=10, sampling_period=0.01, xdot=2.0, initial_y=3.0, final_y=5.0)
    traj = trajectories.cubic_trajectory(cfg=cfg)
    plot_bicycle_trajectory(traj)

    cfg = trajectories.SineTrajectoryConfig(final_t=10, sampling_period=0.01, xdot=2.0, amplitude=3.0, frequency=0.15)
    traj = trajectories.sine_trajectory(cfg=cfg)
    plot_bicycle_trajectory(traj)

    plt.show()


if __name__ == "__main__":
    main()