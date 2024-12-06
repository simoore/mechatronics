"""
This purpose of this application is to visually examine the trajectories in `mechatronics.trajectories` for the
automonous car example.
"""

import matplotlib.pyplot as plt
import mechatronics.trajectories as trajectories


def plot_car_trajectory(traj: trajectories.BicycleModelTrajectory):

    # Plot the world
    _, axs = plt.subplots(nrows=2, ncols=2, layout="constrained", figsize=(10, 10))
    axs[0, 0].plot(traj.xg, traj.yg, label="The ref trajectory")
    axs[0, 1].plot(traj.t, traj.xg, label="X Position")
    axs[0, 1].plot(traj.t, traj.yg, label="Y Position")
    axs[1, 0].plot(traj.t, traj.xdotb, label="X Velocity Body Frame")
    axs[1, 0].plot(traj.t, traj.ydotb, label="Y Velocity Body Frame")
    axs[1, 1].plot(traj.t, traj.psi, label="Heading")
    axs[0, 0].set_xlabel("X Position [m]")
    axs[0, 0].set_ylabel("Y Position [m]")
    axs[0, 0].grid(True)
    axs[0, 0].legend(loc='upper right',fontsize='small')
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Position [m]")
    axs[0, 1].grid(True)
    axs[0, 1].legend(loc='upper right',fontsize='small')
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Velocity [m/s]")
    axs[1, 0].grid(True)
    axs[1, 0].legend(loc='upper right',fontsize='small')
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Heading [rad]")
    axs[1, 1].grid(True)
    axs[1, 1].legend(loc='upper right',fontsize='small')


def main():
    sampling_period=0.02
    traj = trajectories.car_trajectory_1(sampling_period)
    plot_car_trajectory(traj)
    
    traj = trajectories.car_trajectory_2(sampling_period)
    plot_car_trajectory(traj)

    traj = trajectories.car_trajectory_3(sampling_period)
    plot_car_trajectory(traj)

    plt.show()


if __name__ == "__main__":
    main()