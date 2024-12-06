import numpy as np
import matplotlib.pyplot as plt

from mechatronics.models import BicycleModelParameters, bicycle_model, augmented_bicycle_model
from mechatronics.mpc import unconstrained_lti, MPCParameters
from mechatronics.trajectories import CubicTrajectoryConfig, cubic_trajectory, BicycleModelTrajectory
from mechatronics.trajectories import StraightTrajectoryConfig, straight_trajectory
from mechatronics.trajectories import SineTrajectoryConfig, sine_trajectory, car_trajectory_3, car_trajectory_1, car_trajectory_2


def clamp(x: float, lb: float, ub: float) -> float:
    """
    The return value is `x` if lb < x < ub, otherwise it returns `lb` if x < lb or `ub` if x > ub.
    Used to apply saturation.
    """
    return lb if x < lb else ub if x > ub else x


def generate_reference_signal(k: int, hz: int, traj: BicycleModelTrajectory) -> np.ndarray:
    """
    Generates the global reference signal for a given timestep of the MPC simulation.

    Parameters
    ----------
    k
        The timestep of the simulation.
    hz
        The MPC horizon size.
    traj
        The trajectory we want the vehicle to follow over the entire simulation. We extract just a portion of the
        y-coord reference, and the yaw (psi) reference to create the global reference vector.
    Returns
    -------
    ref
        The global reference vector for MPC.
    """
    b = k + hz
    ref = np.empty((2 * hz, 1))
    ref[0::2, 0] = traj.psi[k:b]
    ref[1::2, 0] = traj.y[k:b] 
    return ref


def main():
    
    ###########################################################################
    # PARAMETERS
    ###########################################################################

    Ts = 0.02           # Sampling period.
    xdot = 20.0         # Longitudinal velocity.
    lb = -np.pi/6.0     # Lower bound on tire angle.
    ub = np.pi/6.0      # Upper bounf on tire angle.
    TRAJ_TYPE = "cubic" # The type of trajectory to use.

    Q = np.array([[1, 0], [0, 1]])  # weights for outputs (all samples, except the last one)
    S = np.array([[1, 0], [0, 1]])  # weights for the final horizon period outputs
    R = np.array([[1]])             # weights for inputs (only 1 input in our case)

    model_params = BicycleModelParameters(m=1500, Iz=3000, Caf=19000, Car=33000, lf=2, lr=3, Ts=Ts, x_dot=xdot)
    straight_config = StraightTrajectoryConfig(final_t=10, sampling_period=Ts, xdot=xdot, ycoord=-9.0)
    cubic_config = CubicTrajectoryConfig(final_t=10, sampling_period=Ts, xdot=xdot, initial_y=-7.0, final_y=7.0)
    sine_config = SineTrajectoryConfig(final_t=10.0, sampling_period=Ts, xdot=xdot, amplitude=5.0, frequency=0.25)
    mpc_params = MPCParameters(hz=20, Q=Q, S=S, R=R)

    ###########################################################################
    # SIMULATION COMPONENTS
    ###########################################################################

    if TRAJ_TYPE == "straight":
        traj = straight_trajectory(straight_config)
    elif TRAJ_TYPE == "cubic":
        traj = cubic_trajectory(cubic_config)
    elif TRAJ_TYPE == "sine":
        traj = sine_trajectory(sine_config)
    else:
        raise RuntimeError("Invalid trajectory type selected")
    
    sysd = bicycle_model(model_params)
    sysd_aug = augmented_bicycle_model(sysd)
    Hbar, Fbar = unconstrained_lti(sysd_aug, mpc_params)
    control_law = -np.linalg.inv(Hbar) @ Fbar

    ###########################################################################
    # INITIAL STATE OF THE SYSTEM
    ###########################################################################

    ydot0 = 0.0
    psi0 = 0.0
    psidot0 = 0.0
    y0 = straight_config.ycoord + 10.0
    u0 = 0.0
    x0 = np.array([[ydot0, psi0, psidot0, y0]]).T
    
    ###########################################################################
    # SIMULATION OF CONTROL SYSTEMS
    ###########################################################################

    sim_size = len(traj.t) - mpc_params.hz          # Number of timesteps to simulate.
    u = u0                                          # The control action applied to this interation.
    x = x0.copy()                                   # The state of the un-augmented system.
    t_record = np.zeros((sim_size,))                # Store the time vector for the simulation
    u_record = np.zeros((sim_size,))                # Stores the control action applied throught the simulation.
    x_record = np.zeros((sysd.nstates, sim_size))   # Stores the state of the system calculated at each time step.

    for k in range(sim_size):

        t_record[k] = traj.t[k]
        x_record[:, k] = x[:, 0]
        rg = generate_reference_signal(k, mpc_params.hz, traj)

        assert rg.shape == (sysd.noutputs * mpc_params.hz, 1)
        assert x.shape == (sysd.nstates, 1)
        
        ug = control_law @ np.vstack((x, np.array([[u]]), rg))
        u = clamp(u + ug[0, 0], lb, ub)
        x = sysd.A @ x + sysd.B @ np.array([[u]])
        u_record[k] = u


    ###########################################################################
    # VISUALIZATION
    ###########################################################################

    fig, axs = plt.subplots(nrows=3)
    axs[0].plot(traj.t, traj.y, label="reference")
    axs[0].plot(t_record, x_record[3, :], label="state")
    axs[0].set_ylabel("Y Displacement [m]")
    axs[0].legend()
    axs[0].set_title("MPC Simulation")

    axs[1].plot(t_record, u_record)
    axs[1].set_ylabel("Tire Angle [rad]")

    axs[2].plot(traj.t, traj.psi, label="reference")
    axs[2].plot(t_record, x_record[1, :], label="state")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Yaw Angle [rad]")
    axs[2].legend()

    fig.set_size_inches(8, 6)
    plt.show()

    
if __name__ == "__main__":
    main()