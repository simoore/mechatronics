import numpy as np
import matplotlib.pyplot as plt

from mechatronics.models import CarModelParameters, car_model, augmented_car_model
from mechatronics.mpc import Constraints, MPCParameters, constrained_lti, solve_constrained_lti
from mechatronics.trajectories import CarTrajectory, car_trajectory_3


def control_action(u: np.ndarray, ug: np.ndarray, constraints: Constraints) -> np.ndarray:
    """
    Takes the solution of an MPC iteration and computes the control action to apply to the system. This function
    adds the delta values of the control action computed by the MPC optimization and adds them to the previous control
    action. It also saturates the control action.

    Parameters
    ----------
    u
        The previous iteration control action.
    ug
        The global control action solution from the MPC optimization.
    contraints
        Contains the limits of the control action for saturation.

    Returns
    -------
    unew
        The new control action to apply to the system.
    """
    unew = np.clip(u + ug[0:2, :], constraints.xlb[2:, :], constraints.xub[2:, :])
    return unew


def generate_reference_signal(k: int, hz: int, traj: CarTrajectory) -> np.ndarray:
    """
    Generates the global reference signal for a given timestep of the MPC simulation.

    Parameters
    ----------
    k
        The timestep of the simulation.
    hz
        The MPC horizon size.
    traj
        The trajectory we want the vehicle to follow over the entire simulation. We extract just a portion of xdotb,
        psi, x-coord, and y-coord to create the global reference vector.

    Returns
    -------
    ref
        The global reference vector for MPC.
    """
    b = k + hz
    ref = np.empty((4 * hz, 1))
    ref[0::4, 0] = traj.xdotb[k:b]
    ref[1::4, 0] = traj.psi[k:b] 
    ref[2::4, 0] = traj.xg[k:b] 
    ref[3::4, 0] = traj.yg[k:b] 
    return ref


def mpc_constraints(x: np.ndarray) -> Constraints:
    """
    Calculates the constraint parameters for an MPC iteration.

    An extension may be to limit body frame acceleration rather than engine/brake acceleration as that is what
    a passenger or load would experience.

    Parameters
    ----------
    x
        The current state of the iteration. The state is required because we want the lateral velocity to be small 
        w.r.t. the longitudinal velocity.

    Returns
    -------
    constraints
        The constraint parameters to execute an MPC iteration. Note that these are constraints for the augmented
        system as this is what MPC is applied to.
    """
    # States
    xdotb = x[0, 0]

    # Limit on rate of change of delta per control interval, and delta.
    ddelta_lim = np.pi/300
    delta_lim = np.pi/6

    # Limit on the rate of change engine acceleration per control interval.
    da_lim = 0.1

    # Limit on the longitudinal velocity.
    xdotb_max = 30.0
    xdotb_min = 1.0

    # Limit on the lateral velocity- we also want the ydotb to be much less than the longitudinal velocity.
    if 0.17 * xdotb < 3:
        ydotb_lim = 0.17 * xdotb
    else:
        ydotb_lim = 3.0

    # Limit the engine/brake acceleration.
    a_max = 1.0
    a_min = -4.0

    # We are applying limits to the body frame velocities, and the control inputs.
    Cc = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 0, 0, 1]
    ])
    
    constraints = Constraints(
        uub=np.array([[ddelta_lim, da_lim]]).T,
        ulb=np.array([[-ddelta_lim, -da_lim]]).T,
        xub=np.array([[xdotb_max, ydotb_lim, delta_lim, a_max]]).T,
        xlb=np.array([[xdotb_min, -ydotb_lim, -delta_lim, a_min]]).T,
        Cc=Cc,
    )
    return constraints


def main():
    
    ###########################################################################
    # PARAMETERS
    ###########################################################################

    Ts = 0.02           # Sampling period.
    model_params = CarModelParameters(m=1500, Iz=3000, Caf=38000, Car=66000, lf=2, lr=3, Ts=Ts, mju=0.02, g=9.81)
    Q = np.diag([100, 20000, 1000, 1000]) 
    S = np.diag([100, 20000, 1000, 1000]) 
    R = np.diag([100, 1])             
    mpc_params = MPCParameters(hz=30, Q=Q, S=S, R=R) # Not sure how generic these parameters are.
    traj = car_trajectory_3(Ts, 2.0)

    ###########################################################################
    # INITIAL STATE OF THE SYSTEM
    ###########################################################################

    xdotb0 = traj.xdotb[0]
    ydotb0 = traj.ydotb[0]
    psi0 = traj.psi[0]
    psidot0 = 0.0
    xpos0 = traj.xg[0]
    ypos0 = traj.yg[0]
    u0 = np.array([[0.0, 0.0]]).T
    x0 = np.array([[xdotb0, ydotb0, psi0, psidot0, xpos0, ypos0]]).T
    
    ###########################################################################
    # SIMULATION OF CONTROL SYSTEMS
    ###########################################################################

    sim_size = len(traj.t) - mpc_params.hz          # Number of timesteps to simulate.
    u = u0                                          # The control action applied to this interation.
    x = x0.copy()                                   # The state of the un-augmented system.
    t_record = np.zeros((sim_size,))                # Store the time vector for the simulation
    u_record = np.zeros((u0.shape[0], sim_size))    # Stores the control action applied throught the simulation.
    x_record = np.zeros((x0.shape[0], sim_size))    # Stores the state of the system calculated at each time step.

    for k in range(sim_size):

        # Store the state at each time step for visualization.
        t_record[k] = traj.t[k]
        x_record[:, k] = x[:, 0]

        # Pick out the section of the reference trajectory to be used for this MPC interation.
        rg = generate_reference_signal(k, mpc_params.hz, traj)

        # Build the linearized model fo the car.
        sysd = car_model(x, u, model_params)
        
        assert rg.shape == (sysd.noutputs * mpc_params.hz, 1)
        assert x.shape == (sysd.nstates, 1)

        # Augment the system, compute the MPC matrices and apply the control law.
        sysd_aug = augmented_car_model(sysd)
        constraints = mpc_constraints(x)
        xaug = np.vstack((x, u))
        Hbar, Fbar, G, h = constrained_lti(sysd_aug, mpc_params, constraints, xaug)
        ug = solve_constrained_lti(Hbar, Fbar, G, h, xaug, rg)
        if ug is None:
            print("solver failed, exiting sim")
            break
    
        assert ug.shape == (sysd.ninputs * mpc_params.hz, 1)

        # Apply the control action to the system.
        u = control_action(u, ug, constraints)
        x = sysd.A @ x + sysd.B @ u     # Should technically use the nonlinear model
        u_record[:, k] = u[:, 0]

        if k % 100 == 0:
            print(f"Completed {k} iterations of {sim_size}")

    ###########################################################################
    # VISUALIZATION
    ###########################################################################

    _, axs = plt.subplots(nrows=2, ncols=3, layout="constrained", figsize=(10, 15))
    axs[0, 0].plot(traj.xg, traj.yg, label="The ref trajectory")
    axs[0, 0].plot(x_record[4, :], x_record[5, :], label="state")
    axs[0, 1].plot(traj.t, traj.xg, label="X Ref")
    axs[0, 1].plot(t_record, x_record[4, :], label="X Pos")
    axs[0, 1].plot(traj.t, traj.yg, label="Y Ref")
    axs[0, 1].plot(t_record, x_record[5, :], label="Y Pos")
    axs[0, 2].plot(t_record, u_record[0, :], label="Steering Angle")
    axs[1, 0].plot(traj.t, traj.xdotb, label="X Vel Ref")
    axs[1, 0].plot(t_record, x_record[0, :], label="X Vel")
    axs[1, 0].plot(traj.t, traj.ydotb, label="Y Vel Ref")
    axs[1, 0].plot(t_record, x_record[1, :], label="Y Vel")
    axs[1, 1].plot(traj.t, traj.psi, label="Heading Ref")
    axs[1, 1].plot(t_record, x_record[2, :], label="Heading")
    axs[1, 2].plot(t_record, u_record[1, :], label="Engine/brake Acc")

    axs[0, 0].set_xlabel("X Position [m]")
    axs[0, 0].set_ylabel("Y Position [m]")
    axs[0, 0].grid(True)
    axs[0, 0].legend(loc='upper right',fontsize='small')
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Position [m]")
    axs[0, 1].grid(True)
    axs[0, 1].legend(loc='upper right',fontsize='small')
    axs[0, 2].set_xlabel("Time [s]")
    axs[0, 2].set_ylabel("Angle [rad]")
    axs[0, 2].grid(True)
    axs[0, 2].legend(loc='upper right',fontsize='small')
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Velocity [m/s]")
    axs[1, 0].grid(True)
    axs[1, 0].legend(loc='upper right',fontsize='small')
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Heading [rad]")
    axs[1, 1].grid(True)
    axs[1, 1].legend(loc='upper right',fontsize='small')
    axs[1, 2].set_xlabel("Time [s]")
    axs[1, 2].set_ylabel("Accel [m/s/s]")
    axs[1, 2].grid(True)
    axs[1, 2].legend(loc='upper right',fontsize='small')

    plt.show()

    
if __name__ == "__main__":
    main()