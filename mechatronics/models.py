import control as ct
import dataclasses
import numpy as np


DENSITY_OF_WATER_KG_M3 = 1000.0
GRAVITY_M_S2 = 9.8


def water_tank() -> ct.StateSpace:
    """
    This function returns a model of a water tank. The input is the net kg/s mass flow into the tank. The state is the 
    volume of water in the tank. The output is the volume of water in the tank.

    Returns
    -------
    model
        The state space model of the water tank.
    """
    rho = DENSITY_OF_WATER_KG_M3
    A = np.zeros((1, 1))
    B = np.array([[1.0 / rho]])
    C = np.array([[1.0]])
    D = np.zeros((1, 1))
    inputs = ["mass_flow_kg_s"]
    outputs = ["volume_m3"]
    states = ["volume_m3"]
    return ct.ss(A, B, C, D, inputs=inputs, outputs=outputs, states=states)


def proportional_controller(kp: float) -> ct.TransferFunction:
    """
    Returns a transfer function object for a proportional controller with gain `kp`.
    """
    return ct.tf(kp, 1)


def pid_controller(kp: float, ki: float, kd: float, td: float, name: str) -> ct.TransferFunction:
    """
    Returns a transfer function object for a PID controller.
    """
    sys = kp + ct.tf(ki, [1, 0]) + ct.tf([kd, 0], [td, 1])
    return ct.tf(sys, inputs=["e"], outputs=["u"], name=name)


def one_dof_mass(mass_kg: float, damping_Ns_m: float, stiffness_N_m: float, name: str = "") -> ct.StateSpace:
    """
    Returns the model of a spring-mass-damper system.
    """
    A = np.array([[-damping_Ns_m/mass_kg, -stiffness_N_m/mass_kg], [1.0, 0.0]])
    B = np.array([[1.0/mass_kg, 1.0/mass_kg], [0.0, 0.0]])
    C = np.array([[0.0, 1.0]])
    D = np.array([[0.0, 0.0]])
    inputs = ["control_force_N", "disturbance_force_N"]
    outputs = ["displacement_m"]
    states = ["velocity_m_s", "displacement_m"]
    return ct.ss(A, B, C, D, inputs=inputs, outputs=outputs, states=states, name=name)


@dataclasses.dataclass
class BicycleModelParameters:
    m: float
    """The mass of the vehicle in kg."""
    Iz: float
    """The moment of inertia of the vehicle around the z-axis.""" 
    Caf: float
    """The cornering stiffness of a front wheel."""
    Car: float 
    """The cornering stiffness of a read wheel."""
    lf: float 
    """The distance between the center of mass and the front wheels"""
    lr: float 
    """The distance between the center of mass and the rear wheels"""
    Ts: float 
    """The sampling period of the the discrete time representation in s"""
    x_dot: float
    """The longitudinal velocity in m/s"""


def bicycle_model(p: BicycleModelParameters) -> ct.StateSpace:
    """
    The bicycle model is a model of the car where the where the two rear tires are fused as one and the two front
    tires are fused as one to simplify the modeling of the forces applied by the tires. This is the linearized 
    model that can be used when yaw angle is small.

    The states are [lateral_velocity, yaw_angle, yaw_angle_rate, lateral_position]
    The input is the angle of the tire.
    The output is the [yaw_angle, lateral_position]

    Parameters
    ----------
    p
        The parameters of the bicycle model.

    Returns
    -------
    sysd
        The discrete time model of the car using the bicycle model.
    """
    A1 = -(2 * p.Caf + 2 * p.Car) / (p.m * p.x_dot)
    A2 = -p.x_dot - (2 * p.Caf * p.lf - 2 * p.Car * p.lr) / (p.m * p.x_dot)
    A3 = -(2 * p.lf * p.Caf - 2 * p.lr * p.Car) / (p.Iz * p.x_dot)
    A4 = -(2 * p.lf**2 * p.Caf + 2 * p.lr** 2 * p.Car) / (p.Iz * p.x_dot)

    A = np.array([[A1, 0, A2, 0],[0, 0, 1, 0],[A3, 0, A4, 0],[1, p.x_dot, 0, 0]])
    B = np.array([[2 * p.Caf / p.m], [0], [2 * p.lf * p.Caf / p.Iz], [0]])
    C = np.array([[0, 1, 0, 0], [0, 0, 0, 1]])
    D = np.array([[0], [0]])
    sysc = ct.ss(A, B, C, D)
    sysd = ct.sample_system(sysc, p.Ts)

    return sysd


def augmented_bicycle_model(sysd: ct.StateSpace) -> ct.StateSpace:
    """
    We want to change the the input of the system to be the change in tire angle, not the tire angle itself. In this
    case the tire angle becomes a state of the system. This function actually works for any inputs to the system and
    will convert all inputs to rate of change intputs.

    Parameters
    ----------
    sysd
        The system to augment.

    Returns
    -------
    sys_aug
        The augmented system.
    """
    A1 = np.hstack((sysd.A, sysd.B))
    A2 = np.hstack((np.zeros((sysd.ninputs, sysd.nstates)), np.identity(sysd.ninputs)))

    A_aug = np.vstack((A1, A2))
    B_aug = np.vstack((sysd.B, np.identity(sysd.ninputs)))
    C_aug = np.hstack((sysd.C, np.zeros((sysd.noutputs, sysd.ninputs))))
    D_aug = sysd.D

    sys_aug = ct.StateSpace(A_aug, B_aug, C_aug, D_aug, sysd.dt)
    return sys_aug
