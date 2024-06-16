import control as ct
import mechatronics.models as models
import matplotlib.pyplot as plt
import numpy as np

dt = 0.02
incl_angle = np.pi/6
mass_kg = 100
time = np.arange(0.0, 10.0, dt)

G = models.one_dof_mass(mass_kg=mass_kg, damping_Ns_m=0.0, stiffness_N_m=0.0, name="G")
C = models.pid_controller(kp=1000, ki=400, kd=1000, td=1e-2, name="C")

# This saids the input C.e is fed by output negative -G.displacement and input G.control_force_N is fed by output
# C.u. Note that you can specify multiple signal feeding an input and they are summed.
connections = [["C.e", "-G.displacement_m"], ["G.control_force_N", "C.u"]]

# This says that the combined system has two inputs feeding into the subsystem inputs C.e and G.disturbance_Force_N
inplist = ["C.e", "G.disturbance_force_N"]

# This say the combined system has one output from the subsystem output G.displacement_m
outlist = ["G.displacement_m"]

# The combined system wit reference input and disturbance input, and the measurement output.
T = ct.interconnect([C, G], connections=connections, inplist=inplist, outlist=outlist)

# The reference is a step response.
reference = 1.0*np.ones(time.shape)

# We are feeding a disturbance into the system that models the magnetic train on an incline so some portion of 
# gravity is applied to the mass
disturbance_force = mass_kg*np.sin(incl_angle)*models.GRAVITY_M_S2*np.ones(time.shape)

# Combine the two input vectors and run the simulation.
u = np.vstack((reference, disturbance_force))
res = ct.forced_response(T, T=time, U=u)

# Plot the results
_, axs = plt.subplots()
axs.plot(res.t, res.y.squeeze())
plt.show()
