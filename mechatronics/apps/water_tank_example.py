import control as ct
import numpy as np
import matplotlib.pyplot as plt
import mechatronics.models as models

t0 = 0                                  # initial time of the simulation
dt = 0.04                               # simulation time interval
t_end = 50                              # final time of the simulation
t = np.arange(t0, t_end + dt, dt)       # time vector

G1 = models.water_tank()
G2 = models.water_tank()
G3 = models.water_tank()
C1 = models.proportional_controller(1000)
C2 = models.proportional_controller(1000)
C3 = models.proportional_controller(5000)

T1 = ct.feedback(G1 * C1)
T2 = ct.feedback(G2 * C2)
T3 = ct.feedback(G3 * C3)

r1 = np.zeros(t.shape)
r1[t < t[300]] = 70
r1[(t >= t[300]) & (t < t[600])] = 20
r1[(t >= t[600]) & (t < t[900])] = 90
r1[t >= t[900]] = 50
r2 = np.zeros(t.shape)
r2[t < t[600]] = 10 + 3 * t[:600]
r2[t >= t[600]] = r2[599] - (t[600:] - t[599])
r3 = 50 + t * np.sin(2*np.pi*(0.005*t)*t)

res1 = ct.forced_response(T1, T=t, U=r1, X0=30)
res2 = ct.forced_response(T2, T=t, U=r2, X0=40)
res3 = ct.forced_response(T3, T=t, U=r3, X0=50)

_, axs = plt.subplots(figsize=(16,9), dpi=120, facecolor=(0.8,0.8,0.8))
axs.plot(res1.t, res1.y.squeeze(), "blue", linewidth=4, label="Tank 1")
axs.plot(res2.t, res2.y.squeeze(), "green", linewidth=4, label="Tank 2")
axs.plot(res3.t, res3.y.squeeze(), "red", linewidth=4, label="Tank 3")
axs.plot(t, r1, "blue", linewidth=1, label="Setpoint 1")
axs.plot(t, r2, "green", linewidth=1, label="Setpoint 2")
axs.plot(t, r3, "red", linewidth=1, label="Setpoint 3")
axs.grid(True)
axs.set_ylabel("Time [s]")
axs.set_ylabel("Tank Volume [m^3]")
axs.legend(loc="upper right", fontsize="small")

plt.show()