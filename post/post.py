import numpy as np
import sys
import matplotlib.pyplot as plt

PATH = "../OUTPUT/CYLINDER_NEW"

if len(sys.argv) < 2:
    raise RuntimeError("Usage: python plot.py <run_name>")

run = sys.argv[1]
print("Post-processing run:", run)

info_path = f"{PATH}/{run}/cp_surface.dat"
data = np.loadtxt(info_path, skiprows=1)

theta = data[:,0]
z     = data[:,1]
p     = data[:,2]   # <-- this is pressure, not Cp

# ---- constants ----
cs2 = 1.0 / 3.0
rho_inf = 1.0
u_inf = 0.1

p_inf = cs2 * rho_inf

# ---- compute Cp ----
Cp = (p - p_inf) / (0.5 * rho_inf * u_inf**2)

# ---- select z plane ----
z_loc = 10
mask = (np.abs(z - z_loc) < 1e-6)

theta_z = theta[mask]
Cp_z = Cp[mask]

# ---- sort ----
idx = np.argsort(theta_z)

# ---- plot ----
plt.plot(theta_z[idx], Cp_z[idx])
plt.xlabel("theta (deg)")
plt.ylabel("Cp")
plt.gca().invert_yaxis()
plt.title(f"Cp at z = {z_loc}")
plt.grid()
plt.show()