import numpy as np
import matplotlib.pyplot as plt

faces   = np.loadtxt("faces.dat",   usecols=(0,1,2))
edges   = np.loadtxt("edges.dat",   usecols=(0,1,2))
corners = np.loadtxt("corners.dat", usecols=(0,1,2))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(*faces.T,   s=4,  label="Faces")
ax.scatter(*edges.T,   s=8,  label="Edges")
ax.scatter(*corners.T, s=20, label="Corners")

# aspect ratio fix
all_pts = np.vstack((faces, edges, corners))
mins = all_pts.min(axis=0)
maxs = all_pts.max(axis=0)
ax.set_box_aspect(maxs - mins)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.legend()
plt.show()

