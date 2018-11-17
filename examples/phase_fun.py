import numpy as np
import matplotlib.pylab as plt

from DLA_Control import MZI, Layer, Mesh

N = 100
phi1_list = np.linspace(0, 2*np.pi, N)
phi2_list = np.linspace(0, 2*np.pi, N)

max_phase = np.zeros((N, N))

for i1, p1 in enumerate(phi1_list):
    for i2, p2 in enumerate(phi2_list):
        mzi = MZI(p1, p2)
        phases = np.angle(mzi.M)

        max_phase[i1, i2] = np.max(np.abs(phases))


plt.pcolormesh(phi1_list, phi2_list, max_phase.T)
plt.colorbar()
plt.xlabel('phi1')
plt.ylabel('phi2')
plt.show()