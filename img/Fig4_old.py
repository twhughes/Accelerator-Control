import numpy as np
import numpy.random as npr
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

from DLA_Control import Mesh
from DLA_Control import TriangleOptimizer, ClementsOptimizer
from DLA_Control.plots import plot_bar_3d, colorbar, plot_powers

f = plt.figure(figsize=(5, 9))
gs = gridspec.GridSpec(2, 2, figure=f, height_ratios=[1, 1])
ax_a = plt.subplot(gs[0, 0])
ax_b = plt.subplot(gs[0, 1])
ax_c = plt.subplot(gs[1, 0])
ax_d = plt.subplot(gs[1, 1])

# # create a clements mesh
N = 15
M = 5
mesh = Mesh(N, mesh_type='clements', initialization='random', M=M)
print(mesh)

# comlex valued input coupling vector
random_vals = npr.random((N, 1))
uniform_vals = np.ones((N, 1))
middle_vals = np.zeros((N, 1))
middle_vals[N//2, 0] = 1

vmax=0.3

### A
print("working on figure a")

CO = ClementsOptimizer(mesh, input_values=random_vals, output_target=uniform_vals)
CO.optimize(algorithm='smart', verbose=False)

im = plot_powers(mesh, ax=ax_a)
im.set_clim(0,vmax)

ax_a.get_xaxis().set_visible(False)

ax_a.set_title('')

colorbar(im)

### B
print("working on figure b")
random_vals = npr.random((N, 1))

CO = ClementsOptimizer(mesh, input_values=random_vals, output_target=uniform_vals)
CO.optimize(algorithm='smart', verbose=False)

im = plot_powers(mesh, ax=ax_b)
im.set_clim(0,vmax)

ax_b.get_xaxis().set_visible(False)
ax_b.get_yaxis().set_visible(False)

ax_b.set_title('')

colorbar(im)

### C
print("working on figure c")
random_vals = npr.random((N, 1))

CO = ClementsOptimizer(mesh, input_values=random_vals, output_target=uniform_vals)
CO.optimize(algorithm='smart', verbose=False)

im = plot_powers(mesh, ax=ax_c)
im.set_clim(0,vmax)

ax_c.set_title('')

colorbar(im)


### D
print("working on figure d")
random_vals = npr.random((N, 1))

CO = ClementsOptimizer(mesh, input_values=random_vals, output_target=uniform_vals)
CO.optimize(algorithm='smart', verbose=False)

im = plot_powers(mesh, ax=ax_d)
im.set_clim(0,vmax)

ax_d.get_yaxis().set_visible(False)

ax_d.set_title('')

colorbar(im)

# Saving / Displaying

plt.show()

