import numpy as np
import numpy.random as npr
import matplotlib.pylab as plt

from DLA_Control import Mesh
from DLA_Control import TriangleOptimizer, ClementsOptimizer
from DLA_Control.plots import plot_bar_3d, colorbar, plot_powers

""" FIRST OPTIMIZE A TRIANGULAR MESH """

# create a triangular mesh
N = 10
mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)
print(mesh)

# comlex valued input coupling vector
input_values = np.zeros((N, 1))
input_values[-1] = 1
input_values = npr.random((N, 1))

f, (ax1, ax2) = plt.subplots(2, constrained_layout=True, figsize=(5, 5))

# couple light in and look at powers throughout mesh
mesh.input_couple(input_values)
im1 = plot_powers(mesh, ax=ax1)
colorbar(im1)
ax1.set_title('power distribution before optimizing')

# target output complex amplitude
output_target = np.ones((N, 1))

# define an optimizer over the triangular mesh
TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)

# optimize the mesh by pushing power to top port and redistributing
TO.optimize(algorithm='up_down')

# can also try the 'spread' algorithm, which distributes power as much as possible
TO.optimize(algorithm='spread')

# look at powers after optimizing
im2 = plot_powers(mesh, ax=ax2)
colorbar(im2)
ax2.set_title('power distribution after optimizing')


""" NEXT OPTIMIZE A CLEMENTS MESH """

# create a clements mesh
N = 20
M = 10
mesh = Mesh(N, mesh_type='clements', initialization='random', M=M)
print(mesh)

# comlex valued input coupling vector
input_values = np.zeros((N, 1))
input_values[N//2] = 1
input_values = npr.random((N, 1))

fig_inches = 9
f1, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(fig_inches, fig_inches*M/N))

# couple light in and look at powers throughout mesh
mesh.input_couple(input_values)
im = plot_powers(mesh, ax=ax1)
colorbar(im)
ax1.set_title('power distribution before optimizing')

# target output complex amplitude
output_target = np.zeros((N, 1))
output_target[N//2] = 1
output_target = np.ones((N, 1))

# define an optimizer over the triangular mesh
CO = ClementsOptimizer(mesh, input_values=input_values, output_target=output_target)

# optimize the mesh by pushing power to top port and redistributing
CO.optimize(algorithm='smart', verbose=False)

# look at powers after optimizing
im = plot_powers(mesh, ax=ax2)
colorbar(im)
ax2.set_title('power distribution after optimizing')
plt.show()

power_map = np.zeros((N, M))
for li in range(M):
    pow_layer_i = mesh.get_layer_powers(layer_index=li)
    power_map[:, li] = pow_layer_i[:,0]

################ 3D bar plot
# fig = plt.figure(figsize=(8, 3))
# ax = fig.add_subplot(111, projection='3d')
# plot_bar_3d(power_map, ax=ax)

plt.show()
