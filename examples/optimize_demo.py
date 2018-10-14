import numpy as np
import numpy.random as npr
import matplotlib.pylab as plt

from DLA_Control import Mesh
from DLA_Control import TriangleOptimizer, ClementsOptimizer

# create a triangular mesh
N = 10
mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)
print(mesh)

# comlex valued input coupling vector
input_values = np.zeros((N, 1))
input_values[-1] = 1
input_values = npr.random((N, 1))

f, (ax1, ax2) = plt.subplots(2, constrained_layout=True, figsize=(7, 7))

# couple light in and look at powers throughout mesh
mesh.input_couple(input_values)
mesh.plot_powers(ax=ax1)
ax1.set_title('power distribution before optimizing')

# target output complex amplitude
output_target = np.ones((N,1))

# define an optimizer over the triangular mesh
TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)

# optimize the mesh by pushing power to top port and redistributing
TO.optimize(algorithm='up_down')

# look at powers after optimizing
mesh.plot_powers(ax=ax2)
ax2.set_title('power distribution after optimizing')

# create a clements mesh
N = 10
mesh = Mesh(N, mesh_type='clements', initialization='random', M=20)
print(mesh)

# comlex valued input coupling vector
input_values = np.zeros((N, 1))
input_values[-1] = 1
input_values = npr.random((N, 1))

f2, (ax3, ax4) = plt.subplots(2, constrained_layout=True, figsize=(7, 7))

# couple light in and look at powers throughout mesh
mesh.input_couple(input_values)
mesh.plot_powers(ax=ax3)
ax3.set_title('power distribution before optimizing')

# target output complex amplitude
output_target = np.ones((N, 1))

# define an optimizer over the triangular mesh
CO = ClementsOptimizer(mesh, input_values=input_values, output_target=output_target)

# optimize the mesh by pushing power to top port and redistributing
CO.optimize(algorithm='basic')

# look at powers after optimizing
mesh.plot_powers(ax=ax4)
ax4.set_title('power distribution after optimizing')
plt.show()