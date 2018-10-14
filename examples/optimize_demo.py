import numpy as np

from DLA_Control import Mesh
from DLA_Control import TriangleOptimizer

# create a triangular mesh
N = 10
mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

# comlex valued input coupling vector
input_values = np.zeros((N,1))
input_values[-1] = 1

# couple light in and look at powers throughout mesh
print('before optimizing')
mesh.input_couple(input_values)
mesh.plot_powers()

# target output complex amplitude
output_target = np.ones((N,1))

# define an optimizer over the triangular mesh
TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)

# optimize the mesh by pushing power to top port and redistributing
TO.optimize(algorithm='top_down')

# look at powers after optimizing
print('after optimizing')
mesh.plot_powers()