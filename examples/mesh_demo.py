from DLA_Control.mesh import Mesh

# simple demo of mesh creation and manipulation

# first a triangular mesh
N = 10
mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)
print('Triangular, N = {}, M = None:'.format(N))
print(mesh)
print('')

# then, a full clements mesh
mesh = Mesh(N, mesh_type='clements', initialization='random', M=None)
print('Clements, N = {}, M = None:'.format(N))
print(mesh)
print('')

# finally, a partial clements mesh
N = 3
M = 2   # depth
mesh = Mesh(N, mesh_type='clements', initialization='random', M=M)
print('Clements, N = {}, M = {}:'.format(N, M))
print(mesh)
print('')

# mesh objects contain several 'layers'
print('the first layer of mesh:')
print(mesh.layers[0])
print('')

print('the second layer of mesh:')
print(mesh.layers[1])
print('')

# they hold their partial transfer matrices to each layer
print('the transfer matrix after 0th (no) layers:')
print(mesh.partial_matrices[0])
print('')

# they hold their partial transfer matrices to each layer
print('the transfer matrix after the 1st layer:')
print(mesh.partial_matrices[1])
print('')

print('the transfer matrix after the 2nd layer:')
print(mesh.partial_matrices[2])
print('')

# they also hold their full transfer matrix
print('the full transfer matrix of mesh:')
print(mesh.full_matrix)
print('')

# mesh initialization can be either 'random' or 'zeros'
N = 4
mesh = Mesh(N, mesh_type='triangular', initialization='zeros', M=None)
print('zeros initialization leads to an identity transfer matrix:')
print(mesh.full_matrix)
print('')