from DLA_Control.mesh import Mesh

# simple demo of mesh creation and manipulation

# first a triangular mesh
N = 10
mesh1 = Mesh(N, mesh_type='triangular', initialization='random', M=None)
print('Triangular, N = {}, M = None:'.format(N))
print(mesh1)

# then, a full clements mesh
mesh2 = Mesh(N, mesh_type='clements', initialization='random', M=None)
print('Clements, N = {}, M = None:'.format(N))
print(mesh2)

# finally, a partial clements mesh
M = 5   # depth
mesh2 = Mesh(N, mesh_type='clements', initialization='random', M=M)
print('Clements, N = {}, M = {}:'.format(N, M))
print(mesh2)

