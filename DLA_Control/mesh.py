import numpy as np
import numpy.random as npr
import scipy.sparse as sp

from .linalg import make_M, make_layer_matrix, make_partial_matrix, make_full_matrix


class MZI:

    def __init__(self, phi1=None, phi2=None):

        if phi1 is None:
            phi1 = 2*np.pi*npr.random()
        if phi2 is None:
            phi2 = 2*np.pi*npr.random()

        self._phi1 = phi1
        self._phi2 = phi2
        self.M = make_M(self._phi1, self._phi2)

    def __repr__(self):
        return 'MZI with \n\tphi1 = {}\n\tphi2 = {}\n\tmatrix = \n{}'.format(self._phi1, self._phi2, self.M)

    @property
    def phi1(self):
        return self._phi1

    @phi1.setter
    def phi1(self, phi_new):
        self._phi1 = phi_new
        self.M = make_M(self._phi1, self._phi2)

    @property
    def phi2(self):
        return self._phi2

    @phi2.setter
    def phi2(self, phi_new):
        self._phi2 = phi_new
        self.M = make_M(self._phi1, self._phi2)


class Layer:
    """ One layer of the MZI """

    def __init__(self, N):
        self.N = N
        self.mzis = self.N*[None]
        self.M = np.eye(self.N, dtype=np.complex64)  # make sparse later?

    def get_layer_string(self):
        layer_str = []
        for mzi in self.mzis:
            if mzi is not None and mzi != 'skip':
                layer_str.append('-v-')
            elif mzi == 'skip':
                layer_str.append('-^-')                
            else:
                layer_str.append('---')
        return '\n'.join(layer_str)

    def __repr__(self):
        layer_str = self.get_layer_string()
        return 'layer with \n\tmzis = {}\n\tmatrix pattern = \n{}'.format(layer_str, 1.*(np.abs(self.M) > 0))

    def __str__(self):
        return self.get_layer_string()

    def embed_MZI(self, mzi, offset):
        """ offset of 0 means MZI connecting top two waveguides, max offset = N - 2 """
        self.M[offset:offset + 2, offset:offset + 2] = mzi.M
        self.mzis[offset] = mzi
        self.mzis[offset+1] = 'skip'

    def reset_MZI(self, offset, phi1, phi2):
        m = MZI(phi1, phi2)
        self.embed_MZI(m, offset)


class Mesh:
    """ A full MZI mesh, containing several layers"""

    def __init__(self, N, mesh_type='Clements', initialization='random', M=None):
        """
            N:  the number of inputs & outputs
            mesh_type:  {'clements','triangular'} the pattern of the MZI in the mesh
            initialization:  {'random','zeros'} how the mzi phases start
        """

        self.N = N
        self.mesh_type = mesh_type.lower()
        self.initialization = initialization
        self.layers = []

        # if using a clements mesh type
        if self.mesh_type == 'clements':
            # if the layer depth M is not specified, default to full mesh
            if M is None:
                self.M = N
            else:
                self.M = M
        # triangular mesh type has a fixed depth
        elif self.mesh_type == 'triangular':
            self.M = 2*N - 1
        else:
            raise ValueError("'mesh_type' must be one of {'clements','triangular'}")


        # set up the matrices
        self.full_matrix = np.eye(N, dtype=np.complex64)
        self.partial_matrices = []

        # construct the mesh
        self.construct_mesh()

    def __repr__(self):
        """ prints the mesh """
        port_strings = [['' for _ in range(self.M)] for _ in range(self.N)]
        for layer_index, L in enumerate(self.layers):
            layer_string = str(L).split('\n')[:]
            for port_index, string in enumerate(layer_string):
                port_strings[port_index][layer_index] = string
        display_string = ''
        for port_string in port_strings:
            display_string += ''.join(port_string) + '\n'
        return display_string

    def add_layer(self, L):
        """ Adds a new layer to the mesh and computes the partial matrices"""
        self.layers.append(L)
        if not self.partial_matrices:
            self.partial_matrices.append(L.M)
        else:
            new_partial_M = np.dot(self.partial_matrices[-1], L.M)
            self.partial_matrices.append(new_partial_M)
        self.full_matrix = self.partial_matrices[-1]

    def construct_mesh(self):

        if self.mesh_type == 'clements':
            for layer_index in range(self.M):
                L = Layer(self.N)
                if layer_index % 2 == 0:
                    port_indeces = range(0, self.N-1, 2)
                else:
                    if self.N % 2 == 0:
                        port_indeces = range(1, self.N-2, 2)
                    else:
                        port_indeces = range(1, self.N, 2)                        
                for port_index in port_indeces:
                    if self.initialization == 'random':
                        mzi = MZI()   # random MZI
                    else:
                        mzi = MZI(0, 0)
                    L.embed_MZI(mzi, offset=port_index)
                self.add_layer(L)

        elif self.mesh_type == 'triangular':
            for port_index in range(self.N-2, 0, -1):
                L = Layer(self.N)
                if self.initialization == 'random':
                    mzi = MZI()   # random MZI
                else:
                    mzi = MZI(0, 0)
                L.embed_MZI(mzi, offset=port_index)
                self.add_layer(L)
            for port_index in range(self.N-1):
                L = Layer(self.N)
                if self.initialization == 'random':
                    mzi = MZI()   # random MZI
                else:
                    mzi = MZI(0, 0)                
                mzi = MZI()   # random MZI
                L.embed_MZI(mzi, offset=port_index)
                self.add_layer(L)
