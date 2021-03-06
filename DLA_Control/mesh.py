import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import matplotlib.pylab as plt

from DLA_Control.utils import power_tot, power_vec, normalize_vec

class MZI:

    def __init__(self, phi1=None, phi2=None):

        if phi1 is None:
            phi1 = 2*np.pi*npr.random()
        if phi2 is None:
            phi2 = 2*np.pi*npr.random()

        self._phi1 = phi1
        self._phi2 = phi2
        self.M = self.make_M(self._phi1, self._phi2)

    def __repr__(self):
        return 'MZI with \n\tphi1 = {}\n\tphi2 = {}\n\tmatrix = \n{}'.format(self._phi1, self._phi2, self.M)

    @staticmethod
    def make_M(phi1, phi2):
        # returns a 2x2 tunable beam splitter (MZI) transfer matrix
        # given the settings of the two integrated phase shifters: phi1, phi2
        M = np.zeros((2, 2), dtype=complex)
        M[0, 0] = np.cos(phi2/2)
        M[1, 0] = -np.sin(phi2/2)
        M[0, 1] = np.exp(-1j*phi1/2)*np.sin(phi2/2)
        M[1, 1] = np.exp(-1j*phi1/2)*np.cos(phi2/2)
        M = np.exp(1j*phi1)*np.exp(1j*phi2/2)*M
        return M

    @property
    def phi1(self):
        return self._phi1

    @phi1.setter
    def phi1(self, phi_new):
        self._phi1 = phi_new
        self.M = self.make_M(self._phi1, self._phi2)

    @property
    def phi2(self):
        return self._phi2

    @phi2.setter
    def phi2(self, phi_new):
        self._phi2 = phi_new
        self.M = self.make_M(self._phi1, self._phi2)


class Layer:
    """ One layer of the MZI """

    def __init__(self, N):
        self.N = N
        self.mzis = self.N*[None]   # for printing only
        self.mzi_map = {}           # maps offset to mzi object
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
        self.mzi_map[offset] = mzi
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
            initialization:  {'random','zeros','real'} how the mzi phases start
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
            self.M = 2*N - 3
        else:
            raise ValueError("'mesh_type' must be one of {'clements','triangular'}")

        # set up the matrices
        # full transfer matrix for the mesh
        self.full_matrix = np.eye(N, dtype=np.complex64)
        # list of partial transfer matrices (partial_matrices[i] is the transfer matrix UP TO the ith layer)
        self.partial_matrices = [np.eye(N, dtype=np.complex64)]

        # construct the mesh
        self.construct_mesh()
        self.coupled = False  # whether light has been coupled in

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

    def append_layer(self, L):
        """ Adds a new layer to the mesh and adds to the list of partial matrices"""
        self.layers.append(L)
        new_partial_M = np.dot(L.M, self.partial_matrices[-1])
        self.partial_matrices.append(new_partial_M)
        self.full_matrix = self.partial_matrices[-1]

    def recompute_matrices(self):
        """ Computes all of the partial matrices (and full matrix) """
        self.partial_matrices = [np.eye(self.N, dtype=np.complex64)]
        for L in self.layers:
            new_partial_M = np.dot(L.M, self.partial_matrices[-1])
            self.partial_matrices.append(new_partial_M)
        self.full_matrix = self.partial_matrices[-1]

    def _mzi_init(self):
        """ Returns an MZI based on the mesh initialization """
        if self.initialization == 'random':
            return MZI()
        elif self.initialization == 'zeros':
            return MZI(0, 0)
        elif self.initialization == 'real':
            mzi = MZI()
            mzi.phi1 = 0
            return mzi
        else:
            raise ValueError("incorrect initialization, must be one of 'random','zeros','real', given {}".format(initialization))

    def construct_mesh(self):
        """ Makes the mesh """
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
                    mzi = self._mzi_init()
                    L.embed_MZI(mzi, offset=port_index)
                self.append_layer(L)

        elif self.mesh_type == 'triangular':
            for port_index in range(self.N-2, 0, -1):
                L = Layer(self.N)
                mzi = self._mzi_init()
                L.embed_MZI(mzi, offset=port_index)
                self.append_layer(L)
            for port_index in range(self.N-1):
                L = Layer(self.N)
                mzi = self._mzi_init()               
                L.embed_MZI(mzi, offset=port_index)
                self.append_layer(L)

    def input_couple(self, input_values):
        """ Specify input coupling (complex) values to mesh.
            Compute the fields at each layer of the structure.
            And at the output of the structure  """        
        self.input_values = normalize_vec(input_values)
        self.partial_values = []
        for p_mat in self.partial_matrices:
            layer_value = np.dot(p_mat, self.input_values)
            self.partial_values.append(layer_value)
        self.output_values = np.dot(self.full_matrix, self.input_values)
        self.coupled = True

    def get_layer_powers(self, layer_index):
        # returns the power right BEFORE layer index (0 = input)
        if not self.coupled:
            raise ValueError("must run `Mesh.input_couple(input_values)` before getting layer powers")
        partial_values = self.partial_values[layer_index]
        return power_vec(partial_values)
