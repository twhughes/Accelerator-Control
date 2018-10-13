import unittest
from numpy.linalg import det
import numpy.random as npr
import numpy as np

from DLA_Control import MZI, Layer, Mesh

class TestMesh(unittest.TestCase):
    """ Code for testing the MZI mesh"""

    def setUp(self):
        pass

    @staticmethod
    def is_unitary(M):
        M_dagger = M.conj().T
        np.testing.assert_array_almost_equal(np.dot(M, M_dagger), np.eye(M.shape[0]))

    def test_MZI(self):
        """ Tests an invidual MZI"""
        m = MZI()

        print(m)

        self.is_unitary(m.M)

        # make a 2x2 unitary matrix
        phi1 = 2*np.pi*npr.random()
        phi2 = 2*np.pi*npr.random()

        m = MZI(phi1, phi2)
        self.is_unitary(m.M)

    def test_layer(self):
        """ Tests a MZI layer"""
        N = 5
        L = Layer(N)
        m1 = MZI()
        m2 = MZI()

        L.embed_MZI(m1, offset=0)
        L.embed_MZI(m2, offset=2)

        print('\n')
        print(L)

        # check the sparsity pattern of the matrix
        matrix_pattern = 1.*(np.abs(L.M)>0)
        np.testing.assert_array_almost_equal(matrix_pattern, np.array([[1, 1, 0, 0, 0],
                                                                       [1, 1, 0, 0, 0],
                                                                       [0, 0, 1, 1, 0],
                                                                       [0, 0, 1, 1, 0],
                                                                       [0, 0, 0, 0, 1]]))

        # make sure its still unitary
        self.is_unitary(L.M)

        # make sure resetting works
        L.reset_MZI(offset=2, phi1=0, phi2=0)

        np.testing.assert_array_almost_equal(L.M[2:4, 2:4], np.array(np.eye(2)))

    def test_mesh(self):
        """ prints out some meshes for debugging MZI mesh"""

        # triangular mesh
        print('')
        N = 10
        M = Mesh(N, mesh_type='triangular', initialization='random', M=None)
        print('Triangular, N = {}, M = None:'.format(N))
        print(M)

        # full clements mesh
        M = Mesh(N, mesh_type='clements', initialization='random', M=None)
        print('Clements, N = {}, M = None:'.format(N))
        print(M)

        # clements mesh with custom number of layers
        M_temp = 50
        M = Mesh(N-1, mesh_type='clements', initialization='random', M=M_temp)
        print('Clements, N = {}, M = {}:'.format(N-1, M_temp))

        # check if unitary with random initialization
        self.is_unitary(M.full_matrix)

        N = 4
        # check if identity matrix with zero initialization
        M = Mesh(N, mesh_type='clements', initialization='zeros', M=1)
        np.testing.assert_array_almost_equal(M.full_matrix, np.eye(N, dtype=np.complex64))

if __name__ == '__main__':
    unittest.main()
