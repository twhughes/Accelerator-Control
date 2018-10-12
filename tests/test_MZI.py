import unittest
from numpy.linalg import det
import numpy.random as npr
import numpy as np

from DLA_Control.linalg import (make_M, make_layer_matrix, make_partial_matrix,
    make_full_matrix)


class TestMZI(unittest.TestCase):
    """ Code for testing the MZI mesh"""
    def setUp(self):
        pass

    @staticmethod
    def is_unitary(M):
        M_dagger = M.conj().T
        np.testing.assert_array_almost_equal(np.dot(M, M_dagger), np.eye(M.shape[0]))

    def test_2x2(self):

        # make a 2x2 unitary matrix
        phi1 = 2*np.pi*npr.random()
        phi2 = 2*np.pi*npr.random()
        M = make_M(phi1, phi2)

        # assert it's 2x2
        assert M.shape == (2, 2)

        # assert |det(M)| = 1
        self.assertAlmostEqual(np.abs(det(M)), 1)

        # check if it's unitary
        self.is_unitary(M)

    def test_layer_matrix(self):

        # make a 2x2 unitary matrix
        phi1 = 2*np.pi*npr.random()
        phi2 = 2*np.pi*npr.random()
        N = 10
        layer_index = 4
        M = make_layer_matrix(N, layer_index, phi1, phi2)

        # check if it's unitary
        self.is_unitary(M)

    def test_partial_matrix(self):
        N = 10
        layer_index = 2*N-5
        phi_list = npr.random((2*N-3, 2))
        M = make_partial_matrix(N, phi_list, layer_index=layer_index)
        self.is_unitary(M)

        phi_list = npr.random((2*N-3, 2))
        M = make_partial_matrix(N, phi_list, layer_index=-1)
        self.is_unitary(M)

    def test_full_matrix(self):
        N = 10
        layer_index = 2*N-5
        phi_list = npr.random((2*N-3, 2))
        M = make_full_matrix(N, phi_list)
        self.is_unitary(M)


if __name__ == '__main__':
    unittest.main()
