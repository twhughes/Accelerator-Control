import unittest
from numpy.linalg import det
import numpy.random as npr
import numpy as np

from DLA_Control import MZI

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
        M = MZI.make_M(phi1, phi2)

        # assert it's 2x2
        assert M.shape == (2, 2)

        # assert |det(M)| = 1
        self.assertAlmostEqual(np.abs(det(M)), 1)

        # check if it's unitary
        self.is_unitary(M)

        M = MZI.make_M(0, 0)
        np.testing.assert_array_almost_equal(M, np.eye(2))

if __name__ == '__main__':
    unittest.main()
