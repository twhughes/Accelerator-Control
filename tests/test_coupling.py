import unittest
from numpy.linalg import norm
import numpy.random as npr
import numpy as np
from numpy.testing.utils import assert_array_compare
import operator
from DLA_Control.utils import power_tot, power_vec, normalize_vec

from DLA_Control import Mesh
from DLA_Control import TriangleOptimizer

class TestCoupling(unittest.TestCase):
    """ Code for testing the MZI controllers"""
    def setUp(self):

        self.N = 8
        self.mesh_t = Mesh(self.N, mesh_type='triangular', initialization='random', M=None)
        self.mesh_c_r = Mesh(self.N, mesh_type='clements', initialization='random', M=None)
        self.mesh_c_z = Mesh(self.N, mesh_type='clements', initialization='zeros', M=None)

    def test_IO(self):
        """ Tests basic coupling"""

        # first on the zeros initialized clements mesh

        input_values = npr.random((self.N,))
        self.mesh_c_z.input_couple(input_values)    
        output_values = self.mesh_c_z.output_values

        # because the matrix was initialized with zeros, output should be identical to input
        np.testing.assert_array_almost_equal(input_values, output_values)        

        # same goes for the partial values within MZI
        for partial_value in self.mesh_c_z.partial_values:
            np.testing.assert_array_almost_equal(input_values, partial_value)

        # then on the random initialized clements mesh

        input_values = npr.random((self.N,))
        self.mesh_c_r.input_couple(input_values)    
        output_values = self.mesh_c_r.output_values

        # should not be equal for random initialization
        assert_array_compare(operator.__ne__, input_values, output_values)

        # same goes for the partial values within MZI
        for partial_value in self.mesh_c_r.partial_values:
            assert_array_compare(operator.__ne__, input_values, output_values)

        # finally on the random initialized triangular mesh

        input_values = npr.random((self.N,))
        self.mesh_t.input_couple(input_values)    
        output_values = self.mesh_t.output_values

        # should not be equal for random initialization
        assert_array_compare(operator.__ne__, input_values, output_values)

        # same goes for the partial values within MZI
        for partial_value in self.mesh_t.partial_values:
            assert_array_compare(operator.__ne__, input_values, output_values)

    def test_optimize_triangular(self):

        input_values = np.zeros((self.N,1))
        input_values[-1] = 1

        output_target = np.ones((self.N,1))/self.N

        TO = TriangleOptimizer(self.mesh_t, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='top_down')

if __name__ == '__main__':
    unittest.main()
