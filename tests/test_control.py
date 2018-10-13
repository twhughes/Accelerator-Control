import unittest
from numpy.linalg import det
import numpy.random as npr
import numpy as np

# from controller import ControllerFactory
from DLA_Control import Mesh

class TestController(unittest.TestCase):
    """ Code for testing the MZI controllers"""
    def setUp(self):

        N = 10
        self.mesh_t = Mesh(N, mesh_type='triangular', initialization='random', M=None)
        self.mesh_c = Mesh(N, mesh_type='clements', initialization='random', M=None)

    def test_setup(self):
        pass

if __name__ == '__main__':
    unittest.main()
