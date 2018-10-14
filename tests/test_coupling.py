import unittest
from numpy.linalg import norm
import numpy.random as npr
import numpy as np
from numpy.testing.utils import assert_array_compare
import operator
import matplotlib.pylab as plt

from DLA_Control.utils import power_tot, power_vec, normalize_vec

from DLA_Control import Mesh
from DLA_Control import TriangleOptimizer, ClementsOptimizer

class TestCoupling(unittest.TestCase):
    """ Code for testing the MZI controllers"""
    def setUp(self):

        self.N = 50
        self.mesh_t = Mesh(self.N, mesh_type='triangular', initialization='random', M=None)
        self.mesh_c_r = Mesh(self.N, mesh_type='clements', initialization='random', M=None)
        self.mesh_c_z = Mesh(self.N, mesh_type='clements', initialization='zeros', M=None)

        self.one = normalize_vec(np.ones((self.N, 1)))
        self.top = np.zeros((self.N, 1))
        self.top[0] = 1
        self.mid = np.zeros((self.N, 1))
        self.mid[self.N//2] = 1
        self.bot = np.zeros((self.N, 1))
        self.bot[-1] = 1

    def test_IO(self):
        """ Tests basic coupling"""

        # first on the zeros initialized clements mesh

        input_values = normalize_vec(npr.random((self.N,1)))
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

        input_values = normalize_vec(npr.random((self.N,1)))
        self.mesh_t.input_couple(input_values)    
        output_values = self.mesh_t.output_values

        # should not be equal for random initialization
        assert_array_compare(operator.__ne__, input_values, output_values)

        # same goes for the partial values within MZI
        for partial_value in self.mesh_t.partial_values:
            assert_array_compare(operator.__ne__, input_values, output_values)

    def check_power(self, mesh, output_target):
        # checks if output equals target
        EPSILON = 1e-5
        P_out = power_vec(mesh.output_values)
        P_target = power_vec(output_target)
        error = norm(P_out - P_target)/self.N
        self.assertLess(error, EPSILON)


    def test_clements(self):
        N = self.N
        mesh = Mesh(N, mesh_type='clements', initialization='random', M=100)

        input_values = self.bot
        input_values = npr.random((N, 1))
        output_target = self.one

        CO = ClementsOptimizer(mesh, input_values=input_values, output_target=output_target)
        CO.optimize(algorithm='basic')

        ax = mesh.plot_powers()
        plt.show()
        self.check_power(mesh, output_target)

#### VARIOUS INPUTS TO UNIFORM OUTPUTS ####

    def test_updown_1_1(self):
        # uniform to uniform
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.one
        output_target = self.one

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')
        
        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_bot_1(self):
        # bottom input to uniform
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.bot
        output_target = self.one

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_mid_1(self):
        # middle input to uniform
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.mid
        output_target = self.one

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_top_1(self):
        # top input to uniform
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.top
        output_target = self.one

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_rand_1(self):
        # random input to uniform
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = npr.random((N,1))
        output_target = self.one

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

#### VARIOUS INPUTS TO BOTTOM OUTPUTS ####

    def test_updown_1_bot(self):
        # uniform to bottom
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.one
        output_target = self.bot

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_bot_bot(self):
        # bottom input to bottom
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.bot
        output_target = self.bot

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_mid_bot(self):
        # middle input to bottom
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.mid
        output_target = self.bot

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_top_bot(self):
        # top input to bottom
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.top
        output_target = self.bot

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_rand_bot(self):
        # random input to bottom
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = npr.random((N,1))
        output_target = self.bot

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

#### VARIOUS INPUTS TO MIDDLE OUTPUTS ####

    def test_updown_1_mid(self):
        # uniform to middle
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.one
        output_target = self.mid

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_bot_mid(self):
        # bottom input to middle
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.bot
        output_target = self.mid

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_mid_mid(self):
        # middle input to middle
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.mid
        output_target = self.mid

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_top_mid(self):
        # top input to middle
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.top
        output_target = self.mid

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_rand_mid(self):
        # random input to middle
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = npr.random((N,1))
        output_target = self.mid

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

#### VARIOUS INPUTS TO TOP OUTPUTS ####

    def test_updown_1_top(self):
        # uniform to top
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.one
        output_target = self.top

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_bot_top(self):
        # bottom input to top
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.bot
        output_target = self.top

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_mid_top(self):
        # middle input to top
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.mid
        output_target = self.top

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_top_top(self):
        # top input to top
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.top
        output_target = self.top

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_rand_top(self):
        # random input to top
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = npr.random((N,1))
        output_target = self.top

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

#### VARIOUS INPUTS TO RANDOM OUTPUTS ####

    def test_updown_1_rand(self):
        # uniform to rand
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.one
        output_target = normalize_vec(npr.random((N,1)))

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_bot_rand(self):
        # bottom input to rand
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.bot
        output_target = normalize_vec(npr.random((N,1)))

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_mid_rand(self):
        # middle input to rand
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.mid
        output_target = normalize_vec(npr.random((N,1)))

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_top_rand(self):
        # top input to rand
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.top
        output_target = normalize_vec(npr.random((N,1)))

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_updown_rand_rand(self):
        # random input to rand
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = npr.random((N,1))
        output_target = normalize_vec(npr.random((N,1)))

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='up_down')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

#### VARIOUS INPUTS TO UNIFORM OUTPUTS ####

    def test_spread_1_1(self):
        # uniform to uniform
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.one
        output_target = self.one

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')
        
        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_bot_1(self):
        # bottom input to uniform
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.bot
        output_target = self.one

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_mid_1(self):
        # middle input to uniform
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.mid
        output_target = self.one

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_top_1(self):
        # top input to uniform
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.top
        output_target = self.one

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_rand_1(self):
        # random input to uniform
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = npr.random((N,1))
        output_target = self.one

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

#### VARIOUS INPUTS TO BOTTOM OUTPUTS ####

    def test_spread_1_bot(self):
        # uniform to bottom
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.one
        output_target = self.bot

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_bot_bot(self):
        # bottom input to bottom
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.bot
        output_target = self.bot

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_mid_bot(self):
        # middle input to bottom
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.mid
        output_target = self.bot

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_top_bot(self):
        # top input to bottom
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.top
        output_target = self.bot

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_rand_bot(self):
        # random input to bottom
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = npr.random((N,1))
        output_target = self.bot

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

#### VARIOUS INPUTS TO MIDDLE OUTPUTS ####

    def test_spread_1_mid(self):
        # uniform to middle
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.one
        output_target = self.mid

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_bot_mid(self):
        # bottom input to middle
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.bot
        output_target = self.mid

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_mid_mid(self):
        # middle input to middle
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.mid
        output_target = self.mid

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_top_mid(self):
        # top input to middle
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.top
        output_target = self.mid

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_rand_mid(self):
        # random input to middle
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = npr.random((N,1))
        output_target = self.mid

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

#### VARIOUS INPUTS TO TOP OUTPUTS ####

    def test_spread_1_top(self):
        # uniform to top
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.one
        output_target = self.top

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_bot_top(self):
        # bottom input to top
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.bot
        output_target = self.top

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_mid_top(self):
        # middle input to top
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.mid
        output_target = self.top

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_top_top(self):
        # top input to top
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.top
        output_target = self.top

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_rand_top(self):
        # random input to top
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = npr.random((N,1))
        output_target = self.top

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

#### VARIOUS INPUTS TO RANDOM OUTPUTS ####

    def test_spread_1_rand(self):
        # uniform to rand
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.one
        output_target = normalize_vec(npr.random((N,1)))

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_bot_rand(self):
        # bottom input to rand
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.bot
        output_target = normalize_vec(npr.random((N,1)))

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_mid_rand(self):
        # middle input to rand
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.mid
        output_target = normalize_vec(npr.random((N,1)))

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_top_rand(self):
        # top input to rand
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = self.top
        output_target = normalize_vec(npr.random((N,1)))

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

    def test_spread_rand_rand(self):
        # random input to rand
        N = self.N
        mesh = Mesh(N, mesh_type='triangular', initialization='random', M=None)

        input_values = npr.random((N,1))
        output_target = normalize_vec(npr.random((N,1)))

        TO = TriangleOptimizer(mesh, input_values=input_values, output_target=output_target)
        TO.optimize(algorithm='spread')

        # mesh.plot_powers(); plt.show()
        self.check_power(mesh, output_target)

if __name__ == '__main__':
    unittest.main()
