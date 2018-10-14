import numpy as np
from numpy.linalg import solve, pinv
import scipy.optimize

from DLA_Control.utils import power_tot, power_vec, normalize_vec, normalize_pow, MSE
from DLA_Control import Layer, MZI

class Optimizer:

    def __init__(self, mesh, input_values, output_target):
        self.mesh = mesh
        self.input_values = normalize_vec(input_values)
        self.output_target = normalize_vec(output_target)
        self.N = mesh.N
        self.M = mesh.M

    def optimizer(self):
        pass

    @staticmethod
    def tune_layer(L, input_values, desired_power, verbose=False):
        # tunes a single layer to aceive the desired output power

        # offset_map[i] gives the offset of the ith MZI
        offset_map = []
        phi_list = []
        for offset, mzi in L.mzi_map.items():
            offset_map.append(offset)
            phi_list.append(mzi.phi1)
            phi_list.append(mzi.phi2)            
        # phis[2*i] gives phi1 of the ith MZI
        # phis[2*i+1] gives phi2 of the ith MZI

        phis = np.array(phi_list)

        def construct_layer(offset_map, phis, N):
            # construct a matrix given a set of offsets and phase shifters
            num_mzis = len(offset_map)
            L = Layer(N)
            for i in range(num_mzis):
                offset = offset_map[i]
                phi1 = phis[2*i]
                phi2 = phis[2*i+1]
                mzi = MZI(phi1, phi2)
                L.embed_MZI(mzi, offset)
            return L

        def objfn(phis, *args):

            offset_map, input_values, desired_power = args
            L = construct_layer(offset_map, phis, N=input_values.size)
            matrix = L.M
            out_values = np.dot(matrix, input_values)
            out_power = power_vec(out_values)

            # return MSE with desired
            return MSE(out_power, desired_power)

        args = offset_map, input_values, desired_power
        phis_optimal =  scipy.optimize.fmin(objfn, phis, args=args, disp=verbose)
        new_layer = construct_layer(offset_map, phis_optimal, N=input_values.size)

        return new_layer


class TriangleOptimizer(Optimizer):


    def optimize(self, algorithm='up_down', verbose=False):

        self.MSE_list = []
        self.mesh.input_couple(self.input_values)

        if algorithm == 'up_down':
            self.optimize_top_down(verbose=verbose)
        elif algorithm == 'spread':
            self.optimize_spread(verbose=verbose)
        else:
            raise ValueError('algorithm "{}" not recognized'.format(algorithm))

    def optimize_top_down(self, verbose=False):
        # optimizes a triangular mesh by pushing power to the top port first

        # loop from layers bottom to top
        for layer_index in range(self.M//2):

            values_prev = self.mesh.partial_values[layer_index]
            layer = self.mesh.layers[layer_index]
            desired_out_power = np.zeros((self.N, 1))
            desired_out_power[self.N-layer_index-2] = 1

            new_layer = self.tune_layer(L=layer, input_values=values_prev,
                                        desired_power=desired_out_power, verbose=verbose)

            self.mesh.layers[layer_index] = new_layer
            self.mesh.recompute_matrices()
            self.mesh.input_couple(self.input_values)
        
        # loop from top to bottom        
        for layer_index in range(self.M//2, self.M):
            values_prev = self.mesh.partial_values[layer_index]
            layer = self.mesh.layers[layer_index]

            index_over = layer_index - self.M//2
            desired_out_power = np.zeros((self.N, 1))
            output_target_pow = power_vec(self.output_target)

            desired_out_power[index_over] = output_target_pow[index_over]
            desired_out_power[index_over+1] = 1-np.sum(output_target_pow[:index_over+1])

            new_layer = self.tune_layer(L=layer, input_values=values_prev,
                                        desired_power=desired_out_power, verbose=verbose)

            self.mesh.layers[layer_index] = new_layer
            self.mesh.recompute_matrices()
            self.mesh.input_couple(self.input_values)


    def optimize_spread(self, verbose=False):
        # NEEDS WORK!
        # optimizes a triangular mesh by spreading power when possible

        # z is the projected coupling from each measuring point to each output
        output_target_pow = power_vec(self.output_target)
        P0 = output_target_pow[0]

        # loop through layers
        for layer_index in range(self.M//2):

            values_prev = self.mesh.partial_values[layer_index]
            powers_prev = power_vec(values_prev)
            layer = self.mesh.layers[layer_index]

            port_index = self.N - layer_index - 1
            print('layer: ', layer_index)
            print('port: ', port_index)
            print('pos: ', port_index-1, self.N-1)
            print('neg: ', port_index+1, self.N-1)

            P_req = np.sum(output_target_pow[port_index-1:self.N-1])
            P_req -= np.sum(powers_prev[port_index+1:self.N-1])
            print(np.sum(powers_prev[port_index+1:self.N-1]))
            m_layer = min(P_req, (1-P0)/(self.N-1))
            desired_out_power = powers_prev
            desired_out_power[self.N-layer_index-1] = m_layer
            desired_out_power[self.N-layer_index-2] =  np.sum(powers_prev) - m_layer

            new_layer = self.tune_layer(L=layer, input_values=values_prev,
                                        desired_power=desired_out_power, verbose=verbose)

            self.mesh.layers[layer_index] = new_layer
            self.mesh.recompute_matrices()
            self.mesh.input_couple(self.input_values)
        
        desired_out_power = output_target_pow
        for layer_index in range(self.M//2, self.M):
            values_prev = self.mesh.partial_values[layer_index]
            powers_prev = power_vec(values_prev)            
            layer = self.mesh.layers[layer_index]

            index_over = layer_index - self.M//2
            desired_out_power = np.zeros((self.N, 1))            
            desired_out_power[index_over] = output_target_pow[index_over]
            desired_out_power[index_over+1] = 1-np.sum(output_target_pow[:index_over+1])

            new_layer = self.tune_layer(L=layer, input_values=values_prev,
                                        desired_power=desired_out_power, verbose=verbose)

            self.mesh.layers[layer_index] = new_layer
            self.mesh.recompute_matrices()
            self.mesh.input_couple(self.input_values)

    @staticmethod
    def construct_coupling_estimate(N, a=0.5):
        Z = np.zeros((N, N))
        Z[0, 0] = 1
        for row_ind in range(1, N-1):
            Z[row_ind, 1] = a**(row_ind)
        for col_ind in range(2, N):
            for row_ind in range(col_ind - 1, N-1):
                Z[row_ind, col_ind] = a**(row_ind - col_ind + 2)
        Z[-1, 1:] = Z[-2, 1:]    
        return Z

