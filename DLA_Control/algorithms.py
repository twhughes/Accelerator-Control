import numpy as np
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


    def optimize(self, algorithm='top_down'):

        if algorithm == 'top_down':
            self.optimize_top_down()


    def optimize_top_down(self):
        # optimizes a triangular mesh by pushing power to the top port first

        self.MSE_list = []
        self.mesh.input_couple(self.input_values)

        # loop through layers
        for layer_index in range(self.M//2):

            values_prev = self.mesh.partial_values[layer_index]
            layer = self.mesh.layers[layer_index]
            desired_out_power = np.zeros((self.N, 1))
            desired_out_power[self.N-layer_index-2] = 1

            new_layer = self.tune_layer(L=layer, input_values=values_prev,
                                        desired_power=desired_out_power, verbose=False)

            self.mesh.layers[layer_index] = new_layer
            self.mesh.recompute_matrices()
            self.mesh.input_couple(self.input_values)
        
        for layer_index in range(self.M//2, self.M):
            values_prev = self.mesh.partial_values[layer_index]
            layer = self.mesh.layers[layer_index]

            index_over = layer_index - self.M//2
            desired_out_power = np.zeros((self.N, 1))
            output_target_pow = power_vec(self.output_target)

            desired_out_power[index_over] = output_target_pow[index_over]
            desired_out_power[index_over+1] = 1-np.sum(output_target_pow[:index_over+1])

            new_layer = self.tune_layer(L=layer, input_values=values_prev,
                                        desired_power=desired_out_power, verbose=False)

            self.mesh.layers[layer_index] = new_layer
            self.mesh.recompute_matrices()
            self.mesh.input_couple(self.input_values)

