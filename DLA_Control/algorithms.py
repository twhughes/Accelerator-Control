import numpy as np
from numpy.linalg import solve, pinv
import scipy.optimize
# from progressbar import ProgressBar
from DLA_Control.utils import power_tot, power_vec, normalize_vec, normalize_pow, MSE
from DLA_Control import Layer, MZI

class Optimizer:

    def __init__(self, mesh, input_values, output_target):

        # initialization for both Triangular and Clements Mesh

        self.mesh = mesh
        self.input_values = normalize_vec(input_values)
        self.output_target = normalize_vec(output_target)
        self.N = mesh.N
        self.M = mesh.M

    def optimize(self):
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


class ClementsOptimizer(Optimizer):

    def optimize(self, algorithm='basic', verbose=False):

        self.MSE_list = []
        self.mesh.input_couple(self.input_values)

        if algorithm == 'basic':
            self.optimize_basic(verbose=verbose)
        else:
            raise ValueError('algorithm "{}" not recognized'.format(algorithm))

    def optimize_basic(self, verbose=False):
        # optimizes a clements mesh by attempting to get close to target each layer

        """ BASIC IDEA:
                Go through each layer from left to right.
                At each step, try to get the power output of this layer equal to the 
                eventual target output.
                Once the optimization gives up, move to the next layer.
        """   

        # loop through layers
        # bar = ProgressBar(max_value=self.M)
        for layer_index in range(self.M):
            print('working on layer {} of {}'.format(layer_index, self.M))
            # bar.update(layer_index)

            # get previous powers and layer
            values_prev = self.mesh.partial_values[layer_index]        
            layer = self.mesh.layers[layer_index]

            # desired output powers = target outputs
            D = self.output_target
            
            new_layer = self.tune_layer(L=layer, input_values=values_prev,
                                        desired_power=D, verbose=verbose)

            # insert into the mesh and recompute / recouple
            self.mesh.layers[layer_index] = new_layer
            self.mesh.recompute_matrices()
            self.mesh.input_couple(self.input_values)


class TriangleOptimizer(Optimizer):


    def optimize(self, algorithm='up_down', verbose=False):

        self.mesh.input_couple(self.input_values)

        if algorithm == 'up_down':
            self.optimize_up_down(verbose=verbose)
        elif algorithm == 'spread':
            self.optimize_spread(verbose=verbose)
        else:
            raise ValueError('algorithm "{}" not recognized'.format(algorithm))

    def optimize_up_down(self, verbose=False):
        # optimizes a triangular mesh by two step process

        """ BASIC IDEA:
                With the upward pass, we can push all of the power into the top MZI
                Then, on the downward pass, we can redistribute the power to the output ports as it is needed.
                This is simple and effective, but concentrates power, which isn't good for DLA.
                See 'spread' algorithm for an improvement
        """        

        # loop throgh MZI from bottom layers to top
        for layer_index in range(self.M//2):

            # get the previous field values, the current layer, and the port index
            values_prev = self.mesh.partial_values[layer_index]        
            layer = self.mesh.layers[layer_index]
            port_index = (self.N - 1) - layer_index

            # make desired output vector for this layer 'D'
            # all of the power from this MZI should go to the top port
            D = np.zeros((self.N, 1))
            D[port_index - 1] = 1

            # tune the layer
            new_layer = self.tune_layer(L=layer, input_values=values_prev,
                                        desired_power=D, verbose=verbose)

            # insert into the mesh and recompute / recouple
            self.mesh.layers[layer_index] = new_layer
            self.mesh.recompute_matrices()
            self.mesh.input_couple(self.input_values)
        
        # loop throgh MZI from top layers to bottom
        for layer_index in range(self.M//2, self.M):

            # get the previous field values, the current layer, and the port index
            values_prev = self.mesh.partial_values[layer_index]
            layer = self.mesh.layers[layer_index]
            port_index = layer_index - self.M//2

            # output target powers
            P = power_vec(self.output_target)

            # make desired output vector for this layer 'D'            
            D = np.zeros((self.N, 1))

            # the desired output power for this port is the target output
            D[port_index] = P[port_index]

            # the desired output power for the next port is the remaining power          
            D[port_index+1] = 1-np.sum(P[:port_index + 1])

            # set this layer
            new_layer = self.tune_layer(L=layer, input_values=values_prev,
                                        desired_power=D, verbose=verbose)

            # insert into the mesh and recompute / recouple
            self.mesh.layers[layer_index] = new_layer
            self.mesh.recompute_matrices()
            self.mesh.input_couple(self.input_values)


    def optimize_spread(self, verbose=False):
        # optimizes a triangular mesh by spreading power when possible

        """ BASIC IDEA:
                The problem with up down is the power gets concentrated at the top
                Here, we try to spread the power evenly in the middle section.
                Only push power up if it is needed in the top out ports.
                Otherwise, keep power distributed.
        """

        # target output powers
        P = power_vec(self.output_target)        
        P0 = P[0]

        # input powers
        I = power_vec(self.mesh.partial_values[0])

        # middle section powers
        M = np.zeros((self.N, 1))

        # iterate from bottom to top
        for layer_index in range(self.M//2):

            # get the layer, previous field values, and port index
            layer = self.mesh.layers[layer_index]
            values_prev = self.mesh.partial_values[layer_index]
            port_index = (self.N - 1) - layer_index

            # construct a 'desired' power array for the output of this layer (equal to previous powers to start)
            D = power_vec(values_prev)

            # sum the desired powers that are supplied by this port
            P_sum = np.sum(P[port_index - 1:])

            # sum the existing middle powers that can also contribute
            M_sum = np.sum(D[port_index + 1:])

            # the remaining power to be spread over the middle ports
            P_rem = 1 - P_sum

            # split this remaining power evenly between midle ports
            P_avg = (1 - P0 - M_sum) / (self.N - 1 - layer_index)

            # the output port is the minimum of the average power and the required power
            D[port_index] = min(P_avg, P_sum - M_sum)

            # the lower output port is just the sum of the remaining power            
            D[port_index - 1] = 0
            D[port_index - 1] = 1 - np.sum(D)

            # tune the layer MZI and move on
            new_layer = self.tune_layer(L=layer, input_values=values_prev,
                                        desired_power=D, verbose=verbose)

            self.mesh.layers[layer_index] = new_layer
            self.mesh.recompute_matrices()
            self.mesh.input_couple(self.input_values)
        
        # loop from top down
        for layer_index in range(self.M//2, self.M):

            # get previous values, powers, and layer
            values_prev = self.mesh.partial_values[layer_index]
            powers_prev = power_vec(values_prev)            
            layer = self.mesh.layers[layer_index]

            # computes the port index
            port_index = layer_index - self.M//2

            # desired powers.
            D = np.zeros((self.N, 1))

            # we know the desired power of this port is just the target
            D[port_index] = P[port_index] 

            # the sum of powers into this MZI
            P_in = np.sum(powers_prev[port_index:port_index+2])

            # the other port target power should just be the remaining power
            D[port_index+1] = P_in - P[port_index]

            # tune the layer and move on
            new_layer = self.tune_layer(L=layer, input_values=values_prev,
                                        desired_power=D, verbose=verbose)

            self.mesh.layers[layer_index] = new_layer
            self.mesh.recompute_matrices()
            self.mesh.input_couple(self.input_values)

    """
    @staticmethod
    def construct_coupling_estimate(N, a=0.5):
        # not needed right now
        Z = np.zeros((N, N))
        Z[0, 0] = 1
        for row_ind in range(1, N-1):
            Z[row_ind, 1] = a**(row_ind)
        for col_ind in range(2, N):
            for row_ind in range(col_ind - 1, N-1):
                Z[row_ind, col_ind] = a**(row_ind - col_ind + 2)
        Z[-1, 1:] = Z[-2, 1:]    
        return Z
    """
