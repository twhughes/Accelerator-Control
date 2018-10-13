import numpy as np

from DLA_Control.utils import power_tot, power_vec

class TriangleOptimizer:

    def __init__(self, mesh, input_values, output_target):
        self.mesh = mesh
        self.input_values = input_values
        self.output_target = output_target
        self.N = mesh.N
        self.M = mesh.M

    def optimize(self):
        # optimizes a triangular mesh
        self.MSE_list = []
        self.mesh.input_couple(self.input_values)
        self.mesh.plot_powers()


