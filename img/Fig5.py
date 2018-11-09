import numpy as np
import numpy.random as npr
import matplotlib.pylab as plt

import DLA_Control
from DLA_Control import Mesh
from DLA_Control import TriangleOptimizer, ClementsOptimizer
from DLA_Control.plots import plot_bar_3d, colorbar, plot_powers
from DLA_Control.utils import MSE, power_vec, normalize_vec

"""
This script is used to gain intuition about how many layers of Clements mesh are needed
for sorting out optical power in DLA applications.
It optimizes Clements meshes of several sizes for random -> uniform coupling.
Then computes the MSE at each layer between power and target, averaged over several random inputs.
"""

N_avg = 5    # number of inputs to average over
N_max = 200    # largest clements mesh
M = N_max
N_list = range(2, N_max, 20)      # list of clements mesh sizes to try
tol = 1e-4    # mse tolerance for "converged"

# stores the convergences.
convergences = np.zeros((len(N_list), M))

# for each mesh size
for N_index, N in enumerate(N_list):

    # construct an NxN clements mesh.
    mesh = Mesh(N, mesh_type='clements', initialization='random', M=M)
    print('N = {} / {}:'.format(N, N_max))
    # print(mesh)

    # uniform output target
    output_target = np.ones((N,))
    target_power = power_vec(normalize_vec(output_target))

    # store MSE of each layer in each run in averaging
    mses = M*[0.]

    # for N_avg random inputs (to average over)
    for _ in range(N_avg):

        # make random inputs and couple them in
        input_values = npr.random((N, 1))
        mesh.input_couple(input_values)

        # make a clements mesh and optimize using the 'smart' algorithm
        CO = ClementsOptimizer(mesh, input_values=input_values, output_target=output_target)
        CO.optimize(algorithm='smart_seq', verbose=False)

        # loop through layers of optimized device
        for li in range(M):

            # get the power in layer i and compute MSE with target
            pow_layer_i = mesh.get_layer_powers(layer_index=li)            
            mse = MSE(pow_layer_i, target_power)

            # add to sums
            # print(mse_sums)
            mses[li] += mse/N_avg

    # average the MSE sums, add to convergences array
    convergences[N_index, :M] = mses

im = plt.pcolormesh(range(M), N_list, np.log(convergences), cmap='magma')
im.set_rasterized(True)
plt.title('log$_{10}$(MSE) betwen layer power and target.')
plt.xlabel('number of layers')
plt.ylabel('number of ports')
cbar = colorbar(im)
plt.show()
