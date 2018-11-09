import numpy as np
import numpy.random as npr
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

from DLA_Control import Mesh
from DLA_Control import TriangleOptimizer, ClementsOptimizer
from DLA_Control.plots import plot_bar_3d, colorbar, plot_powers, apply_sublabels

NR = 1
NC = 6

fig, (axes) = plt.subplots(nrows=NR, ncols=NC)#, figsize=(8.5, 5))

# # create a clements mesh
N = 30
M = 10

vmax = 3/N

uniform_vals = np.ones((N, 1))
mesh = Mesh(N, mesh_type='clements', initialization='random', M=M)
print(mesh)

for i, ax in enumerate(axes.flat):
    print("working on {}/{}".format(i+1, NR*NC))

    random_vals = npr.random((N, 1))

    mesh = Mesh(N, mesh_type='clements', initialization='random', M=M)

    CO = ClementsOptimizer(mesh, input_values=random_vals, output_target=uniform_vals)
    CO.optimize(algorithm='smart', verbose=False)

    im = plot_powers(mesh, ax=ax)
    im.set_clim(0,vmax)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.set_title('')


# Saving / Displaying

cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.ax.tick_params(labelsize=16)

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

# apply_sublabels(axes.flat, invert_color_inds=NC*NR*[True], x=-2, y=-2, size='large', ha='right', va='top', prefix='', postfix='', weight='bold')

plt.savefig('./img/fig4.pdf')
plt.show()

