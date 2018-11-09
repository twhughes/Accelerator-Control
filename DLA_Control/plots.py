import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from string import ascii_lowercase

import numpy as np


def plot_powers(mesh, ax=None):
    # plots the powers throughout the mesh (must have run mesh.input_couple() first)
    if not mesh.coupled:
        raise ValueError("must run `Mesh.input_couple(input_values)` before getting layer powers")

    if ax is None:
        fig, ax = plt.subplots(1, constrained_layout=True)

    power_im = np.zeros((mesh.N, mesh.M+1))
    for layer_index in range(0, mesh.M+1):
        power_im[:, layer_index] = mesh.get_layer_powers(layer_index)[:,0]

    ax.set_xlabel('layer index')
    ax.set_ylabel('port index')
    im = ax.imshow(power_im, cmap='magma')
    ax.set_title('power in each layer')
    return im


def plot_bar_3d(power_map, ax=None):
    (N, M) = power_map.shape
    # make bar plot
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.projection = '3d'

    _x = np.arange(N)
    _y = np.arange(M)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    bottom = np.zeros_like(power_map).ravel()
    width = depth = 1
    top = power_map.ravel()
    ax.bar3d(x, y, bottom, width, depth, top, shade=True)

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def set_axis_font(ax, font_size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)

def apply_sublabels(axs, invert_color_inds, x=19, y=-5, size='large', ha='right', va='top', prefix='(', postfix=')', weight=None):
    # axs = list of axes
    # invert_color_ind = list of booleans (True to make sublabels white, else False)
    for n, ax in enumerate(axs):
        if invert_color_inds[n]:
            color='w'
        else:
            color='k'
        ax.annotate(prefix + ascii_lowercase[n] + postfix,
                    xy=(1, 1),
                    xytext=(x, y),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    size=size,
                    color=color,
                    horizontalalignment=ha,
                    verticalalignment=va,
                    weight=weight)
