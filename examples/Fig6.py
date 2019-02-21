import numpy as np
import matplotlib.pylab as plt

import sys
sys.path.append('../')

from DLA_Control.plots import apply_sublabels

lambda0 = 2e-6   # free space wavelength (m)
beta = 1         # speed of electron / c0
N_pil = 10       # pillers per waveguide
eta = 1          # acceleration efficiency G = eta * E0 (unitless)
DE = 1e6         # energy gain (eV)
Ud = 20e-9       # damage threshold of waveguides (J)
h = 2e-6         # pillar height (m)
tau = 250e-15    # pulse duration (s)
c0 = 3e8         # speed of light (m/s)
e0 = 8.85e-12    # free space permittivity (C/V/m)
n = 2            # refractive index of waveguides
zeta = np.pi/2   # peak field ratio (Gaussian)
eta_s = 0.95     # splitting power transmission efficiency
eta_b = 0.95     # bending power transmission efficiency
eta_mzi = 0.25   # power efficiency of MZI mesh

# number of output waveguides needed to supply DE of energy gain for a splitted structure
N = lambda DE: 2**(np.log2((zeta * DE**2 * tau * h * n * c0 * e0) / (2 * eta**2 * Ud * lambda0 * beta * N_pil)) / np.log2(2 * eta_s * eta_b) )

# the graident of a splitted structure with energy gain DE
gradient = lambda DE: DE / length(DE)

# the length of a splitted structure with energy gain DE
length = lambda DE: N(DE) * N_pil * lambda0 * beta

# the graidnet of a direct couped structure
gradient_direct = eta * np.sqrt((2 * Ud * eta_mzi) / (zeta * tau * N_pil * lambda0 * beta * h * n * e0 * c0))

# range of energies to look at
NE = 10000
DE_range = np.logspace(4, 6, NE)

N_list = np.zeros((NE, ))
gradient_list = np.zeros((NE, ))
length_list = np.zeros((NE, ))

# compute and store data
for i, DE in enumerate(DE_range):
    N_list[i] = N(DE)
    gradient_list[i] = gradient(DE)
    length_list[i] = length(DE)

# print some stats for the highest energy
print('\nfor splitting structure, to accomplish {:.2e} eV of energy gain'.format(DE_range[-1]))
print('\tgradient of {:.2e} V/m'.format(gradient_list[-1]))
print('\tN_wg of {:.2e}'.format(N_list[-1]))
print('\tlength of (meters){:.2e}\n'.format(length_list[-1]))

print('for direct coupling structure, to accomplish {:.2e} eV of energy gain'.format(DE_range[-1]))
print('\tgradient of {:.2e} V/m'.format(gradient_direct))
print('\tN_wg of {:.2e}'.format(np.sqrt(N_list[-1])))
print('\tlength of {:.2e} (meters)\n'.format(DE_range[-1] / gradient_direct))

# make plots for Fig 6

fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
(ax1, ax2, ax3) = axs

color_split = '#2d89ef'
color_direct = '#da532c'

ax1.plot(DE_range, gradient_list, label=r"splitting structure", color=color_split)
ax1.plot([DE_range[0], DE_range[-1]], [gradient_direct, gradient_direct], '--', label=r"direct coupling", color=color_direct)
ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_xlabel('energy gain (eV)')
ax1.set_ylabel('G (V/m)')
ax1.grid()
ax1.set_ylim([1e3, 1e9])
ax1.set_yticks([1e3, 1e5, 1e7, 1e9])

ax2.plot(DE_range, N_list, label=r"splitting structure", color=color_split)
ax2.plot(DE_range, np.sqrt(N_list), '--', label=r"direct coupling", color=color_direct)

ax2.legend()
ax2.set_xscale('log')
ax2.set_yscale('log')
# ax2.set_xlabel('energy gain (eV)')
ax2.set_ylabel(r'N')
ax2.grid()
ax2.set_ylim([1, 1e8])
ax2.set_yticks([1, 1e2, 1e4, 1e6, 1e8])

# ax3 = plt.subplot(3, 1, 3)
ax3.plot(DE_range, length_list, label=r"splitting structure", color=color_split)
ax3.plot([DE_range[0], DE_range[-1]], [DE_range[0] / gradient_direct , DE_range[-1] / gradient_direct], '--', label=r"direct coupling", color=color_direct)
ax3.legend()
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel(r'$\Delta_E$ (eV)')
ax3.set_ylabel('L (m)')
ax3.grid()
ax3.set_ylim([1e-5, 1e3])
ax3.set_yticks([1e-5, 1e-3, 1e-1, 1e1, 1e3])

fig.subplots_adjust(left=0.2, wspace=0.6)

# just align the last column of axes:
fig.align_ylabels(axs)

fig.set_figheight(7)
fig.set_figwidth(5)

apply_sublabels(axs, len(axs)*[False], x=-310, y=0, size='large', ha='left', va='top', prefix='', postfix='', weight='bold')

plt.savefig('/Users/twh/Desktop/Fig6.pdf', dpi=400)

plt.show()
