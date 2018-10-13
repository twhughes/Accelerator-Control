import numpy as np


def get_power(mode):
    return np.square(np.abs(mode))


def make_M(phi1, phi2):
    # returns a 2x2 tunable beam splitter (MZI) transfer matrix
    # given the settings of the two integrated phase shifters: phi1, phi2
    M = np.zeros((2, 2), dtype=complex)
    M[0, 0] = np.cos(phi2/2)
    M[1, 0] = -np.sin(phi2/2)
    M[0, 1] = np.exp(1j*phi1)*np.sin(phi2/2)
    M[1, 1] = np.exp(1j*phi1)*np.cos(phi2/2)
    M      = np.exp(1j*phi2/2)*M
    return M

"""    M[0, 0] = -np.sin(phi2/2)
    M[1, 0] =  np.cos(phi2/2)
    M[0, 1] =  np.exp(1j*phi1)*np.cos(phi2/2)
    M[1, 1] =  np.exp(1j*phi1)*np.sin(phi2/2)
    M      = -1j*np.exp(1j*phi2/2)*M
    """

def make_layer_matrix(N, layer_index, phi1, phi2):
    # gives the transfer matrix for the layer_index-th layer of the total system

    if layer_index == 0:
        return np.eye((N), dtype=complex)

    matrix_indeces = list(range(N-2, 0, -1)) + list(range(0, N-1))
    matrix_index = matrix_indeces[layer_index-1]

    M = np.eye((N), dtype=complex)
    M_2x2 = make_M(phi1, phi2)
    M[matrix_index:matrix_index+2, matrix_index:matrix_index+2] = M_2x2
    return M


def make_partial_matrix(N, phi_list, layer_index=-1):
    # gives the transfer matrix up directly after layer_index-th layer in the system
    if layer_index == -1:
        layer_index = 2*N-3

    M = np.eye((N), dtype=complex)
    for mzi_index in range(1, layer_index+1):
        phi1 = phi_list[mzi_index-1, 0]
        phi2 = phi_list[mzi_index-1, 1]
        M_i = make_layer_matrix(N, mzi_index, phi1, phi2)
        M = np.dot(M_i, M)
    return M


def make_full_matrix(N, phi_list):
    # gives the full, multiplied transfer matrix (not needed)
    N_MZI = 2*N-3
    M = np.eye((N), dtype=complex)
    matrix_indeces = list(range(N-2, 0, -1)) + list(range(0, N-1))
    for i in range(N_MZI):
        phi1 = phi_list[i, 0]
        phi2 = phi_list[i, 1]
        matrix_index = matrix_indeces[i]
        M_i = make_layer_matrix(N, matrix_index, phi1, phi2)
        M = np.dot(M_i, M)
    return M
