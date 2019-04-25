import matplotlib.pyplot as plt
import numpy as np

import tipsi


def lattice(a=0.312, d=0, z=1):
    """MoS2/WS2 lattice.

    Parameters
    ----------
    a : float
        lattice constant
    d : float
        z-distance between Mo and S atoms
    z : float
        interlayer distance

    Returns
    ----------
    tipsi.Lattice object
        XS2 lattice.
    """

    b = a / np.sqrt(3.)
    # lattice vectors
    vectors = [[0.5 * a, 1.5 * b, 0],
               [0.5 * a, -1.5 * b, 0],
               [0, 0, z]]
    # first the three top S orbitals, then three bottom S orbitals,
    # finally five Mo orbitals
    orbital_coords = [[0, 0, d],
                      [0, 0, d],
                      [0, 0, d],
                      [0, 0, -d],
                      [0, 0, -d],
                      [0, 0, -d],
                      [0, b, 0],
                      [0, b, 0],
                      [0, b, 0],
                      [0, b, 0],
                      [0, b, 0]]
    return tipsi.Lattice(vectors, orbital_coords)


def hop_dict(X="Mo"):
    """MoS2/WS2 hopping dictionary

    Parameters
    ----------
    X : string
        Either "Mo" or "W".

    Returns
    ----------
    hops : tipsi.HopDict object
        XS2 HopDict.
    """

    if X == "Mo":
        # define parameters MoS2
        Delta0 = -1.09353
        Delta1 = 5.5
        Delta2 = -1.51187
        DeltaP = -3.55909
        DeltaZ = -6.88559
        V_pd_sigma = 3.68886
        V_pd_pi = -1.24057
        V_dd_sigma = -0.895078
        V_dd_pi = 0.252318
        V_dd_delta = 0.228446
        V_pp_sigma = 1.22524
        V_pp_pi = -0.467313
        SOC_dd = 0.075
        SOC_pp = 0.05
        V_pp_sigma_interlayer = -0.774
        V_pp_pi_interlayer = 0.123

    elif X == "W":
        # define parameters WS2
        Delta0 = -0.872
        Delta1 = 0.42
        Delta2 = -2.065
        DeltaP = -3.468
        DeltaZ = -3.913
        V_pd_sigma = 3.603
        V_pd_pi = -0.942
        V_dd_sigma = -1.216
        V_dd_pi = 0.177
        V_dd_delta = 0.243
        V_pp_sigma = 0.749
        V_pp_pi = 0.236
        SOC_dd = 0.215
        SOC_pp = 0.057
        V_pp_sigma_interlayer = -0.55
        V_pp_pi_interlayer = -0.6

    a = np.sqrt(1. / 7) / 7

    ######################################
    ######################################
    # define site-to-site hopping matrices

    # Mo-Mo, next-nearest neighbours
    # in three directions (alpha, beta, gamma)
    #
    # Mo_alpha
    #       Mo  Mo_beta
    # Mo_gamma
    #
    A_Mo_Mo_nnb_alpha = np.zeros((5, 5), dtype=complex)
    A_Mo_Mo_nnb_beta = np.zeros((5, 5), dtype=complex)
    A_Mo_Mo_nnb_gamma = np.zeros((5, 5), dtype=complex)
    A_Mo_Mo_nnb_alpha[0, 0] = (
        1 / 16) * (9 * V_dd_sigma + 4 * V_dd_pi + 3 * V_dd_delta)
    A_Mo_Mo_nnb_alpha[0, 3] = np.sqrt(
        3) / 16 * (3 * V_dd_sigma - 4 * V_dd_pi + V_dd_delta)
    A_Mo_Mo_nnb_alpha[0, 4] = 3 / 8 * (V_dd_sigma - V_dd_delta)
    A_Mo_Mo_nnb_alpha[1, 1] = 1. / 4 * (3 * V_dd_pi + V_dd_delta)
    A_Mo_Mo_nnb_alpha[1, 2] = -np.sqrt(3.) / 4 * (V_dd_pi - V_dd_delta)
    A_Mo_Mo_nnb_alpha[2, 1] = -np.sqrt(3.) / 4 * (V_dd_pi - V_dd_delta)
    A_Mo_Mo_nnb_alpha[2, 2] = 1. / 4 * (V_dd_pi + 3 * V_dd_delta)
    A_Mo_Mo_nnb_alpha[3, 0] = np.sqrt(
        3.) / 16 * (3 * V_dd_sigma - 4 * V_dd_pi + V_dd_delta)
    A_Mo_Mo_nnb_alpha[3, 3] = 1. / 16 * \
        (3 * V_dd_sigma + 12 * V_dd_pi + V_dd_delta)
    A_Mo_Mo_nnb_alpha[3, 4] = np.sqrt(3.) / 8 * (V_dd_sigma - V_dd_delta)
    A_Mo_Mo_nnb_alpha[4, 0] = 3. / 8 * (V_dd_sigma - V_dd_delta)
    A_Mo_Mo_nnb_alpha[4, 3] = np.sqrt(3.) / 8 * (V_dd_sigma - V_dd_delta)
    A_Mo_Mo_nnb_alpha[4, 4] = 1. / 4 * (V_dd_sigma + 3 * V_dd_delta)
    A_Mo_Mo_nnb_beta[0, 0] = V_dd_pi
    A_Mo_Mo_nnb_beta[1, 1] = V_dd_delta
    A_Mo_Mo_nnb_beta[2, 2] = V_dd_pi
    A_Mo_Mo_nnb_beta[3, 3] = 1 / 4 * (3 * V_dd_sigma + V_dd_delta)
    A_Mo_Mo_nnb_beta[3, 4] = -np.sqrt(3.) / 4 * (V_dd_sigma - V_dd_delta)
    A_Mo_Mo_nnb_beta[4, 3] = -np.sqrt(3.) / 4 * (V_dd_sigma - V_dd_delta)
    A_Mo_Mo_nnb_beta[4, 4] = 1. / 4 * (V_dd_sigma + 3 * V_dd_delta)
    A_Mo_Mo_nnb_gamma[0, 0] = 1. / 16 * \
        (9 * V_dd_sigma + 4 * V_dd_pi + 3 * V_dd_delta)
    A_Mo_Mo_nnb_gamma[0, 3] = - \
        np.sqrt(3.) / 16 * (3 * V_dd_sigma - 4 * V_dd_pi + V_dd_delta)
    A_Mo_Mo_nnb_gamma[0, 4] = -3. / 8 * (V_dd_sigma - V_dd_delta)
    A_Mo_Mo_nnb_gamma[1, 1] = 1. / 4 * (3 * V_dd_pi + V_dd_delta)
    A_Mo_Mo_nnb_gamma[1, 2] = np.sqrt(3.) / 4 * (V_dd_pi - V_dd_delta)
    A_Mo_Mo_nnb_gamma[2, 1] = np.sqrt(3.) / 4 * (V_dd_pi - V_dd_delta)
    A_Mo_Mo_nnb_gamma[2, 2] = 1. / 4 * (V_dd_pi + 3 * V_dd_delta)
    A_Mo_Mo_nnb_gamma[3, 0] = - \
        np.sqrt(3.) / 16 * (3 * V_dd_sigma - 4 * V_dd_pi + V_dd_delta)
    A_Mo_Mo_nnb_gamma[3, 3] = 1. / 16 * \
        (3 * V_dd_sigma + 12 * V_dd_pi + V_dd_delta)
    A_Mo_Mo_nnb_gamma[3, 4] = np.sqrt(3.) / 8 * (V_dd_sigma - V_dd_delta)
    A_Mo_Mo_nnb_gamma[4, 0] = -3. / 8 * (V_dd_sigma - V_dd_delta)
    A_Mo_Mo_nnb_gamma[4, 3] = np.sqrt(3.) / 8 * (V_dd_sigma - V_dd_delta)
    A_Mo_Mo_nnb_gamma[4, 4] = 1. / 4 * (V_dd_sigma + 3 * V_dd_delta)

    # S_top-S_top / S_bottom-S_bottom, next-nearest neighbours
    # in three directions (alpha, beta, gamma)
    #
    # S_alpha
    #      S  S_beta
    # S_gamma
    #
    A_S_S_nnb_alpha = np.zeros((3, 3), dtype=complex)
    A_S_S_nnb_beta = np.zeros((3, 3), dtype=complex)
    A_S_S_nnb_gamma = np.zeros((3, 3), dtype=complex)
    A_S_S_nnb_alpha[0, 0] = 1. / 4 * (3 * V_pp_pi + V_pp_sigma)
    A_S_S_nnb_alpha[0, 1] = np.sqrt(3.) / 4 * (V_pp_pi - V_pp_sigma)
    A_S_S_nnb_alpha[1, 0] = np.sqrt(3.) / 4 * (V_pp_pi - V_pp_sigma)
    A_S_S_nnb_alpha[1, 1] = 1. / 4 * (V_pp_pi + 3 * V_pp_sigma)
    A_S_S_nnb_alpha[2, 2] = V_pp_pi
    A_S_S_nnb_beta[0, 0] = V_pp_sigma
    A_S_S_nnb_beta[1, 1] = V_pp_pi
    A_S_S_nnb_beta[2, 2] = V_pp_pi
    A_S_S_nnb_gamma[0, 0] = 1. / 4 * (3 * V_pp_pi + V_pp_sigma)
    A_S_S_nnb_gamma[0, 1] = -np.sqrt(3.) / 4 * (V_pp_pi - V_pp_sigma)
    A_S_S_nnb_gamma[1, 0] = -np.sqrt(3.) / 4 * (V_pp_pi - V_pp_sigma)
    A_S_S_nnb_gamma[1, 1] = 1. / 4 * (V_pp_pi + 3 * V_pp_sigma)
    A_S_S_nnb_gamma[2, 2] = V_pp_pi

    # S_bottom-Mo, nearest neighbours
    # in three directions (alpha, beta, gamma)
    #
    #      Mo_beta
    #         S
    # Mo_alpha  Mo_gamma
    #
    A_Sbot_Mo_nb_alpha = np.zeros((3, 5), dtype=complex)
    A_Sbot_Mo_nb_beta = np.zeros((3, 5), dtype=complex)
    A_Sbot_Mo_nb_gamma = np.zeros((3, 5), dtype=complex)
    A_Sbot_Mo_nb_alpha[0, 0] = -1 * a * \
        (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Sbot_Mo_nb_alpha[0, 1] = 3 * a * (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Sbot_Mo_nb_alpha[0, 2] = np.sqrt(
        3.) * a * (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Sbot_Mo_nb_alpha[0, 3] = - \
        np.sqrt(3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Sbot_Mo_nb_alpha[0, 4] = - \
        np.sqrt(3.) * a * (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_alpha[1, 0] = - \
        np.sqrt(3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Sbot_Mo_nb_alpha[1, 1] = np.sqrt(
        3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Sbot_Mo_nb_alpha[1, 2] = 3 * a * (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Sbot_Mo_nb_alpha[1, 3] = - \
        np.sqrt(3.) * a * (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_alpha[1, 4] = -1 * a * \
        (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_alpha[2, 0] = 3 * a * (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Sbot_Mo_nb_alpha[2, 1] = -1 * a * \
        (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Sbot_Mo_nb_alpha[2, 2] = - \
        np.sqrt(3.) * a * (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Sbot_Mo_nb_alpha[2, 3] = a * (3 * V_pd_sigma - 2 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_alpha[2, 4] = np.sqrt(
        3.) * a * (V_pd_sigma + 4 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_beta[0, 0] = 14 * a * V_pd_pi
    A_Sbot_Mo_nb_beta[0, 2] = 7 * a * np.sqrt(3.) * V_pd_pi
    A_Sbot_Mo_nb_beta[1, 1] = np.sqrt(
        3.) * a * (4 * np.sqrt(3.) * V_pd_sigma - V_pd_pi)
    A_Sbot_Mo_nb_beta[1, 3] = -1 * a * \
        (4 * np.sqrt(3.) * V_pd_sigma + 6 * V_pd_pi)
    A_Sbot_Mo_nb_beta[1, 4] = 2 * a * (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_beta[2, 1] = 2 * a * (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Sbot_Mo_nb_beta[2, 3] = -2 * a * \
        (3 * V_pd_sigma - 2 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_beta[2, 4] = np.sqrt(
        3.) * a * (V_pd_sigma + 4 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_gamma[0, 0] = -1 * a * \
        (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Sbot_Mo_nb_gamma[0, 1] = -3 * a * \
        (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Sbot_Mo_nb_gamma[0, 2] = np.sqrt(
        3.) * a * (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Sbot_Mo_nb_gamma[0, 3] = np.sqrt(
        3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Sbot_Mo_nb_gamma[0, 4] = np.sqrt(
        3.) * a * (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_gamma[1, 0] = np.sqrt(
        3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Sbot_Mo_nb_gamma[1, 1] = np.sqrt(
        3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Sbot_Mo_nb_gamma[1, 2] = -3 * a * \
        (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Sbot_Mo_nb_gamma[1, 3] = - \
        np.sqrt(3.) * a * (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_gamma[1, 4] = -1 * a * \
        (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_gamma[2, 0] = -3 * a * \
        (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Sbot_Mo_nb_gamma[2, 1] = -1 * a * \
        (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Sbot_Mo_nb_gamma[2, 2] = np.sqrt(
        3.) * a * (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Sbot_Mo_nb_gamma[2, 3] = a * (3 * V_pd_sigma - 2 * np.sqrt(3.) * V_pd_pi)
    A_Sbot_Mo_nb_gamma[2, 4] = np.sqrt(
        3.) * a * (V_pd_sigma + 4 * np.sqrt(3.) * V_pd_pi)

    # S_top-Mo, nearest neighbours
    # in three directions (alpha, beta, gamma)
    #
    #      Mo_beta
    #         S
    # Mo_alpha  Mo_gamma
    #
    A_Stop_Mo_nb_alpha = np.zeros((3, 5), dtype=complex)
    A_Stop_Mo_nb_beta = np.zeros((3, 5), dtype=complex)
    A_Stop_Mo_nb_gamma = np.zeros((3, 5), dtype=complex)
    A_Stop_Mo_nb_alpha[0, 0] = -1 * a * \
        (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Stop_Mo_nb_alpha[0, 1] = -3 * a * \
        (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Stop_Mo_nb_alpha[0, 2] = - \
        np.sqrt(3.) * a * (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Stop_Mo_nb_alpha[0, 3] = - \
        np.sqrt(3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Stop_Mo_nb_alpha[0, 4] = - \
        np.sqrt(3.) * a * (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_alpha[1, 0] = - \
        np.sqrt(3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Stop_Mo_nb_alpha[1, 1] = - \
        np.sqrt(3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Stop_Mo_nb_alpha[1, 2] = -3 * a * \
        (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Stop_Mo_nb_alpha[1, 3] = - \
        np.sqrt(3.) * a * (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_alpha[1, 4] = -1 * a * \
        (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_alpha[2, 0] = -3 * a * \
        (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Stop_Mo_nb_alpha[2, 1] = -1 * a * \
        (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Stop_Mo_nb_alpha[2, 2] = - \
        np.sqrt(3.) * a * (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Stop_Mo_nb_alpha[2, 3] = -a * \
        (3 * V_pd_sigma - 2 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_alpha[2, 4] = - \
        np.sqrt(3.) * a * (V_pd_sigma + 4 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_beta[0, 0] = 14 * a * V_pd_pi
    A_Stop_Mo_nb_beta[0, 2] = -7 * a * np.sqrt(3.) * V_pd_pi
    A_Stop_Mo_nb_beta[1, 1] = - \
        np.sqrt(3.) * a * (4 * np.sqrt(3.) * V_pd_sigma - V_pd_pi)
    A_Stop_Mo_nb_beta[1, 3] = -1 * a * \
        (4 * np.sqrt(3.) * V_pd_sigma + 6 * V_pd_pi)
    A_Stop_Mo_nb_beta[1, 4] = 2 * a * (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_beta[2, 1] = 2 * a * (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Stop_Mo_nb_beta[2, 3] = 2 * a * \
        (3 * V_pd_sigma - 2 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_beta[2, 4] = - \
        np.sqrt(3.) * a * (V_pd_sigma + 4 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_gamma[0, 0] = -1 * a * \
        (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Stop_Mo_nb_gamma[0, 1] = 3 * a * (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Stop_Mo_nb_gamma[0, 2] = - \
        np.sqrt(3.) * a * (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Stop_Mo_nb_gamma[0, 3] = np.sqrt(
        3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Stop_Mo_nb_gamma[0, 4] = np.sqrt(
        3.) * a * (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_gamma[1, 0] = np.sqrt(
        3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Stop_Mo_nb_gamma[1, 1] = - \
        np.sqrt(3.) * a * (np.sqrt(3.) * V_pd_sigma + 5 * V_pd_pi)
    A_Stop_Mo_nb_gamma[1, 2] = 3 * a * (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Stop_Mo_nb_gamma[1, 3] = - \
        np.sqrt(3.) * a * (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_gamma[1, 4] = -1 * a * \
        (V_pd_sigma - 3 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_gamma[2, 0] = 3 * a * (np.sqrt(3.) * V_pd_sigma - 2 * V_pd_pi)
    A_Stop_Mo_nb_gamma[2, 1] = -1 * a * \
        (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Stop_Mo_nb_gamma[2, 2] = np.sqrt(
        3.) * a * (3 * np.sqrt(3.) * V_pd_sigma + V_pd_pi)
    A_Stop_Mo_nb_gamma[2, 3] = -a * \
        (3 * V_pd_sigma - 2 * np.sqrt(3.) * V_pd_pi)
    A_Stop_Mo_nb_gamma[2, 4] = - \
        np.sqrt(3.) * a * (V_pd_sigma + 4 * np.sqrt(3.) * V_pd_pi)

    # S_top-S-bot, within the unit cell
    A_Stop_Sbot_uc = np.zeros((3, 3), dtype=complex)
    A_Stop_Sbot_uc[0, 0] = V_pp_pi
    A_Stop_Sbot_uc[1, 1] = V_pp_pi
    A_Stop_Sbot_uc[2, 2] = V_pp_sigma

    # S_top-S-top / S_bottom-S_bottom, onsite
    A_S_S_uc = np.zeros((3, 3), dtype=complex)
    A_S_S_uc[0, 0] = DeltaP
    A_S_S_uc[1, 1] = DeltaP
    A_S_S_uc[2, 2] = DeltaZ

    # Mo-Mo, onsite
    A_Mo_Mo_uc = np.zeros((5, 5), dtype=complex)
    A_Mo_Mo_uc[0, 0] = Delta2
    A_Mo_Mo_uc[1, 1] = Delta1
    A_Mo_Mo_uc[2, 2] = Delta1
    A_Mo_Mo_uc[3, 3] = Delta2
    A_Mo_Mo_uc[4, 4] = Delta0
    ######################################
    ######################################

    # set empty hopping matrices
    hop_dict = tipsi.HopDict()
    hop_dict.empty((0, 0, 0), (11, 11))  # uc; nb: beta
    hop_dict.empty((-1, 0, 0), (11, 11))  # nb: alpha; nnb: gamma
    hop_dict.empty((0, 1, 0), (11, 11))  # nb: gamma
    hop_dict.empty((0, -1, 0), (11, 11))  # nnb: alpha
    hop_dict.empty((1, 1, 0), (11, 11))  # nnb: beta

    # fill hopping matrices
    # conjugates are added automatically
    hop_dict.dict[(0, 0, 0)][0:3, 0:3] = A_S_S_uc[:, :]
    hop_dict.dict[(0, 0, 0)][3:6, 3:6] = A_S_S_uc[:, :]
    hop_dict.dict[(0, 0, 0)][6:11, 6:11] = A_Mo_Mo_uc[:, :]
    hop_dict.dict[(0, 0, 0)][0:3, 3:6] = A_Stop_Sbot_uc[:, :]
    hop_dict.dict[(0, 0, 0)][0:3, 6:11] = A_Stop_Mo_nb_beta[:, :]
    hop_dict.dict[(0, 0, 0)][3:6, 6:11] = A_Sbot_Mo_nb_beta[:, :]

    hop_dict.dict[(-1, 0, 0)][0:3, 6:11] = A_Stop_Mo_nb_alpha[:, :]
    hop_dict.dict[(-1, 0, 0)][3:6, 6:11] = A_Sbot_Mo_nb_alpha[:, :]
    hop_dict.dict[(-1, 0, 0)][0:3, 0:3] = A_S_S_nnb_gamma[:, :]
    hop_dict.dict[(-1, 0, 0)][3:6, 3:6] = A_S_S_nnb_gamma[:, :]
    hop_dict.dict[(-1, 0, 0)][6:11, 6:11] = A_Mo_Mo_nnb_gamma[:, :]

    hop_dict.dict[(0, 1, 0)][0:3, 6:11] = A_Stop_Mo_nb_gamma[:, :]
    hop_dict.dict[(0, 1, 0)][3:6, 6:11] = A_Sbot_Mo_nb_gamma[:, :]

    hop_dict.dict[(0, -1, 0)][0:3, 0:3] = A_S_S_nnb_alpha[:, :]
    hop_dict.dict[(0, -1, 0)][3:6, 3:6] = A_S_S_nnb_alpha[:, :]
    hop_dict.dict[(0, -1, 0)][6:11, 6:11] = A_Mo_Mo_nnb_alpha[:, :]

    hop_dict.dict[(1, 1, 0)][0:3, 0:3] = A_S_S_nnb_beta[:, :]
    hop_dict.dict[(1, 1, 0)][3:6, 3:6] = A_S_S_nnb_beta[:, :]
    hop_dict.dict[(1, 1, 0)][6:11, 6:11] = A_Mo_Mo_nnb_beta[:, :]

    return hop_dict


def sheet_rectangle(W, H):
    """XS2 SiteSet, rectangular.

    Parameters
    ----------
    W : integer
        width of the SiteSet, in unit cells
    H : integer
        height of the SiteSet, in unit cells

    Returns
    ----------
    site_set : tipsi.SiteSet object
        Rectangular XS2 SiteSet.
    """

    site_set = tipsi.SiteSet()
    for x in range(W):
        for y in range(int(H / 2)):
            i, j = x + y, x - y
            for k in range(11):
                unit_cell_coords = (i, j, 0)
                site_set.add_site(unit_cell_coords, k)
                unit_cell_coords = (i + 1, j, 0)
                site_set.add_site(unit_cell_coords, k)
    return site_set


def pbc_rectangle(W, H, unit_cell_coords, orbital):
    """PBC for a rectangular XS2 sample.

    Parameters
    ----------
    W : integer
        width of the sample, in unit cells
    H : integer
        height of the sample, in unit cells
    unit_cell_coords : 3-tuple of integers
        unit cell coordinates
    orbital : integer
        orbital index

    Returns
    ----------
    unit_cell_coords : 3-tuple of integers
        unit cell coordinates
    orbital : integer
        orbital index
    """

    # get input
    x, y, z = unit_cell_coords
    # transform to rectangular coords (xloc, yloc)
    xloc = (x + y) / 2.
    yloc = (x - y) / 2.
    # use standard pbc
    xloc = xloc % W
    yloc = yloc % (H / 2)
    # transform back
    x = int(xloc + yloc)
    y = int(xloc - yloc)
    # done
    return (x, y, z), orbital
