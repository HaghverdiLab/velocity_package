import numpy as np
import scipy.optimize as opt


def S(alpha, gamma, U0, S0, unspliced):
    """
    Function to calculate s(u).

    Parameters
    ----------
    alpha: `float` (set to 0 in down-reg)
    gamma: `float`
    U0: `float` (set to 0 in up-reg)
    S0: `float` (set to 0 in up-reg)
    unspliced: `list of float`
        unspliced counts

    Returns
        -------
    `list of float`
        spliced counts
    """
    A = S0 - (alpha / gamma) + ((alpha - U0) / (gamma - 1))
    e_gamma_t = ((unspliced - alpha) / (U0 - alpha)) ** gamma
    C = ((unspliced - alpha) / (gamma - 1)) + (alpha / gamma)
    return (A * e_gamma_t) + C


def dist(alpha, gamma, Pi, U0, S0, unspliced, spliced, weight=1):
    """
    Calculates distance between a set of points on the curve and real measurements defined bi unspliced[], spliced[].
    The curve is defined by alpha, gamma, U0, S0, and the points are defined by their unspliced value Pi.

    Parameters
    ----------
    alpha: `float` (set to 0 in down-reg)
    gamma: `float`
    Pi: `list of float`
    U0: `float` (set to 0 in up-reg)
    S0: `float` (set to 0 in up-reg)
    unspliced: `list of float`
        real measurements of unspliced counts
    spliced: `list of float`
        real measurements of spliced counts
    weight: `float`(default=1)
        scaling parameter for returned costs, s.t. the distance in u and s is weighted equally.
    Returns
        -------
    distS: `list of float`
        distances in s
    distU: `list of float`
        weighted distances in u
    """

    distS, distU = (S(alpha, gamma, U0, S0, Pi) - spliced), ((Pi - unspliced) * weight)
    if len(Pi) > 1:
        i = np.isnan(distS)  # past the exponential function limit, we go straight down
        if np.sum(i) > 0:
            distS[i], distU[i] = 0, ((Pi[i] - unspliced[i]) * weight)
    else:
        if np.isnan(distS[0]):
            distS, distU = 0, ((Pi - unspliced) * weight)
    return distS, distU


def dist_k(alpha, gamma, Uk, Pi, k, U0, S0, unspliced, spliced, weight=1):
    """
    Wrapper for dist function that accounts for cells being in up- (k=1) or down (k=0) regulation.

    Parameters not equal to those called in dist function:
    ----------
    Uk: `float`
        u value at switching time point
    k: `list of int` or `list of bool`
        saves whether the cell is assigned to up- (k=1) or down (k=0) regulation
        has equal length as spliced, unspliced
        """
    Sk = S(alpha, gamma, U0, S0, Uk)
    distS, distU = np.zeros(Pi.shape), np.zeros(Pi.shape)
    if np.sum(k) > 0:
        distS[k], distU[k] = dist(alpha, gamma, Pi[k], U0, S0, unspliced[k], spliced[k], weight)
    if np.sum(~k) > 0:
        distS[~k], distU[~k] = dist(0, gamma, Pi[~k], Uk, Sk, unspliced[~k], spliced[~k], weight)
    return distS, distU


def get_closest(U_ref, S_ref, unspliced, spliced, weight):
    """
    Calculates the same as dist() but on matrices for speed-up.
    """
    shape = (len(unspliced), len(U_ref))
    U_Matrix = (np.ones(shape) * U_ref).T
    S_Matrix = (np.ones(shape) * S_ref).T
    distU, distS = ((U_Matrix - unspliced) * weight), (S_Matrix - spliced)
    dist = distS ** 2 + distU ** 2
    idx = np.argmin(dist, axis=0)
    minU = distU.T[np.arange(distU.shape[1]), idx]
    minS = distS.T[np.arange(distU.shape[1]), idx]
    return U_ref[idx], (minU, minS)


def get_Pi_fast(alpha, gamma, U0, S0, Uk, unspliced, spliced, weight, n=100):
    k = unspliced > gamma * spliced

    Pi = np.zeros(unspliced.shape)
    D_u, D_s = np.zeros(unspliced.shape), np.zeros(unspliced.shape)
    U_range = np.arange(0, Uk, Uk / n)
    Sk = S(alpha, gamma, U0, S0, Uk)

    # up
    S_up = S(alpha, gamma, U0, S0, U_range)
    up_Pi, (d_u, d_s) = get_closest(U_range, S_up, unspliced[k], spliced[k], weight)
    Pi[k], D_u[k], D_s[k] = up_Pi, d_u, d_s

    # down
    S_down = S(0, gamma, Uk, Sk, U_range)  # S_k(alpha, gamma, 0, 0, Uk, Sk, U_range, np.zeros(n).astype(bool))
    down_Pi, (d_u, d_s) = get_closest(U_range, S_down, unspliced[~k], spliced[~k], weight)
    Pi[~k], D_u[~k], D_s[~k] = down_Pi, d_u, d_s

    return Pi, (D_u, D_s)


def cost_wrapper_full_Pi(Pi, alpha, gamma, U0, S0, unspliced, spliced, weight):
    Si = S(alpha, gamma, U0, S0, Pi)

    distS, distU = (Si - spliced), (Pi - unspliced) * weight
    return distS ** 2 + distU ** 2


def get_Pi_full(alpha, gamma, k, U0, S0, Uk, unspliced, spliced, weight):
    Pi = np.zeros(unspliced.shape)

    # up
    if np.sum(k) > 0:
        up_Pi = np.zeros(np.sum(k))
        for i in range(np.sum(k)):
            res1 = opt.minimize(cost_wrapper_full_Pi,
                                x0=unspliced[k][i] if unspliced[k][i] < Uk else Uk * .9,
                                args=(alpha, gamma, U0, S0, unspliced[k][i], spliced[k][i], weight),
                                bounds=[(0, Uk)],
                                method="Nelder-Mead")
            up_Pi[i] = res1.x
        Pi[k] = up_Pi

    # down
    if np.sum(~k) > 0:
        Sk = S(alpha, gamma, U0, S0, Uk)
        down_Pi = np.zeros(np.sum(~k))
        for i in range(np.sum(~k)):
            res1 = opt.minimize(cost_wrapper_full_Pi,
                                x0=unspliced[~k][i] if unspliced[~k][i] < Uk else Uk * .9,
                                args=(0, gamma, Uk, Sk, unspliced[~k][i], spliced[~k][i], weight),
                                bounds=[(0, Uk)],
                                method="Nelder-Mead")
            down_Pi[i] = res1.x
        Pi[~k] = down_Pi
    return Pi


def cost_wrapper_fastPi(alpha_gamma_Uk, U0, S0, unspliced, spliced, n):
    alpha, gamma, Uk = alpha_gamma_Uk[0], alpha_gamma_Uk[1], alpha_gamma_Uk[2]

    penalty = 0
    if Uk >= alpha * .99:  # constraint: switching time point cannot be bigger than steady-state value
        # note: this is a bit hacky but works
        penalty = 100 * (Uk - alpha)
        Uk = alpha * .99

    weight = np.std(spliced) / np.std(unspliced)

    Pi, (distS, distU) = get_Pi_fast(alpha=alpha, gamma=gamma, U0=U0, S0=S0, Uk=Uk,
                                     unspliced=unspliced, spliced=spliced, weight=weight, n=n)

    return np.sum(distS ** 2 + distU ** 2) + penalty


def cost_wrapper_fastPi_scaling(alpha_gamma_Uk, U0, S0, unspliced, spliced, n):
    alpha, gamma, Uk, scaling = alpha_gamma_Uk[0], alpha_gamma_Uk[1], alpha_gamma_Uk[2], alpha_gamma_Uk[3]
    u_s = unspliced * scaling
    return cost_wrapper_fastPi([alpha, gamma, Uk], U0, S0, u_s, spliced, n)


import matplotlib.pyplot as plt


def get_likelihood(alpha, gamma, U0, S0, Uk, spliced, unspliced, k, Pi):
    distS, distU = dist_k(alpha, gamma, Uk, Pi, k, U0, S0, unspliced, spliced, weight=1)

    std_s, std_u = np.std(spliced), np.std(unspliced)
    #plt.hist(distS, bins=50)
    #plt.hist(distU, bins=50, alpha=.5)
    #plt.show()
    #print(np.std(distS), np.std(distU))
    distS /= std_s
    distU /= std_u
    #plt.hist(distS, bins=50)
    #plt.hist(distU, bins=50, alpha=.5)
    #plt.show()
    #print(np.std(distS), np.std(distU))

    distX = distU ** 2 + distS ** 2
    varx = np.var(np.sign(distS) * np.sqrt(distX))
    varx += varx == 0
    n = len(distS)
    loglik = - (1 / (2 * n)) * np.sum(distX / varx)

    return np.exp(loglik) * (1 / np.sqrt(2 * np.pi * varx))
