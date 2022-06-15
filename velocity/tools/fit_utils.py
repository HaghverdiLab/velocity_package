import numpy as np


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


def D2(alpha, gamma, Pi, U0, S0, unspliced, spliced, cost_scaling=1):
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
    cost_scaling: `float`(default=1)
        scaling parameter for returned costs, s.t. the distance in u and s is weighted equally. This parameter
        is only used to weights the returned distances.
    Returns
        -------
    distS: `list of float`
        distances in s
    distU: `list of float`
        cost_scaled distances in u
    """

    distS, distU = (S(alpha, gamma, U0, S0, Pi) - spliced), ((Pi - unspliced) * cost_scaling)
    if len(Pi) > 1:
        i = np.isnan(distS)
        if np.sum(i) > 0:
            distS[i], distU[i] = ((alpha / gamma) - spliced[i]), ((Pi[i] - unspliced[i]) * cost_scaling)
    else:
        if np.isnan(distS[0]):
            distS, distU = ((alpha / gamma) - spliced), ((Pi - unspliced) * cost_scaling)
    return distS, distU


def D2_k(alpha, gamma, Uk, Pi, k, U0, S0, unspliced, spliced, cost_scaling=1):
    """
    Wrapper for D2 that accounts for cells being in up- (k=1) or down (k=0) regulation.

    Parameters
    ----------
    alpha: `float` (set to 0 in down-reg)
    gamma: `float`
    Uk: `float`
        u value at switching time point
    Pi: `list of float`
    k: `list of int` or `list of bool`
        saves whether the cell is assigned to up- (k=1) or down (k=0) regulation
        has equal length as spliced, unspliced
    U0: `float` (set to 0 in up-reg)
    S0: `float` (set to 0 in up-reg)
    unspliced: `list of float`
        real measurements of unspliced counts
    spliced: `list of float`
        real measurements of spliced counts
    cost_scaling: `float`(default=1)
        scaling parameter for returned costs, s.t. the distance in u and s is weighted equally. This parameter
        is only used to weights the returned distances.
    Returns
        -------
    distS: `list of float`
        distances in s
    distU: `list of float`
        cost_scaled distances in u
        """
    Sk = S(alpha, gamma, U0, S0, Uk)
    distS, distU = np.zeros(Pi.shape), np.zeros(Pi.shape)
    if np.sum(k) > 0:
        distS[k], distU[k] = D2(alpha, gamma, Pi[k], U0, S0, unspliced[k], spliced[k], cost_scaling)
    if np.sum(~k) > 0:
        distS[~k], distU[~k] = D2(0, gamma, Pi[~k], Uk, Sk, unspliced[~k], spliced[~k], cost_scaling)
    return distS, distU


def get_closest(U_ref, S_ref, unspliced, spliced, scaling):
    shape = (len(unspliced), len(U_ref))
    U_Matrix = (np.ones(shape) * U_ref).T
    S_Matrix = (np.ones(shape) * S_ref).T
    distU, distS = ((U_Matrix - unspliced) * scaling) ** 2, (S_Matrix - spliced) ** 2
    dist = distS + distU
    return U_ref[np.argmin(dist, axis=0)]  # , (np.min(distU, axis=0), np.min(distS, axis=0))


def get_Pi_fast(alpha, gamma, k, U0, S0, Uk, unspliced, spliced, scaling, n=100):
    Pi = np.zeros(unspliced.shape)
    U_range = np.arange(0, Uk, Uk / n)
    Sk = S(alpha, gamma, U0, S0, Uk)

    # up
    S_up = S(alpha, gamma, U0, S0, U_range)
    up_Pi = get_closest(U_range, S_up, unspliced[k], spliced[k], scaling)

    Pi[k] = up_Pi

    # down
    S_down = S(0, gamma, Uk, Sk, U_range)  # S_k(alpha, gamma, 0, 0, Uk, Sk, U_range, np.zeros(n).astype(bool))
    down_Pi = get_closest(U_range, S_down, unspliced[~k], spliced[~k], scaling)
    Pi[~k] = down_Pi

    return Pi


def cost(alpha, gamma, Uk, U0, S0, unspliced, spliced, cost_scaling, n):
    # todo speedup by computing Pi and distS, distU in one step instead of 2
    k = (unspliced / spliced) > (Uk / S(alpha, gamma, U0, S0, Uk))
    Pi = get_Pi_fast(alpha=alpha, gamma=gamma, k=k, U0=U0, S0=S0, Uk=Uk,
                     unspliced=unspliced, spliced=spliced, scaling=cost_scaling, n=n)
    distS, distU = D2_k(alpha=alpha, gamma=gamma, k=k, U0=U0, S0=S0, Uk=Uk,
                        unspliced=unspliced, spliced=spliced, cost_scaling=cost_scaling, Pi=Pi)
    return np.sum(distS ** 2 + distU ** 2)


def cost_wrapper(alpha_gamma_Uk, U0, S0, unspliced, spliced, n):
    alpha, gamma, Uk = alpha_gamma_Uk[0], alpha_gamma_Uk[1], alpha_gamma_Uk[2]

    penalty = 0
    if Uk > alpha:  # constraint: switching time point cannot be bigger than steady-state value
        # note: this is a bit hacky but works
        penalty = 100 * (Uk - alpha)
        Uk = alpha

    return cost(alpha, gamma, Uk, U0, S0, unspliced, spliced, np.std(spliced) / np.std(unspliced), n) + penalty


def cost_wrapper_scaling(alpha_gamma_Uk, U0, S0, unspliced, spliced, n):
    alpha, gamma, Uk, scaling = alpha_gamma_Uk[0], alpha_gamma_Uk[1], alpha_gamma_Uk[2], alpha_gamma_Uk[3]
    u_s = unspliced * scaling

    return cost_wrapper([alpha, gamma, Uk], U0, S0, u_s, spliced, n)
