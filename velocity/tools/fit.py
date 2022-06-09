import numpy as np
import scipy.optimize as opt


def fit_all(adata, use_raw=True, n=300):
    unspliced, spliced = adata.layers["unspliced" if use_raw else "Mu"], adata.layers["spliced" if use_raw else "Ms"]
    n_genes = adata.shape[1]
    alpha, gamma, U_switch, likelihood = np.zeros(n_genes), np.zeros(n_genes), np.zeros(n_genes), np.zeros(n_genes)

    for i in range(adata.shape[1]):
        U_, S_ = unspliced[:, i], spliced[:, i]
        Ms = np.max(S_)
        a, g, Uk, lik = fit(U_ / Ms, S_ / Ms, n=n)
        alpha[i], gamma[i], U_switch[i], likelihood[i] = a, g, Uk, lik
    return alpha, gamma, U_switch, likelihood


def fit_helper(pars):
    n_genes = len(pars)
    alpha, gamma, U_switch, likelihood = np.zeros(n_genes), np.zeros(n_genes), np.zeros(n_genes), np.zeros(n_genes)

    for i, p in enumerate(pars):
        a, g, Uk, lik = fit(p[0], p[1], n=p[2])
        alpha[i], gamma[i], U_switch[i], likelihood[i] = a, g, Uk, lik
    print("-")
    return np.array([alpha, gamma, U_switch, likelihood])


def S(alpha, gamma, U0, S0, unspliced):
    A = S0 - (alpha / gamma) + ((alpha - U0) / (gamma - 1))
    e_gamma_t = ((unspliced - alpha) / (U0 - alpha)) ** gamma
    C = ((unspliced - alpha) / (gamma - 1)) + (alpha / gamma)
    return (A * e_gamma_t) + C


def D2(alpha, gamma, Pi, U0, S0, unspliced, spliced, scaling=1):
    distS, distU = (S(alpha, gamma, U0, S0, Pi) - spliced), ((Pi - unspliced) * scaling)
    if len(Pi) > 1:
        i = np.isnan(distS)
        if np.sum(i) > 0:
            distS[i], distU[i] = ((alpha / gamma) - spliced[i]), ((Pi[i] - unspliced[i]) * scaling)
    else:
        if np.isnan(distS[0]):
            distS, distU = ((alpha / gamma) - spliced), ((Pi - unspliced) * scaling)
    return distS, distU


def cost(alpha, gamma, Uk, Pi, k, U0, S0, unspliced, spliced, scaling):
    distS, distU = D2_k(alpha, gamma, Uk, Pi, k, U0, S0, unspliced, spliced, scaling)
    return np.sum(distS ** 2 + distU ** 2)


def D2_k(alpha, gamma, Uk, Pi, k, U0, S0, unspliced, spliced, scaling=1):
    Sk = S(alpha, gamma, U0, S0, Uk)
    distS, distU = np.zeros(Pi.shape), np.zeros(Pi.shape)
    if np.sum(k) > 0:
        distS[k], distU[k] = D2(alpha, gamma, Pi[k], U0, S0, unspliced[k], spliced[k], scaling)
    if np.sum(~k) > 0:
        distS[~k], distU[~k] = D2(0, gamma, Pi[~k], Uk, Sk, unspliced[~k], spliced[~k], scaling)
    return distS, distU


def D2_full_wrapper(alpha_gamma_Uk, U0, S0, unspliced, spliced, scaling, n):
    alpha, gamma, Uk = alpha_gamma_Uk[0], alpha_gamma_Uk[1], alpha_gamma_Uk[2]
    penalty = 0
    if Uk > alpha:  # constraint: switching time point cannot be bigger than steady-state value
        # note: this is a bit hacky but works
        penalty = 100 * (Uk - alpha)
        Uk = alpha
    k = (unspliced / spliced) > (Uk / S(alpha, gamma, U0, S0, Uk))
    Pi = get_Pi_fast(alpha, gamma, k, U0, S0, Uk, unspliced, spliced, scaling, n)
    cost_ = cost(alpha, gamma, Uk, Pi, k, U0, S0, unspliced, spliced, scaling) + penalty
    return cost_


def get_closest(U_ref, S_ref, unspliced, spliced, scaling):
    shape = (len(unspliced), len(U_ref))
    U_Matrix = (np.ones(shape) * U_ref).T
    S_Matrix = (np.ones(shape) * S_ref).T

    dist = (S_Matrix - spliced) ** 2 + ((U_Matrix - unspliced) ** 2) * scaling
    return U_ref[np.argmin(dist, axis=0)]


def get_Pi_fast(alpha, gamma, k, U0, S0, Uk, unspliced, spliced, scaling, n=100):
    Pi = np.zeros(unspliced.shape)
    U_range = np.arange(0, alpha, alpha / n)
    Sk = S(alpha, gamma, U0, S0, Uk)

    # up
    S_up = S(alpha, gamma, U0, S0, U_range)
    up_Pi = get_closest(U_range, S_up, unspliced[k], spliced[k], scaling)
    Pi[k] = up_Pi

    # down
    S_down = S(0, gamma, Uk, Sk, U_range)
    down_Pi = get_closest(U_range, S_down, unspliced[~k], spliced[~k], scaling)
    Pi[~k] = down_Pi

    return Pi


def get_likelihood(alpha, gamma, U0, S0, Uk, spliced, unspliced, scaling):
    k = (unspliced / spliced) > (Uk / S(alpha, gamma, U0, S0, Uk))
    Pi = get_Pi_fast(alpha, gamma, k, U0, S0, Uk, unspliced, spliced, scaling, 500)  # todo full Pi
    distS, distU = D2_k(alpha, gamma, Uk, Pi, k, U0, S0, unspliced, spliced, scaling)

    std_s, std_u = np.std(spliced * scaling), np.std(unspliced)
    distS /= std_s
    distU /= std_u

    distX = distU ** 2 + distS ** 2
    varx = np.var(np.sign(distS) * np.sqrt(distX))
    varx += varx == 0
    n = len(distS)
    loglik = - (1 / (2 * n)) * np.sum(distX / varx)

    return np.exp(loglik) * (1 / np.sqrt(2 * np.pi * varx))


def fit(unspliced, spliced, n=50):
    max_u, max_s = np.quantile(unspliced, .98), np.quantile(spliced, .98)
    sub = (unspliced > 0) & (spliced > 0) & ((unspliced > 0.05 * max_u) | (spliced > 0.05 * max_s))

    spliced, unspliced = spliced[sub], unspliced[sub]
    std_s, std_u = np.std(spliced), np.std(unspliced)
    # initialisation
    scaling = 1/0.6#std_s / std_u  # used s.t. penalty on S and on U are on the same scale
    if np.sum(sub) > 100:
        print(scaling)
        unspliced *= scaling
        scaling = 1
        max_u, max_s = np.max(unspliced), np.max(spliced)
        alpha, gamma = max_u, max_u / max_s
        U0, S0 = 0, 0
        # fit
        i = 2
        # plot_kinetics(alpha, gamma, spliced, unspliced, alpha, scaling=scaling)
        res1 = opt.minimize(D2_full_wrapper,
                            x0=np.array([alpha, gamma, alpha]),
                            args=(U0, S0, unspliced, spliced, scaling, n),
                            bounds=((alpha / i, alpha * i), (gamma / i, gamma * i), (0, None)),
                            method="Nelder-Mead")
        alpha, gamma, Uk = res1.x
        if Uk > alpha:
            Uk = alpha
        lik = get_likelihood(alpha, gamma, U0, S0, Uk, spliced, unspliced, scaling)
        plot_kinetics(alpha, gamma, spliced, unspliced, Uk, scaling=scaling)
    else:
        alpha, gamma, Uk, lik = np.nan, np.nan, np.nan, 0
    return alpha, gamma, Uk, lik


import matplotlib.pyplot as plt

kwargs = {"scale": 1, "angles": "xy", "scale_units": "xy", "edgecolors": "k",
          "linewidth": 0.01, "headlength": 4, "headwidth": 5, "headaxislength": 3, "alpha": .3}


def plot_kinetics(alpha, gamma, spliced, unspliced, Uk, dist=True, scaling=1, U0=0, S0=0):
    Sk = S(alpha, gamma, U0, S0, Uk)
    k = (unspliced / spliced) > (Uk / S(alpha, gamma, U0, S0, Uk))
    Pi = get_Pi_fast(alpha, gamma, k, U0, S0, Uk, unspliced, spliced, scaling, n=500)

    plt.subplots(1, 1, figsize=(8, 6))  # , frameon=False)
    if dist:
        distS, distU = D2_k(alpha, gamma, Uk, Pi, k, 0, 0, unspliced, spliced, scaling)
        d = distS ** 2 + distU ** 2
        plt.scatter(spliced, unspliced, c=np.log1p(d), s=10)
        plt.colorbar()
    else:
        plt.scatter(spliced, unspliced, color="orange")
        plt.scatter(spliced[k], unspliced[k], color="blue")
    u_range = np.arange(0, Uk + (Uk / 100), Uk / 100)
    plt.plot(S(alpha, gamma, U0, S0, u_range), u_range, color="blue")
    s_down = S(0, gamma, Uk, Sk, u_range)
    plt.plot(s_down, u_range, color="orange")

    u_steady = np.array([0, u_range[s_down == np.max(s_down)]])
    plt.plot((1 / gamma) * u_steady, u_steady, color="grey", alpha=.5)

    plt.quiver(spliced[k], unspliced[k], (S(alpha, gamma, 0, 0, Pi) - spliced)[k], (Pi - unspliced)[k], **kwargs)
    plt.quiver(spliced[~k], unspliced[~k], (S(0, gamma, Uk, Sk, np.array(Pi)) - spliced)[~k], (Pi - unspliced)[~k],
               **kwargs)
    plt.xlabel("spliced")
    plt.ylabel("unspliced")
    plt.show()
