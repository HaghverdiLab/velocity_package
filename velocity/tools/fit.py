import numpy as np
import scipy.optimize as opt


def fit_all(adata, use_raw=True, n=300):
    unspliced, spliced = adata.layers["unspliced" if use_raw else "Mu"], adata.layers["spliced" if use_raw else "Ms"]
    n_genes = adata.shape[1]
    alpha, beta, U_switch, likelihood = np.zeros(n_genes), np.zeros(n_genes), np.zeros(n_genes), np.zeros(n_genes)
    for i in range(adata.shape[1]):
        U_, S_ = unspliced[:, i], spliced[:, i]
        Ms, Mu = np.max(S_), np.max(U_)
        a, g, Uk, lik = fit(U_/Mu, S_/Ms)
        alpha[i], beta[i], U_switch[i], likelihood[i] = a, g, Uk, lik
    return alpha, beta, U_switch, likelihood


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


def D2_full_wrapper(alpha_gamma_Uk, U0, S0, unspliced, spliced, scaling, n=50):
    alpha, gamma, Uk = alpha_gamma_Uk[0], alpha_gamma_Uk[1], alpha_gamma_Uk[2]  # , alpha_gamma_Uk_Sk[3]
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
    std_u, std_s = np.std(unspliced * scaling), np.std(spliced)
    k = (unspliced / spliced) > (Uk / S(alpha, gamma, U0, S0, Uk))
    Pi = get_Pi_fast(alpha, gamma, k, U0, S0, Uk, unspliced, spliced, scaling, 500)  # todo full Pi
    distS, distU = D2_k(alpha, gamma, Uk, Pi, k, U0, S0, unspliced, spliced, scaling)
    distU /= std_u
    distS /= std_s
    distX = distU ** 2 + distU ** 2
    varx = np.mean(distX) - np.mean(np.sign(distS) * np.sqrt(distX)) ** 2  # np.var(np.sign(distS) * np.sqrt(distX))
    varx += varx == 0
    # plt.hist(distU/varx, bins=100)
    # plt.show()

    loglik = (-1 / (2 * len(distS))) * (np.sum(distX) / varx)

    return np.exp(loglik) * (1 / np.sqrt(2 * np.pi * varx))


def fit(unspliced, spliced, n=50):
    sub = (spliced > 0) & (unspliced > 0)
    spliced, unspliced = spliced[sub], unspliced[sub]
    # initialisation
    Ms, Mu = np.max(spliced), np.max(unspliced)
    scaling = Ms / Mu
    alpha, gamma = Mu, Mu / Ms
    U0, S0 = 0, 0
    # fit
    i = 4
    res1 = opt.minimize(D2_full_wrapper,
                        x0=np.array([alpha, gamma, alpha]),
                        args=(U0, S0, unspliced, spliced, scaling, n),
                        bounds=((alpha / i, alpha * i), (gamma / i, gamma * i), (alpha / i, alpha * i)),
                        method="Nelder-Mead")
    alpha, gamma, Uk = res1.x
    if Uk > alpha:
        Uk = alpha

    lik = get_likelihood(alpha, gamma, U0, S0, Uk, spliced, unspliced, scaling)
    return alpha, gamma, Uk, lik


import matplotlib.pyplot as plt

kwargs = {"scale": 1, "angles": "xy", "scale_units": "xy", "edgecolors": "k",
          "linewidth": 0.01, "headlength": 4, "headwidth": 5, "headaxislength": 3, "alpha": .3}


def plot_kinetics(alpha, gamma, spliced, unspliced, k, Uk, Pi, dist=True, scaling=1, U0=0, S0=0):
    Sk = S(alpha, gamma, U0, S0, Uk)

    plt.subplots(1, 1, figsize=(8, 6))  # , frameon=False)
    if dist:
        distS, distU = D2_k(alpha, gamma, Uk, Pi, k, 0, 0, unspliced, spliced, scaling)
        d = distS ** 2 + distU ** 2
        print(get_likelihood(alpha, gamma, 0, 0, Uk, spliced, unspliced, scaling))
        plt.scatter(unspliced, spliced, c=np.log1p(d), s=10)
        plt.colorbar()
    else:
        plt.scatter(unspliced, spliced, color="orange")
        plt.scatter(unspliced[k], spliced[k], color="blue")
    x_range = np.arange(0, Uk, Uk / 100)
    plt.plot([0, Uk], (1 / gamma) * np.array([0, Uk]), color="grey", alpha=.5)
    plt.plot(x_range, S(alpha, gamma, U0, S0, x_range), color="blue")
    plt.plot(x_range, S(0, gamma, Uk, Sk, x_range), color="orange")

    plt.quiver(unspliced[k], spliced[k], (Pi - unspliced)[k], (S(alpha, gamma, 0, 0, Pi) - spliced)[k], **kwargs)
    plt.quiver(unspliced[~k], spliced[~k], (Pi - unspliced)[~k], (S(0, gamma, Uk, Sk, np.array(Pi)) - spliced)[~k],
               **kwargs)
    plt.show()
