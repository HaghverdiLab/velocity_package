import scipy.optimize as opt
from velocity.tools.fit_utils import *


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


def fit_all_parallel(adata, use_raw=False, n=100, n_cores=None, n_parts=None, fit_scaling=True):
    from joblib import Parallel, delayed

    n_entries = adata.shape[1]
    if n_cores is None:
        import os
        n_cores = os.cpu_count()

    # preparing input for parallel calling
    # we split the data in n_parts chunks that are run in parallel on n_cores cores
    # note: this is faster than running each gene individually because of overhead when starting new processes

    if n_parts is None:
        n_parts = int((n_entries / n_cores) * 4)

    # prepare input
    unspliced, spliced = adata.layers["unspliced" if use_raw else "Mu"], adata.layers["spliced" if use_raw else "Ms"]
    pars = []
    for i in range(n_entries):
        U_, S_ = unspliced[:, i], spliced[:, i]
        pars.append((U_, S_))
    pars = np.array(pars)

    # partition
    to_split = range(0, n_entries)
    n_ = int(np.ceil(n_entries / n_parts))
    output = [to_split[i:i + n_] for i in range(0, n_entries, n_)]

    # run fitting
    result = Parallel(n_jobs=8)(delayed(fit_helper)(pars[i], n, fit_scaling) for i in output)

    # format result
    r = []
    for i in result:
        r.extend(i)
    r = np.array(r)

    if fit_scaling:
        alpha, gamma, U_switch, scaling, likelihood = r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4]
    else:
        alpha, gamma, U_switch, likelihood = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        scaling = np.ones(n_entries)

    return alpha, gamma, U_switch, scaling, likelihood


def fit_helper(pars, n, fit_scaling):
    out = []
    for i in pars:
        out.append(fit(i[0], i[1], n, fit_scaling))
    return out


def get_likelihood(alpha, gamma, U0, S0, Uk, spliced, unspliced, cost_scaling):
    k = (unspliced / spliced) > (Uk / S(alpha, gamma, U0, S0, Uk))
    Pi = get_Pi_fast(alpha, gamma, k, U0, S0, Uk, unspliced, spliced, cost_scaling, 500)  # todo full Pi
    distS, distU = D2_k(alpha, gamma, Uk, Pi, k, U0, S0, unspliced, spliced, cost_scaling)

    std_s, std_u = np.std(spliced * cost_scaling), np.std(unspliced)
    distS /= std_s
    distU /= std_u

    distX = distU ** 2 + distS ** 2
    varx = np.var(np.sign(distS) * np.sqrt(distX))
    varx += varx == 0
    n = len(distS)
    loglik = - (1 / (2 * n)) * np.sum(distX / varx)

    return np.exp(loglik) * (1 / np.sqrt(2 * np.pi * varx))


def fit(unspliced, spliced, n=50, fit_scaling=True):
    max_u, max_s = np.quantile(unspliced, .98), np.quantile(spliced, .98)
    sub = (unspliced > 0) & (spliced > 0) & ((unspliced > 0.05 * max_u) | (spliced > 0.05 * max_s))

    spliced, unspliced = spliced[sub], unspliced[sub]
    std_s, std_u = np.std(spliced), np.std(unspliced)
    # initialisation
    scaling = std_s / std_u if fit_scaling else 1  # due to unequal sampling of u and s
    if np.sum(sub) > 100:

        U0, S0 = 0, 0
        # fit
        i = 3

        max_u, max_s = np.max(unspliced * scaling), np.max(spliced)
        alpha, gamma = max_u, max_u / max_s
        if fit_scaling:
            x0 = np.array([alpha, gamma, alpha, scaling])
            bounds = ((alpha / i, alpha * i), (gamma / i, gamma * i), (0, None), (scaling / 10, scaling * 10))
        else:
            x0 = np.array([alpha, gamma, alpha])
            bounds = ((alpha / i, alpha * i), (gamma / i, gamma * i), (0, None))
        res1 = opt.minimize(cost_wrapper_scaling if fit_scaling else cost_wrapper,
                            x0=x0,
                            args=(U0, S0, unspliced, spliced, n),
                            bounds=bounds,
                            method="Nelder-Mead")
        if fit_scaling:
            alpha, gamma, Uk, scaling = res1.x
        else:
            alpha, gamma, Uk = res1.x
        if Uk > alpha:
            Uk = alpha
        cost_scaling = np.std(spliced) / np.std(unspliced * scaling)
        lik = get_likelihood(alpha, gamma, U0, S0, Uk, spliced, unspliced * scaling, cost_scaling=cost_scaling)
        plot_kinetics(alpha, gamma, spliced, unspliced * scaling, Uk, scaling=cost_scaling)

    else:
        alpha, gamma, Uk, scaling, lik = np.nan, np.nan, np.nan, 0, 0
    return alpha, gamma, Uk, scaling, lik


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
    plt.plot(u_steady / gamma, u_steady, color="grey", alpha=.5)

    plt.quiver(spliced[k], unspliced[k], (S(alpha, gamma, 0, 0, Pi) - spliced)[k], (Pi - unspliced)[k], **kwargs)
    plt.quiver(spliced[~k], unspliced[~k], (S(0, gamma, Uk, Sk, np.array(Pi)) - spliced)[~k], (Pi - unspliced)[~k],
               **kwargs)
    plt.xlabel("spliced")
    plt.ylabel("unspliced")
    plt.show()
