from velocity.tools.fit_utils import *
from velocity.tools.kappa import *
from sklearn.preprocessing import normalize


def recover_reaction_rate_pars(adata, use_raw, n=100, key="fit", fit_scaling=True, parallel=True, n_cores=None,
                               n_parts=None, inplace=True):
    unspliced, spliced = adata.layers["unspliced" if use_raw else "Mu"], adata.layers["spliced" if use_raw else "Ms"]
    if parallel:
        alpha, beta, gamma, U_switch, scaling, likelihood, Pi = fit_all_parallel(unspliced, spliced, n=n,
                                                                                 n_cores=n_cores, n_parts=n_parts,
                                                                                 fit_scaling=fit_scaling)
    else:
        alpha, beta, gamma, U_switch, scaling, likelihood, Pi = fit_all(unspliced, spliced, n=n,
                                                                        fit_scaling=fit_scaling)

    if not inplace:
        adata = adata.copy()
    # write to adata object
    adata.var[key + "_alpha"] = alpha
    adata.var[key + "_beta"] = beta
    adata.var[key + "_gamma"] = gamma
    adata.var[key + "_U_switch"] = U_switch
    adata.var[key + "_likelihood"] = likelihood
    adata.layers[key + "_Pi"] = Pi
    if fit_scaling:
        adata.var[key + "_scaling"] = scaling
    if not inplace:
        return adata


def fit_all(unspliced, spliced, n=300, fit_scaling=True):
    n_entries = unspliced.shape[1]
    alpha, beta, gamma, U_switch = np.zeros(n_entries), np.zeros(n_entries), np.zeros(n_entries), np.zeros(n_entries)
    likelihood, scaling = np.zeros(n_entries), np.ones(n_entries)
    Pi = np.zeros(unspliced.shape)

    for i in range(n_entries):
        U_, S_ = unspliced[:, i], spliced[:, i]
        Ms = np.max(S_)
        a, b, g, Uk, sc, lik, Pi_ = fit(U_ / Ms, S_ / Ms, n=n, fit_scaling=fit_scaling)
        alpha[i], beta[i], gamma[i], U_switch[i], scaling[i], likelihood[i] = a, b, g, Uk, sc, lik
        Pi[:, i] = Pi_
    return alpha, beta, gamma, U_switch, scaling, likelihood, Pi


def fit_all_parallel(unspliced, spliced, n=100, n_cores=None, n_parts=None, fit_scaling=True):
    from joblib import Parallel, delayed

    n_entries = unspliced.shape[1]
    if n_cores is None:
        import os
        n_cores = os.cpu_count()

    # preparing input for parallel calling
    # we split the data in n_parts chunks that are run in parallel on n_cores cores
    # note: this is faster than running each gene individually because of overhead when starting new processes

    if n_parts is None:
        n_parts = int((n_entries / n_cores) * 4)

    # prepare input
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
    def fit_helper(pars_):
        out = []
        for p in pars_:
            out.append(fit(p[0], p[1], n, fit_scaling))
        return out

    result = Parallel(n_jobs=8)(delayed(fit_helper)(pars[i]) for i in output)

    # format result
    r = []
    for i in result:
        r.extend(i)
    r = np.array(r)

    alpha, beta, gamma, U_switch, scaling, likelihood = r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5]
    Pi = np.stack(r[:, 6]).T

    return alpha, beta, gamma, U_switch, scaling, likelihood, Pi


def fit(unspliced, spliced, n=50, fit_scaling=True, fit_kappa=True, kappa_mode="u"):
    max_u, max_s = np.quantile(unspliced, .99), np.quantile(spliced, .99)
    sub = (unspliced > 0) & (spliced > 0) & ((unspliced > 0.05 * max_u) | (spliced > 0.05 * max_s))

    # subset of cells is used for fitting
    spliced_subset, unspliced_subset = spliced[sub], unspliced[sub]
    scaling = np.std(spliced_subset) / np.std(unspliced_subset) if fit_scaling else 1
    if np.sum(sub) > 100:  # recoverable

        # fit initialisation
        U0, S0, i = 0, 0, 3
        max_u = max_u * scaling
        alpha, gamma = max_u, max_u / max_s
        # fit
        if fit_scaling:
            x0 = np.array([alpha, gamma, alpha * .99, scaling])
            bounds = ((alpha / i, alpha * i), (gamma / i, gamma * i), (0, None), (scaling / 10, scaling * 10))
        else:
            x0 = np.array([alpha, gamma, alpha * .99])
            bounds = ((alpha / i, alpha * i), (gamma / i, gamma * i), (0, None))
        res1 = opt.minimize(cost_wrapper_fastPi_scaling if fit_scaling else cost_wrapper_fastPi,
                            x0=x0,
                            args=
                            (U0, S0, unspliced_subset, spliced_subset, n) if fit_scaling else
                            (U0, S0, unspliced_subset, spliced_subset, n),
                            bounds=bounds,
                            method="Nelder-Mead", tol=1e-8)
        if fit_scaling:
            alpha, gamma, Uk, scaling = res1.x
        else:
            alpha, gamma, Uk = res1.x
        if Uk >= alpha * .99:
            Uk = alpha * .99

        # get final assignments of the cells
        cost_scaling = np.std(spliced_subset) / np.std(unspliced_subset * scaling)
        # k = np.zeros(spliced.shape)
        k = (unspliced * scaling) > (gamma * spliced)
        # k = k.astype("bool")
        Pi = np.zeros(unspliced.shape)
        sub = (unspliced > 0) | (spliced > 0)
        Pi[sub] = get_Pi_full(alpha, gamma, k[sub], U0, S0, Uk, (unspliced * scaling)[sub], spliced[sub], cost_scaling)
        lik = get_likelihood(alpha, gamma, U0, S0, Uk, spliced, unspliced * scaling,
                             weight=cost_scaling, Pi=Pi, k=k)

        if fit_kappa:

            ignore = .1  # removed bc the density approximation becomes tricky towards the limits
            upper = (Pi > ignore * Uk)
            lower = (Pi < (1 - ignore) * Uk)
            sub = upper & lower
            # at least 30% of the cells need to be in the considered transient state for kappa recovery
            if np.sum(k & sub) > 0.30 * np.sum(sub):
                beta = get_kappa(alpha, 1, gamma, Pi, None, k & sub, "up", "u")
            elif np.sum((~k) & sub) > 0.30 * np.sum(upper & lower):
                beta = get_kappa(0, 1, gamma, Pi, None, (~k) & sub, "down", "u")
            else:
                beta = np.nan
                lik = 0
        else:
            beta = 1
        alpha *= beta
        gamma *= beta
        if True:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            plot_kinetics(alpha / beta, gamma / beta, spliced, unspliced * scaling, Uk, weight=cost_scaling, k=k, Pi=Pi,
                            dist=False, ax=ax)
    else:
        alpha, beta, gamma, Uk, scaling, lik = np.nan, np.nan, np.nan, np.nan, 0, 0
        Pi = np.zeros(spliced.shape)
    return alpha, beta, gamma, Uk, scaling, lik, Pi


import matplotlib.pyplot as plt

kwargs = {"scale": 1, "angles": "xy", "scale_units": "xy", "edgecolors": "k",
          "linewidth": 0.01, "headlength": 4, "headwidth": 5, "headaxislength": 3, "alpha": .3}


def plot_kinetics_wrapper(adata, gene, use_raw=False, key="fit", n_cols=1, n_rows=1):
    if (not isinstance(gene, list)) & (not isinstance(gene, np.ndarray)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        gene = [gene]
        axs = [ax]
    elif len(gene) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        gene = [gene]
        axs = [ax]
    else:
        fig, axs = plt.subplots(1, len(gene), figsize=(6*len(gene), 5))
    for i, g in enumerate(gene):
        unspliced, spliced = adata[:, g].layers["unspliced" if use_raw else "Mu"][:, 0], adata[:, g].layers[
                                                                                             "spliced" if use_raw else "Ms"][
                                                                                         :, 0]
        alpha, gamma, beta = adata[:, g].var[key + "_alpha"][0], adata[:, g].var[key + "_gamma"][0], adata[:, g].var[
            key + "_beta"][0]
        Uk, scaling = adata[:, g].var[key + "_U_switch"][0], adata[:, g].var[key + "_scaling"][0]
        plot_kinetics(alpha / beta, gamma / beta, spliced, unspliced * scaling, Uk, dist=True, weight=1, k=None,
                      Pi=None,
                      ax=axs[i])
    # plt.colorbar()
    plt.show()


def plot_kinetics(alpha, gamma, spliced, unspliced, Uk, dist=True, weight=1, k=None, Pi=None, ax=None):
    U0, S0 = 0, 0
    Sk = S(alpha, gamma, U0, S0, Uk)
    if dist and (Pi is not None):
        distS, distU = dist_k(alpha, gamma, Uk, Pi, k, 0, 0, unspliced, spliced, weight)
        d = distS ** 2 + distU ** 2
        ax.scatter(spliced, unspliced, c=np.log1p(d), s=10)

    else:
        ax.scatter(spliced, unspliced, color="darkgrey", s=10)
        # plt.scatter(spliced[k], unspliced[k], color="blue")
    u_range = np.arange(0, Uk + (Uk / 1000), Uk / 1000)
    ax.plot(S(alpha, gamma, U0, S0, u_range), u_range, color="blue")
    s_down = S(0, gamma, Uk, Sk, u_range)
    ax.plot(s_down, u_range, color="orange")

    u_steady = np.array([0, u_range[s_down == np.max(s_down)]], dtype=float)
    ax.plot(u_steady / gamma, u_steady, color="grey", alpha=.5)

    if Pi is not None:
        Pi[Pi > alpha] = alpha
        Si = np.zeros(Pi.shape)
        Si[k] = S(alpha, gamma, 0, 0, Pi[k])
        Si[~k] = S(0, gamma, Uk, Sk, Pi[~k])

        ax.quiver(spliced, unspliced, (Si - spliced), (Pi - unspliced), **kwargs)

    ax.set_xlabel("spliced")
    ax.set_ylabel("unspliced")


def get_velocity(adata, use_raw=True, key="fit", normalise=None, scale=True):
    """Recovers high-dimensional velocity vector from fitted parameters, and saves it under adata.layers["velocity"].

    Parameters
    ----------
    adata: :class:'~anndata.AnnData'
        Annotated data matrix.
    use_raw: 'bool' (default: True)
        Whether to use the raw counts for velocity calculation or the imputed ones (Ms, Mu)
    key: 'str' (default: "fit")
        Key under which the fitted parameters are saved in the anndata object.
        For example with default key, we look for alpha under adata.var["fit_alpha"].
    normalise: 'str' (default: None)
        Whether to normalise the high-dimensional velocity vector. Multiple options are allowed:
            - "L1" for L1-normalisation
            - "L2" for L2-normalisation
            - "std" for scaling s.t. the standard deviation is equal to 1.
    Returns
    -------

    """
    S_, U_ = adata.layers["spliced" if use_raw else "Ms"], adata.layers["unspliced" if use_raw else "Mu"]
    alpha, beta, gamma, scaling = np.array(adata.var[key + "_alpha"]), np.array(adata.var[key + "_beta"]), np.array(
        adata.var[key + "_gamma"]), np.array(adata.var[key + "_scaling"])

    V = (beta * scaling * U_) - (gamma * S_) if scale else (beta * U_) - (gamma * S_)

    if str(key + "_U_switch") in adata.var.columns:
        u_steady = np.array(adata.var[key + "_U_switch"])
        s_steady = S(alpha / beta, gamma / beta, 0, 0, u_steady)
    elif str(key + "_t_") in adata.var.columns:
        u_steady = u_t(alpha, beta, np.array(adata.var[key + "_t_"]), 0)
        s_steady = S(alpha / beta, gamma / beta, 0, 0, u_steady)
    else:
        (u_steady, s_steady) = (alpha / beta, alpha / gamma)
    V[(U_ > u_steady) & (S_ > s_steady)] = 0
    if normalise is not None:
        if normalise == "L1":
            V = normalize(V, norm='l1')
        elif normalise == "L2":
            V = normalize(V, norm='l2')
        elif normalise == "std":
            V /= (np.nanstd(V.flatten()))  # * 10)
    adata.layers["velocity"] = V
