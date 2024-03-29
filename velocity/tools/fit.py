from velocity.tools.fit_utils import *
from velocity.tools.kappa import *
from sklearn.preprocessing import normalize


def recover_reaction_rate_pars(adata, use_raw, n=100, key="fit", fit_scaling=True, parallel=True, n_cores=None,
                               n_parts=None, inplace=True, fit_kappa=True):
    unspliced, spliced = adata.layers["unspliced" if use_raw else "Mu"], adata.layers["spliced" if use_raw else "Ms"]
    if parallel:
        alpha, beta, gamma, U_switch, scaling, likelihood, Pi = fit_all_parallel(unspliced, spliced, n=n,
                                                                                 n_cores=n_cores, n_parts=n_parts,
                                                                                 fit_scaling=fit_scaling,
                                                                                 fit_kappa=fit_kappa)
    else:
        alpha, beta, gamma, U_switch, scaling, likelihood, Pi = fit_all(unspliced, spliced, n=n,
                                                                        fit_scaling=fit_scaling, fit_kappa=fit_kappa)

    if not inplace:
        adata = adata.copy()
    # write to adata object
    adata.var[key + "_alpha"] = alpha.astype(float)
    adata.var[key + "_beta"] = beta.astype(float)
    adata.var[key + "_gamma"] = gamma.astype(float)
    adata.var[key + "_U_switch"] = U_switch.astype(float)
    adata.var[key + "_likelihood"] = likelihood.astype(float)
    adata.layers[key + "_Pi"] = Pi.astype(float)
    if fit_scaling:
        adata.var[key + "_scaling"] = scaling.astype(float)
    if not inplace:
        return adata


def fit_all(unspliced, spliced, n=100, fit_scaling=True, fit_kappa=True):
    n_entries = unspliced.shape[1]
    alpha, beta, gamma, U_switch = np.zeros(n_entries), np.zeros(n_entries), np.zeros(n_entries), np.zeros(n_entries)
    likelihood, scaling = np.zeros(n_entries), np.ones(n_entries)
    Pi = np.zeros(unspliced.shape)

    for i in range(n_entries):
        U_, S_ = unspliced[:, i], spliced[:, i]
        Ms = np.max(S_)
        a, b, g, Uk, sc, lik, Pi_ = fit(U_ / Ms, S_ / Ms, n=n, fit_scaling=fit_scaling, fit_kappa=fit_kappa)
        alpha[i], beta[i], gamma[i], U_switch[i], scaling[i], likelihood[i] = a, b, g, Uk, sc, lik
        Pi[:, i] = Pi_
    return alpha, beta, gamma, U_switch, scaling, likelihood, Pi


def fit_all_parallel(unspliced, spliced, n=100, n_cores=None, n_parts=None, fit_scaling=True, fit_kappa=True):
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
            out.append(fit(p[0], p[1], n, fit_scaling, fit_kappa))
        return out

    result = Parallel(n_jobs=n_cores)(delayed(fit_helper)(pars[i]) for i in output)

    # format result
    r = []
    for i in result:
        r.extend(i)
    r = np.array(r)

    alpha, beta, gamma, U_switch, scaling, likelihood = r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5]
    Pi = np.stack(r[:, 6]).T

    return alpha, beta, gamma, U_switch, scaling, likelihood, Pi


def fit_alpha_gamma(spliced, unspliced, fit_scaling, n):
    scaling = np.std(spliced) / np.std(unspliced) if fit_scaling else 1
    U0, S0, i = 0, 0, 4
    max_u = np.max(unspliced) * scaling
    alpha, gamma = max_u, max_u / np.max(spliced)
    if fit_scaling:
        x0 = np.array([alpha, gamma, alpha * .99, scaling])
        bounds = ((alpha / i, alpha * i), (gamma / i, gamma * i), (0, None), (scaling / i, scaling * i))
    else:
        x0 = np.array([alpha, gamma, alpha * .99])
        bounds = ((alpha / i, alpha * i), (gamma / i, gamma * i), (0, None))
    res1 = opt.minimize(cost_wrapper_fastPi_scaling if fit_scaling else cost_wrapper_fastPi,
                        x0=x0,
                        args=
                        (U0, S0, unspliced, spliced, n) if fit_scaling else
                        (U0, S0, unspliced, spliced, n),
                        bounds=bounds,
                        method="Nelder-Mead", tol=1e-8)
    if fit_scaling:
        alpha, gamma, Uk, scaling = res1.x
    else:
        alpha, gamma, Uk = res1.x
    if Uk >= alpha * .99:
        Uk = alpha * .99

    return alpha, gamma, Uk, scaling


def fit_full_Pi(alpha, gamma, scaling, Uk, spliced, unspliced, k):
    weight = np.std(spliced) / np.std(unspliced)
    Pi = np.zeros(unspliced.shape)
    sub = (unspliced > 0) | (spliced > 0)
    Pi[sub] = get_Pi_full(alpha, gamma, k[sub], 0, 0, Uk, unspliced[sub], spliced[sub], weight)
    return Pi


def fit(unspliced, spliced, n=50, fit_scaling=True, fit_kappa=True, kappa_mode="u"):
    max_u, max_s = np.quantile(unspliced, .99), np.quantile(spliced, .99)
    sub = (unspliced > 0) & (spliced > 0) & ((unspliced > 0.1 * max_u) | (spliced > 0.1 * max_s))

    # subset of cells is used for fitting
    spliced_subset, unspliced_subset = spliced[sub], unspliced[sub]
    if np.sum(sub) > 100:  # recoverable

        # fit alpha, gamma, Uk, scaling
        alpha, gamma, Uk, scaling = fit_alpha_gamma(spliced_subset, unspliced_subset, fit_scaling, n)

        # get final assignments of the cells
        k = (unspliced * scaling) > (gamma * spliced)
        Pi = fit_full_Pi(alpha, gamma, scaling, Uk, spliced, unspliced * scaling, k)
        # get likelihood
        lik = get_likelihood(alpha, gamma, 0, 0, Uk, spliced, unspliced * scaling, Pi=Pi, k=k)

        # get scaling parameter kappa
        if fit_kappa:
            beta = get_kappa(alpha=alpha, beta=1, gamma=gamma, ut=Pi, st=None, u_switch=Uk, k=k, mode="u")
            alpha, gamma = alpha * beta, gamma * beta
        else:
            beta = 1

    else:
        alpha, beta, gamma, Uk, scaling, lik = np.nan, np.nan, np.nan, np.nan, 0, 0
        Pi = np.zeros(spliced.shape)
    return alpha, beta, gamma, Uk, scaling, lik, Pi


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
