from velocity.tools.kappa_utils import *
import numpy as np


def kappa_velo(adata, mode="u", inplace=True, key="fit"):
    """
    Scales the anndata object.
    Parameters
    ----------
    adata: :class:'~anndata.AnnData'
        Annotated data matrix.
    mode: 'str' (default: u)
        Whether to caculate kapas from unspliced counts ("u") or from unspliced and spiced counts ("s")
    inplace: 'bool' (default: True)
        Whether to scale the adata object directly or a copy and return the copy.
    key: 'str' (default: "fit")
        key under which the fitted parameters are saved in the anndata object.
        For example per default "alpha" parameter is searched under adata.var["fit_alpha"].
    Returns
    -------
        adata: scaled anndata object if inplace == False

    """
    # get kappa
    # we get kappa first for down-reg (if there are sufficient cells in that state) then for up-reg, where down-reg
    # did not work. This order is chosen bc alpha=0 in down-reg, meaning we depend on one less fitted parameter.
    kappas = np.array([get_kappa(adata, i, mode=mode, reg="down", key=key) for i in adata.var_names])
    idx = np.where(np.isnan(kappas))[0]
    kappas[idx] = np.array([get_kappa(adata, i, mode=mode, reg="up", key=key) for i in idx])
    # check if any could still not be recovered
    # scale parameters in anndata object
    if not inplace:
        adata = adata.copy()
    adata.var[key + "_beta"] *= kappas
    adata.var[key + "_alpha"] *= kappas
    adata.var[key + "_gamma"] *= kappas
    adata.var[key + "_t_"] /= kappas
    adata.layers[key + "_t"] /= kappas
    adata.var[key + "_kappa"] = kappas
    if not inplace:
        return adata


def get_kappa(adata, gene, mode="u", reg="up", key="fit"):
    """
    Parameters
    ----------
    adata:: :class: '~anndata.AnnData'
        Annotated data matrix on which to compute the kappa estimates
    gene: 'str' or 'int'
        gene name (str) or index (int) for which the kappa estimates should be computed.
    mode: 'str' in ['s', 'u']
        compute the kappa estimates on spliced values (s), unspliced values (u) or on both
    reg: 'str' in ['up', 'down']
        compute the kappa estimates for up- or down- regulation only or for both
    key: 'str' (default: "fit")

    Returns
    -------
    kappa estimate for gene
    """

    # display error if input incorrect
    # mode can be u, s or both to return all u_kappa, all s_kappa or both (simple concatenation)
    if mode not in ["s", "u"]:
        print("error: mode should be \"u\", \"s\"")
        return
    if reg not in ["up", "down", "both"]:
        print("error: reg should be \"up\", \"down\" depending on whether we should calculate the kappa "
              "estimates for up- or down- regulation.")
        return

    # get parameters for each gene
    alpha, beta, gamma, ut, st, up_reg, down_reg = get_pars(adata, gene, key)

    r_ = up_reg if reg == "up" else down_reg

    # at least 30% of the cells need to be in the considered transient state
    if np.sum(r_) > 0.30 * (np.sum(up_reg) + np.sum(down_reg)):
        t_dist, f = get_f_and_delta_t(ut, st, alpha, beta, gamma, r_, reg, mode)
        k = get_slope(t_dist, f)
    else:
        k = np.nan

    return k
