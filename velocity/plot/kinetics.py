import matplotlib.pyplot as plt
import numpy as np
from velocity.tools.fit_utils import S

kwargs = {"scale": 1, "angles": "xy", "scale_units": "xy", "edgecolors": "k",
          "linewidth": 0.01, "headlength": 4, "headwidth": 5, "headaxislength": 3, "alpha": .3}


def scatter(adata, gene, use_raw=False, key="fit", n_cols=None, c=None, show_assignments=None,
            show=True, save=None, ticks=False, figsize=None):
    # number of genes plotted simultaneously
    if (not isinstance(gene, list)) & (not isinstance(gene, np.ndarray)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5) if figsize is None else figsize)
        gene = [gene]
        axs = [ax]
    elif len(gene) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5) if figsize is None else figsize)
        axs = [ax]
    else:
        n_cols = len(gene) if n_cols is None else n_cols
        n_rows = np.ceil(len(gene) / n_cols).astype(int)  # if n_cols is not None else 1
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows) if figsize is None else figsize)

    # colors
    if c is None:
        # default color is cell types
        if "clusters_colors" in adata.uns:
            color_dict = dict(zip(adata.obs["clusters"].cat.categories, adata.uns["clusters_colors"]))
            c = adata.obs["clusters"].map(color_dict)
        else:
            c = np.repeat("grey", adata.shape[0])

    row = 0
    for i, g in enumerate(gene):
        if (i % n_cols == 0) & (i > 0):
            row += 1

        unspliced = adata[:, g].layers["unspliced" if use_raw else "Mu"][:, 0]
        spliced = adata[:, g].layers["spliced" if use_raw else "Ms"][:, 0]

        if key + "_alpha" in adata.var.columns:
            alpha, gamma, beta = adata[:, g].var[key + "_alpha"][0], adata[:, g].var[key + "_gamma"][0], \
                                 adata[:, g].var[key + "_beta"][0]
            Uk, scaling = adata[:, g].var[key + "_U_switch"][0], adata[:, g].var[key + "_scaling"][0]
            if show_assignments and (key + "_Pi" in adata.layers):
                Pi = np.array(adata[:, g].layers[key + "_Pi"][:, 0])
            else:
                Pi = None
            alpha /=beta
            gamma /=beta
        else:
            alpha, gamma, beta, Uk, scaling, Pi = None, None, None, None, None, None
        sub = (unspliced > 0) & (spliced > 0)

        plot_kinetics(alpha, gamma, spliced[sub], unspliced[sub] * scaling if scaling is not None else unspliced[sub],
                      Uk,
                      Pi=Pi[sub] if Pi is not None else Pi,
                      ax=axs[i if n_rows == 1 else (row, int(i % n_cols))], c=c[sub],
                      title=gene[i], ticks=ticks)

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300, transparent=True)
    if show:
        plt.show()


def plot_kinetics(alpha, gamma, spliced, unspliced, Uk, k=None, Pi=None, ax=None, c=None,
                  title=None, ticks=False):
    U0, S0 = 0, 0

    ax.scatter(spliced, unspliced, c=c, s=20)

    # plot kinetics
    if (alpha is not None) and (not np.isnan(alpha)):
        Sk = S(alpha, gamma, U0, S0, Uk)
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
        k = unspliced > gamma * spliced
        Si[~k] = S(0, gamma, Uk, Sk, Pi[~k])
        ax.quiver(spliced, unspliced, (Si - spliced), (Pi - unspliced), **kwargs)

    # ax.set_xlabel("spliced")
    # ax.set_ylabel("unspliced")
    if not ticks:
        ax.set_yticks([])
        ax.set_xticks([])
    ax.set_title(title, fontsize=30)
