from scipy.spatial import cKDTree
from velocity.project.nystroem_utils import *


def nystroem_project(adata, basis="umap", n_neighbors=100,
                     force_no_scale=False):
    """Function to project future states onto a given low-dimensional embedding.

        Parameters
        ----------
        adata: :class:'~anndata.AnnData'
            Annotated data matrix. Should contain both \"velocity_pca\" and \"X_pca\" in obsm.
        basis: 'str' (default: "umap")
            Observation matrix (=low dimensional embedding) on which to project the velocities. "X_"+basis should be
            in adata.obsm.
        n_neighbors: 'int'  (default: 100)
            Number of neighbors for which to calculate the transition probability. Since far-away points have very
            low transition probs (~=0), we can assume trans prob = 0 for those.
            Note: select lower values for slower runtime but potentially at the cost of performance.
        force_no_scale: 'bool' (default:False)
            We automatically check if the future states are too far out of distribution and down / up-scale the velocities
            if so. Set to 'True' to stop scaling. Note that this can result in issues with the projection.
        Returns
        -------
        saves "velocity_"+basis" in adata.obsm
    """
    if "X_"+basis not in adata.obsm:
        print("\"X_"+basis+"\" not found in adata.obsm. You need to compute the low-dimensional embedding and save it "
                           "in the annData object first.")
        print("Stopping.")
        return
    if ("X_pca" not in adata.obsm) or ("velocity_pca" not in adata.obsm):
        print("\"velocity_pca\" or \"X_pca\" not found in adata.obsm. You need to compute the velocities in pca "
              "space first with \"velocity.tl.pca.pca_project\".")
        print("Stopping.")
        return
    Y_data = adata.obsm["X_"+basis]
    X_current, X_velo = adata.obsm["X_pca"], adata.obsm["velocity_pca"]
    X_future = X_current+X_velo

    # check if future states are not too far out of the distribution of original states
    percent_velo = np.max(np.abs(X_future - X_current), axis=0) / (
            np.max(X_current, axis=0) - np.min(X_current, axis=0))
    if np.any(percent_velo > .3):  # too big steps
        print("Warning: Velocity steps are very big, this could cause issues in the method.")
        if not force_no_scale:
            print("Scaling velocities down, set \"force_no_scale=True\" to stop this.")
            v = (X_future - X_current) / (np.max(percent_velo) * 3)
            X_future = X_current + v

    elif np.all(percent_velo < .05):  # probs too small steps
        print("Warning: Some velocity steps are very small, this could cause issues in the method.")
        if not force_no_scale:
            print("Scaling velocities, set \"force_no_scale=True\" to stop this.")
            v = (X_future - X_current) / (np.max(percent_velo) * 3)
            X_future = X_current + v

    print("Projecting velocities using Nyström approach.")

    NN = cKDTree(X_current).query(x=X_future, k=n_neighbors)[1]
    # get row normalisation factors
    P = d2p_NN(pairwise_distances(X_current, metric="euclidean"), NN, row_norm=False)
    D = np.diag(1 / np.sqrt(np.sum(P, axis=1)))
    # compute trans P matrix from old to new
    d2 = pairwise_distances(X_future, X_current, metric="euclidean")
    P_2 = d2p_NN(d2, NN, row_norm=False)
    # normalise trans P matrix
    P_2 = np.dot(np.dot(D, P_2), D)
    Y_future = np.dot(P_2, Y_data)
    print("Saving to annData object.")
    adata.obsm["velocity_"+basis] = Y_future-Y_data


def diffmap_and_project(X_current, X_future, sigma=50):
    """Function to compute diffusion map and project future states onto the new map.

        Parameters
        ----------
        X_current: 'np.ndarray'
            n_obs*n_vars matrix of current states
        X_future: 'np.ndarray'
            n_obs*n_vars matrix of future states
            Corresponds to X_current+velocities
        sigma: 'int'  (default: 50)
            Kernel width parameter for diffusion map
        Returns
        -------
        diffmap: 'np.ndarray' n_obs*d
            Matrix of cells states projected onto a diffusion map.
        diffmap_fut: 'np.ndarray' n_obs*d
            Matrix of future states projected onto the diffusion map.
        """
    evals, evecs, z_x = diffmap_eigen(X_current, sigma=sigma)
    diffmap = evecs
    diffmap_fut = add_proj(X_current, X_future, evecs, evals, sigma, z_x)
    return diffmap, diffmap_fut
