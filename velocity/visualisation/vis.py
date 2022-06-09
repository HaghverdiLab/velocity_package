from scipy.spatial import cKDTree
from velocity.visualisation.vis_utils import *


def nystroem_project(Y_data, X_current, X_future, n_neighbors=100, row_norm=True,
                     force_no_scale=False):
    """Function to project future states onto a given low-dimensional embedding.

    Parameters
    ----------
    Y_data: 'np.ndarray'
        n_obs*d low-dimensional embedding of observations
    X_current: 'np.ndarray'
        n_obs*n_vars matrix of current states
    X_future: 'np.ndarray'
        n_obs*n_vars matrix of future states
        Corresponds to X_current+velocities
    n_neighbors: 'int'  (default: 100)
        Number of neighbors for which to calculate the transition probability. Since far-away points have very
        low transition probs (~=0), we can assume trans prob = 0 for those.
        Note: select lower values for slower runtime but potentially at the cost of performance.
    row_norm: 'bool' (default: True)
        Whether to row-normalise the transition probability matrix.
    force_no_scale: 'bool' (default:False)
        We automatically check if the future states are too far out of distribution and down / up-scale the velocities
        if so. Set to 'True' to stop scaling. Note that this can result in issues with the projection.
    Returns
    -------
    Matrix of future states projected onto the low-dimensional embedding.
    'np.ndarray' n_obs*d
    """
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

    NN = cKDTree(X_current).query(x=X_future, k=n_neighbors, n_jobs=1)[1]
    # get left row normalisation factors
    P = d2p_NN(pairwise_distances(X_current, metric="euclidean"), NN, row_norm=False)
    D_left = np.diag(1 / np.sqrt(np.sum(P, axis=1)))
    # compute trans P matrix from old to new
    d2 = pairwise_distances(X_future, X_current, metric="euclidean")
    P_2 = d2p_NN(d2, NN, row_norm=False)
    # right normalisation factor
    D_right = np.diag(1 / np.sqrt(np.sum(P_2, axis=1)))
    # normalise trans P matrix
    P_2 = np.dot(np.dot(D_left, P_2), D_right)
    Y_future = np.dot(P_2, Y_data)
    return Y_future


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
