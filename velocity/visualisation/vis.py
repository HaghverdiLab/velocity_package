from scipy.spatial import cKDTree
from vis_utils import *


def project_velocities(Y_data, X_current, X_future, n_neighbors=100, row_norm=True,
                       method="nystrom_modif",
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
    method: 'str' (default: 'nystrom_modif')
        Which method to use. Options include "nystrom" for the original nystrom method and "nystrom_modif" for our
        method as shown in the RNA velocity paper.
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

    elif np.all(percent_velo < .01):  # probs too small steps
        print("Warning: Some velocity steps are very small, this could cause issues in the method.")
        if not force_no_scale:
            print("Scaling velocities, set \"force_no_scale=True\" to stop this.")
            v = (X_future - X_current) / (np.max(percent_velo) * 3)
            X_future = X_current + v

    print("Projecting velocities using " +
          ("our modified nystrom approach " if method == "nystrom_modif" else "nystrom approach")+".")

    if method == "nystrom_modif":
        # first calculate W=P^-1*Y
        # get P
        # we restrict to top k NN for speed up
        # it is important that there is no duplicate row in P, s.t. we can calculate the inverse
        first, drop, unique = get_duplicate_row(X_current)  # bit slow
        if len(drop) > 0:
            print("Note: " + str(len(drop)) + " duplicate row(s) found in X_current. Continuing...")

        NN = cKDTree(X_current[unique]).query(x=X_current[unique], k=n_neighbors, n_jobs=1)[1]
        P = d2p_NN(pairwise_distances(X_current[unique], metric="euclidean"), NN, row_norm=row_norm)
        # calculate W=P^-1*Y
        W = nystrom(P, Y_data[unique])
        # get P_2 the transition probability to future states
        # todo: better way of handling duplicate rows here. Just because a row is duplicate in X_current does not mean
        #       it is duplicate in X_future.
        NN = cKDTree(X_current[unique]).query(x=X_future[unique], k=n_neighbors, n_jobs=1)[1]
        d2 = pairwise_distances(X_future[unique], X_current[unique], metric="euclidean")
        P_2 = d2p_NN(d2, NN, row_norm=row_norm)
        Y_future = np.dot(P_2, W)
        if len(drop) > 0:
            # reinsert duplicate rows, so that the array of future states has the same shape as current states
            Y_future = np.insert(Y_future, drop - np.arange(0, len(drop), 1), Y_future[first], axis=0)
    else:
        NN = cKDTree(X_current).query(x=X_future, k=n_neighbors, n_jobs=1)[1]
        d2 = pairwise_distances(X_future, X_current, metric="euclidean")
        P_2 = d2p_NN(d2, NN, row_norm=row_norm)
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
