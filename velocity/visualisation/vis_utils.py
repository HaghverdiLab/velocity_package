import numpy as np
from sklearn.metrics import pairwise_distances
import scipy


def vector_distance(v_t, v_0, metric="cosine"):
    """Calculates the distance between two high-dimensional vectors.
    Possible distances are the cosine distance (metric="cosine")
    and the difference in vector norm (metric="norm_diff").

    Parameters
    ----------
    v_t: 'np.ndarray'
        first n_obs*n_vars high-dimensional vector
    v_0: 'np.ndarray'
        second high-dimensional vector, should have the same shape as v_t
    metric: 'str' (default: "cosine")
        metric to use to compare the vectors
    Returns
    -------
        'np.array' of len n_obs
        contains distance for each observations
    """
    if metric == "cosine":
        return np.array(
            [np.dot(v_0[i], v_t[i].T) / (np.linalg.norm(v_0[i]) * np.linalg.norm(v_t[i])) for i in range(v_t.shape[0])])
    if metric == "norm_diff":
        return np.array([np.linalg.norm(v_0[i]) - np.linalg.norm(v_t[i]) for i in range(v_t.shape[0])])
    if metric == "diff_norm":
        return np.array([np.linalg.norm(v_0[i]-v_t[i]) for i in range(v_t.shape[0])])
    else:
        print("Unknown value chosen for metric. Options include \"cosine\" for cosine distance and \"norm_diff\" "
              "for the difference in vector norm.")


def Hbeta(Di, beta, i=-1):
    """
    Given distances and beta, computes row of transition probability matrix.
    Note: beta corresponds to 1/2\rho^2
    Parameters
    ----------
    Di: 'np.array'
        row of distance matrix
    beta: 'np.array'
        kernel width for each observation ( beta corresponds to 1/2\rho^2 )
    i: 'int' (default: -1)
        Index of observation for which the transition probability should not be calculated, and kept to 0.
        Note: this is bc in some applications the trans p from a cell to itself should be set to 0
    Returns
    -------
        Pi ('np.array') the row of the transition probability matrix.
        entropy (int) the entropy for the given
    """
    Pi = np.zeros(Di.shape)
    n_neighbors = Di.shape[0]
    sum_Pi = 0.0
    for j in range(n_neighbors):
        if j != i:
            Pi[j] = np.exp(-Di[j] * beta)
            sum_Pi += Pi[j]
    if sum_Pi == 0.0:
        sum_Pi = 1e-8  # EPSILON_DBL
    sum_disti_Pi = 0.0
    for j in range(n_neighbors):
        Pi[j] /= sum_Pi
        sum_disti_Pi += Di[j] * Pi[j]
    entropy = np.log(sum_Pi) + beta * sum_disti_Pi
    return Pi, entropy


def d2p_NN(D, NN, u=30, tol=1e-4, row_norm=True):
    """
    d2p_NN: Identifies appropriate sigma's to get k NNs up to some tolerance,
    and turns a distance matrix d to a transition probability matrix P

    Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
    kernel with a certain uncertainty for every datapoint. The desired
    uncertainty can be specified through the perplexity u (default = 30). The
    desired perplexity is obtained up to some tolerance that can be specified
    by tol (default = 1e-4).
    The function returns the final Gaussian kernel in P.

    adapted from: (C) Laurens van der Maaten, 2008 Maastricht University

    Parameters
    ----------
    D: 'np.ndarray'
        n_obs*n_obs distance matrix between observation
    NN: 'np.ndarray'
        n_obs*k list of first k nearest neighbors for each observation
    u: 'int' (default:30)
        perplexity
    tol: 'float' (default:1e-4)
        tolerance
    row_norm: 'bool' (default: True)
        whether to row normalise
    Returns
    -------

    """
    logU = np.log(u)  # note / todo change to log2 if shannon entropy changed back to log2
    beta = np.ones(D.shape[0])  # corresponds to 1/2\rho^2
    n_loop = 50

    # subset distances to nearest neighbors
    D = np.array([D[i, NN[i]] for i in range(D.shape[0])])
    P = np.zeros(D.shape, dtype=np.float64)

    for i in range(D.shape[0]):  # for each row, == for every datapoint
        betamin, betamax = -np.infty, np.infty
        for _ in range(n_loop):
            thisP, HPi = Hbeta(D[i, :], beta[i])
            # range of beta
            # test whether perplexity is whithin tolerance
            Hdiff = HPi - logU
            if np.abs(Hdiff) <= tol:
                break
            # if not, increase or decrease precision
            if Hdiff > 0:  # increase precision
                betamin = beta[i]
                beta[i] = beta[i] * 2 if betamax == np.infty else (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                beta[i] = beta[i] / 2 if betamin == -np.infty else (beta[i] + betamin) / 2
        P[i, :] = thisP  # set row to final value

    if row_norm:
        P = (P.T / np.sum(P, axis=1)).T

    # transform back to n*n transition p matrix
    # todo find better way to do this, this must be a lil slow
    P_ = np.zeros((D.shape[0], D.shape[0]), dtype=np.float64)
    for i in range(D.shape[0]):
        P_[i, NN[i]] = P[i]

    return P_


#################
# diffusion map #
#################

def get_trans_p(D, sigma, density_norm=None):
    """Computes the transition probability matrix as needed for the diffusion map calculation.

    Parameters
    ----------
    D: 'np.ndarray'
        n_obs*n_obs pairwise distance matrix
    sigma: 'int'
        kernel width parameter
    density_norm: 'np.array' or 'None' (default: None)
        Contains pre-computed row-normalisation values if available.
        Note: this is needed to row-normalise the trans. prob. matrix to the future states by the same values as the
        trans. prob. matrix between current states.
    Returns
    -------
        Transition probability matrix.
        'np.ndarray' n_obs*n_obs
    """
    # get transition matrix
    ed2 = np.exp(-D / sigma ** 2)  # S1)
    z_x = np.sum(ed2, axis=0)
    z_x_z_y = None
    if density_norm is None:
        t_p = 1 / z_x * ed2
    else:
        if density_norm is True:  # densitiy normalized transition matrix:
            z_x_z_y = z_x[:, np.newaxis].dot(z_x[:, np.newaxis].T)
        else:  # densitiy normalized transition matrix given z_x_z_y
            z_x_z_y = density_norm
        z_tilde = ed2 / z_x_z_y
        np.fill_diagonal(z_tilde, 0)
        z_tilde_sum = np.sum(z_tilde, axis=0)
        t_p = 1 / z_tilde_sum * ed2 / z_x_z_y
    np.fill_diagonal(t_p, 0)
    return t_p, z_x_z_y


def diffmap_eigen(X_current, sigma=50):
    d2 = pairwise_distances(X_current, metric="euclidean")
    # get transition matrix
    t_p, dens_norm = get_trans_p(d2, sigma, density_norm=True)
    # get eigen decomp of transition matrix
    evals, evecs = scipy.sparse.linalg.eigsh(t_p, k=10)
    return evals[::-1], evecs[:, ::-1], dens_norm


def add_proj(X_current, X_future, evecs, evals, sigma, dens_norm):
    d2 = pairwise_distances(X_current, X_future, metric="euclidean")
    t_p, _ = get_trans_p(d2, sigma, density_norm=dens_norm)
    return (evecs.T.dot(t_p)).T / evals
