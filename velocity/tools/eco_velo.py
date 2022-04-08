from scipy.spatial import cKDTree
import numpy as np


def find_mutual_nn(orig, future, k=20, n_jobs=1):
    # we want the k NN of the current states (orig) in the future states (future)
    # the returned indices need to be the indices of the future states
    orig_NN = cKDTree(future).query(x=orig, k=k, n_jobs=n_jobs)[1]
    # we want to check that the NN of the future states (future) are also in NN fo the current states (orig)
    # the returned indices need to be the indices of the current states
    future_NN = cKDTree(orig).query(x=future, k=k, n_jobs=n_jobs)[1]
    mnn = np.zeros(orig.shape[0]).astype(int) - 1
    for cell in range(orig.shape[0]):
        cell_nn = orig_NN[cell]

        for i in cell_nn:
            if cell in future_NN[i]:
                mnn[cell] = i
                break
    return mnn
