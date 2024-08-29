from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def pca_project(adata, use_raw=False, variance_mean_scale=True, random_state=0, n_pcs=15):
    if "velocity" not in adata.layers:
        print("velocity not found in adata.layers. Please compute velocities first.")
        return
    velocity = adata.layers["velocity"]
    spliced = adata.layers["spliced" if use_raw else "Ms"]
    sub = np.any(np.isnan(np.array(velocity.astype(float))), axis=1)  # velocities need to be non-null for PCA
    sub &= np.any(np.isnan(np.array(spliced.astype(float))), axis=1)
    if np.sum(sub) > 0:
        print("Warning: " + str(np.sum(sub)) + " genes excluded from PCA because the velocities are np.nan. This "
                                               "usually means that the kinetics are not recovered for those genes.")
        print("Continuing.")
        velocity = velocity[:, sub]
        spliced = spliced[:, sub]
    spliced_fut = spliced + velocity
    if variance_mean_scale:
        print("Variance and mean stabilisation of count matrix for PCA.")
        scal = StandardScaler()
        spliced = scal.fit_transform(spliced)
        spliced_fut = scal.transform(spliced_fut)
    print("Calculating PCA.")
    np.random.seed(0)
    pca = PCA(n_components=n_pcs, random_state=random_state)
    pca.fit(spliced)
    pca_pts = pca.transform(spliced)
    print("Projecting future states.")
    # apply pre-trained PCA transformation on scaled future states
    pca_pts_fut = pca.transform(spliced_fut)
    # get velocity vector in PCA space
    pca_v = pca_pts_fut - pca_pts
    print("Saving to annData object.")
    adata.obsm["X_pca"] = pca_pts
    adata.obsm["velocity_pca"] = pca_v
    adata.uns["pca"] = {"n_pcs": n_pcs, "random_state": random_state, "explained_variance":pca.explained_variance_}
