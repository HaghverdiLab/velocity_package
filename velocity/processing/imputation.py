### load required libraries

import numpy as np
from sklearn.decomposition import PCA #for creating PCAs
from sklearn.preprocessing import StandardScaler #for creating PCAs
from scipy.spatial import cKDTree #for calculating nearest neighbours (part of imputation)


### functions

def impute_counts(adata, n_neighbours = 30, n_pcs = 15, layer_NN = 'spliced', unspliced_layer='unspliced', spliced_layer='spliced'):
    '''
    Function to impute the counts in the unspliced and spliced layer of an adata object. First the 
    function reduces the dimensions of the inputed layer (layer_NN) using PCA to the desired number 
    of dimensions (n_pcs). In this lower dimensional space, a selected number of neighbours (n_neighbours)
    is found for every cell. For every gene, we then impute the counts by taking the average counts 
    of all neighbours. 
    
    Parameters
    ----------
    adata
        Annotated data matrix
    n_neighbours: `int` (default: 30)
        Number of neighbours to use for imputation per cell.
    n_pcs: `int` (default: 3)
        Number of principal components (PCs) to use.
    layer_NN: `str` (default: 'spliced')
        Name of layer that is used to find the neighbours of each cell (after reducing dimension 
        using PCA).
    '''
    
    # scale layer 
    scal = StandardScaler()
    spliced_scaled = scal.fit_transform(adata.layers[layer_NN])
    
    # run PCA
    pca = PCA(n_components=n_pcs)
    pca.fit(spliced_scaled)
    pca_embedding = pca.transform(spliced_scaled)
    
    # find nearest neighbours
    NN = cKDTree(pca_embedding).query(x=pca_embedding, k=n_neighbours, n_jobs=1)[1]
    
    # impute counts using nearest neighbours (NN)
    Mu = np.nanmean(np.array(adata.layers[unspliced_layer])[NN], axis=1)
    Ms = np.nanmean(np.array(adata.layers[spliced_layer])[NN], axis=1)

    # add imputed counts to adata
    adata.layers["Ms"]=Ms
    adata.layers["Mu"]=Mu
    
    return us_genes
