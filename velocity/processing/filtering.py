import numpy as np
import pandas as pd
import scipy

### TO-DO:
### Select layer in HVGs?

def pearson_residuals(counts, theta=100):
    """
    Computes analytical residuals for NB model with a fixed theta,
    clipping outlier residuals to sqrt(N) as proposed in
    Lause et al. 2021 https://doi.org/10.1186/s13059-021-02451-7

    Parameters
    ----------
    counts: `matrix`
        Matrix (dense) with cells in rows and genes in columns
    theta: `int` (default: 100)
        Gene-shared overdispersion parameter
    """

    counts_sum0 = np.sum(counts, axis=0)
    counts_sum1 = np.sum(counts, axis=1)
    counts_sum = np.sum(counts)

    # get residuals
    mu = counts_sum1 @ counts_sum0 / counts_sum
    z = (counts - mu) / np.sqrt(mu + (np.square(mu) / theta))

    # clip to sqrt(n)
    n = counts.shape[0]
    z[z > np.sqrt(n)] = np.sqrt(n)
    z[z < -np.sqrt(n)] = -np.sqrt(n)

    return z

def get_hvgs(adata, no_of_hvgs=2000, theta=100, layer='spliced'):
    '''
    Function to select the top x highly variable genes (HVGs)
    from an anndata object.

    Parameters
    ----------
    adata
        Annotated data matrix
    no_of_hvgs: `int` (default: 2000)
        Number of HVGs to return
    theta: `int` (default: 100)
        Gene-shared overdispersion parameter used in pearson_residuals
    layer: `str` (default: 'spliced')
        Name of layer that is used to find the HVGs.
    '''

    ### get pearson residuals
    if scipy.sparse.issparse(adata.layers[layer]):
        residuals = pearson_residuals(adata.layers[layer].todense(), theta)
    else:
        residuals = pearson_residuals(adata.layers[layer], theta)

    ### get variance of residuals
    residuals_variance = np.var(residuals, axis=0)
    variances = pd.DataFrame({"variances": pd.Series(np.array(residuals_variance).flatten()),
                              "genes": pd.Series(np.array(adata.var_names))})

    ### get top x genes with highest variance
    hvgs = variances.sort_values(by="variances", ascending=False)[0:no_of_hvgs]["genes"].values

    return hvgs

def get_high_us_genes(adata, minlim_u=3, minlim_s=3, unspliced_layer='unspliced', spliced_layer='spliced'):
    '''
    Function to select genes that have spliced and unspliced counts above a certain threshold. Genes of 
    which the maximum u and s count is above a set threshold are selected. Threshold varies per dataset 
    and influences the numbers of genes that are selected.
    
    Parameters
    ----------
    adata
        Annotated data matrix
    minlim_u: `int` (default: 3)
        Threshold above which the maximum unspliced counts of a gene should fall to be included in the 
        list of high US genes.
    minlim_s: `int` (default: 3)
        Threshold above which the maximum spliced counts of a gene should fall to be included in the 
        list of high US genes.
    unspliced_layer: `str` (default: 'unspliced')
        Name of layer that contains the unspliced counts.
    spliced_layer: `str` (default: 'spliced')
        Name of layer that contains the spliced counts.
    '''
    
    # test if layers are not sparse but dense
    for layer in [unspliced_layer, spliced_layer]:
        if scipy.sparse.issparse(adata.layers[layer]): adata.layers[layer] = adata.layers[layer].todense()
    
    # get high US genes
    u_genes = np.max(adata.layers[unspliced_layer], axis=0) > minlim_u
    s_genes = np.max(adata.layers[spliced_layer], axis=0) > minlim_s
    us_genes = adata.var_names[np.array(u_genes & s_genes).flatten()].values
    
    return us_genes
