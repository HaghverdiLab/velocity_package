import numpy as np

def normalise_layers(adata, mode='combined', norm='L1', unspliced_layer='unspliced', spliced_layer='spliced', total_counts=None):
    
    """
    Normalise layers of choice in Anndata object. You can choose between an L1 and L2 normalisation. 
    Additionally, there is the option to normalise the layers combined (rather than both separately).
    
    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    mode: `str` (default: 'combined')
        Whether to normalise the layers combined ('combined') (total counts of each cell will be
        calculated using both layers) or seperate ('separate') (total counts of each cell will be
        calculated per layer).
    norm: `str` (default: 'L1')
        Whether to apply L1 normalisation ('L1') or L2 normalisation ('L2').
    unspliced_layer: `str` (default: 'unspliced')
        Name of layer that contains the unspliced counts.
    spliced_layer: `str` (default: 'spliced')
        Name of layer that contains the spliced counts.
    total_counts: `list of int` (default: None)
        XXXXX
    
    """
    
    # test if layers are not sparse but dense
    for layer in [unspliced_layer, spliced_layer]:
        if type(adata.layers[layer]) == scipy.sparse.csr.csr_matrix: adata.layers[layer] = adata.layers[layer].todense()
    
    # get total counts and normalize
    if total_counts is not None:
        print("total_counts given")
        mean_counts = int(np.mean(total_counts))
        adata.layers[unspliced_layer] = np.asarray(adata.layers[unspliced_layer]/total_counts*mean_counts)
        adata.layers[spliced_layer] = np.asarray(adata.layers[spliced_layer]/total_counts*mean_counts)
    
    # normalize if total counts are given
    else:
        print("total_counts not given")
        if mode=='combined':
            us_combined = np.concatenate((adata.layers[unspliced_layer], adata.layers[spliced_layer]), axis=1)
            if norm=='L1': total_counts = get_total_counts(us_combined, squared=False)
            if norm=='L2': total_counts = get_total_counts(us_combined, squared=True)
            mean_counts = int(np.mean(total_counts))
            adata.layers[unspliced_layer] = np.asarray(adata.layers[unspliced_layer]/total_counts*mean_counts)
            adata.layers[spliced_layer] = np.asarray(adata.layers[spliced_layer]/total_counts*mean_counts)
          
        if mode=='separate':
            for layer in [unspliced_layer, spliced_layer]:
                if norm=='L1': total_counts = get_total_counts(adata.layers[layer], squared=False)
                if norm=='L2': total_counts = get_total_counts(adata.layers[layer], squared=True)
                mean_counts = int(np.mean(total_counts))
                adata.layers[layer] = np.asarray(adata.layers[layer]/total_counts*mean_counts)
 
    return adata

def get_total_counts(X, squared=False):
    
    """
    Get total counts in each row (cells).
    
    Parameters
    ----------
    X: 'np.ndarray'
        n_obs (cells) * n_vars (genes) matrix
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    squared: 'bool' (default: False)
        Whether to calculate the sum of squared counts (needed for L2 normalisation).
    
    Returns
    -------
    total_counts: `list of int` (default: None)
        List of total counts per cell.     
    """
    
    if squared == False:
        #total_counts = np.squeeze(np.asarray(X.sum(axis=1)))
        total_counts = np.asarray(X.sum(axis=1))
    if squared == True:
        #total_counts = np.squeeze(np.asarray(np.square(X).sum(axis=1)))
        total_counts = np.asarray(np.square(X).sum(axis=1))
    
    return total_counts