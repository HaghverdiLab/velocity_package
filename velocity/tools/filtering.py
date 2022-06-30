import numpy as np
import pandas as pd

def get_matching_state_genes(adata, total_cells=30, perc_cells=5, perc_match=50):
    """
    Function retrieves all the genes in an AnnData object where the prior state matches the 
    state retrieved by the recover_dynamics function for a set percentage of cells.
    
    Parameters
    ----------
    adata: :class:'~anndata.AnnData'
        Annotated data matrix.
    connections_dict: 'dict' 
        Dictionary connecting a cell state (cell type or cluster) to it's parent cell state.
        Example: connections_dict = {'Alpha cells': ['Pre-endocrine']}
    total_cells: 'int' (default: 30)
        Minimum number of cells with set prior state
    perc_cells: 'int' (default: 5)
        Minimum percentage of cells with set prior state
    perc_match: 'int' (default: 50)
        Minimum percentage of match between prior state and retrieved state

    Returns
    -------
    Genes that were selected because the prior state assignment matches the state
    retrieved after dynamics recovery.
    
    Updates `adata` with the following fields.
    **k_method** : `adata.layers` field
        State assignment derived from recovery dynamics with 0=downregulation, 1=upregulation.

    """
    
    adata.layers["k_method"] = (adata.layers["Mu"]*adata.var["fit_scaling"].values) > ((adata.var["fit_gamma"]/adata.var["fit_beta"]).values * adata.layers["Ms"])
    adata.layers["k_method"] = adata.layers["k_method"].astype(int)
    
    selected_genes = []

    for gene in adata.var_names:
        prior_ks = adata[:,gene].layers["k"][adata[:,gene].layers["k"] != -1]
        method_ks = adata[:,gene].layers["k_method"][adata[:,gene].layers["k"] != -1]
        total_cells_with_prior = len(prior_ks)
        perc_cells_with_prior = total_cells_with_prior/len(adata.obs_names)*100
        percentage = (prior_ks == method_ks).sum()/total_cells_with_prior*100
        if ((total_cells_with_prior > total_cells)
            &(perc_cells_with_prior>perc_cells)
            &(percentage>perc_match)):
            selected_genes.append(gene)
    
    return(selected_genes)
    



