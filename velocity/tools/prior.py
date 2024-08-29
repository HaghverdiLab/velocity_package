import math
import numpy as np
import pandas as pd

def set_prior_state(adata, connections_dict, clusterkey="clusters"):
    """
    Using a dictionary with connected cell states prior states are set for upregulation and downregulation.
    
    Parameters
    ----------
    adata: :class:'~anndata.AnnData'
        Annotated data matrix.
    connections_dict: 'dict' 
        Dictionary connecting a cell state (cell type or cluster) to it's parent cell state.
        Example: connections_dict = {'Alpha cells': ['Pre-endocrine']}
    clusterkey: 'int'
        Key under which the clusters or celltypes mentioned in the connection dictionary are 
        saved in the adata object.

    Returns
    -------
    Updates `adata` with the following fields.
    **k** : `adata.layers` field
        State assignment cell for every gene with -1=unassigned, 0=downregulation, 1=upregulation.

    """
    
    k_df = pd.DataFrame(np.zeros(adata.layers["Mu"].shape, dtype=int), columns=adata.var_names)
    
    for idx, gene in enumerate(adata.var_names):
        counts_df = pd.DataFrame({'Ms':adata[:,gene].layers["Ms"].flatten(),
                                  'Mu':adata[:,gene].layers["Mu"].flatten(), 
                                  'clusters':adata.obs[clusterkey].values})
        counts_df["Ms"] = counts_df["Ms"]/max(counts_df["Ms"])*100
        counts_df["Mu"] = counts_df["Mu"]/max(counts_df["Mu"])*100
        counts_df["k"] = -1

        mean_ms = counts_df.groupby('clusters')['Ms'].mean()
        mean_mu = counts_df.groupby('clusters')['Mu'].mean()

        for key in connections_dict.keys():
            x = mean_ms[connections_dict[key][0]]
            y = mean_mu[connections_dict[key][0]]
            dx = mean_ms[key] - x
            dy = mean_mu[key] - y

            if(math.degrees(math.atan(dy/dx)) > 0) & (abs(dx) > 5) & (abs(dy) > 5):
                if(dx > 0): ### upregulation
                    counts_df.loc[(counts_df["clusters"]==key)
                                  &(counts_df["Ms"]>mean_ms[connections_dict[key][0]])
                                  &(counts_df["Mu"]>mean_mu[connections_dict[key][0]])
                                  &(counts_df["Ms"]<mean_ms[key])
                                  &(counts_df["Mu"]<mean_mu[key]),
                                  "k"] = 1 ### set state child cells to upregulation
                    counts_df.loc[(counts_df["clusters"]==connections_dict[key][0])
                                  &(counts_df["Ms"]>mean_ms[connections_dict[key][0]])
                                  &(counts_df["Mu"]>mean_mu[connections_dict[key][0]])
                                  &(counts_df["Ms"]<mean_ms[key])
                                  &(counts_df["Mu"]<mean_mu[key]),
                                  "k"] = 1 ### set state parent cells to upregulation

                if(dx < 0): ### downregulation
                    counts_df.loc[(counts_df["clusters"]==key)
                                  &(counts_df["Ms"]<mean_ms[connections_dict[key][0]])
                                  &(counts_df["Mu"]<mean_mu[connections_dict[key][0]])
                                  &(counts_df["Ms"]>mean_ms[key])
                                  &(counts_df["Mu"]>mean_mu[key]),
                                  "k"] = 0 ### set state child cells to downregulation
                    counts_df.loc[(counts_df["clusters"]==connections_dict[key][0])
                                  &(counts_df["Ms"]<mean_ms[connections_dict[key][0]])
                                  &(counts_df["Mu"]<mean_mu[connections_dict[key][0]])
                                  &(counts_df["Ms"]>mean_ms[key])
                                  &(counts_df["Mu"]>mean_mu[key]),
                                  "k"] = 0 ### set state parent cells to downregulation
        
        k_df[gene] = counts_df["k"]
            
    adata.layers["k"] = k_df



