import numpy as np
from sklearn.preprocessing import normalize


# todo have one function offering different normalisation options
# e.g. mode = L1 or L2
# joined = True or False defining whether matrices should be norm jointly or separately
# etc...
# todo check if sparse / allow operations on sparse matrices

def L1_normalise(adata):
    # merge spliced and unspliced per cell
    us_combined = np.concatenate((adata.layers['spliced'], adata.layers['unspliced']), axis=1)
    # L1 normasization
    us_combined_L2 = normalize(us_combined, norm='l1')
    # replace X, U and S in adata object
    adata.layers['spliced'] = us_combined_L2[:, 0:len(adata.var_names)]
    adata.layers['unspliced'] = us_combined_L2[:, len(adata.var_names):us_combined_L2.shape[1]]
    adata.X = us_combined_L2[:, 0:len(adata.var_names)]

    return adata
