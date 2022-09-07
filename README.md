# Velocity package

This velocity analysis package provides functions for velocity analyis from single-cell RNA data. We introduce two methods, along with revised processing and visualisation functions.
- In $\kappa$-velo, we solve for the gene-wise reaction rate parameters to recover the high-dimensional velocity vector. We also present a new visualisation method to faithfully represents the velocities on low-dimensional embeddings. 
- In eco-velo, by simply taking the unspliced counts as a proxy of a cell’s future state we estimates cell state velocities directly in the high-dimensional gene space.

The tools in this package are compatible with [scanpy](https://scanpy.readthedocs.io/).

### Instalation

To install the package from GitHub, please use:

     git clone https://github.com/HaghverdiLab/velocity_package
     cd velocity_package
     pip install -e .
     
### How to run

Tutorials explaining the RNA analysis workflow with $\kappa$-velo and eco-velo can be found [here](https://github.com/HaghverdiLab/velocity_notebooks).

### References

Valérie Marot-Lassauzaie, Brigitte Joanne Bouman, et al. [Towards reliable quantification of cell state velocities](https://doi.org/10.1101/2022.03.17.484754 ) bioRxiv (2022) 

### Support

If you found a bug, would like to see a feature implemented, or have a question please open an [issue](https://github.com/HaghverdiLab/velocity_package/issues).
