# CCUR
Dimension reduction is an essential tool for analyzing high dimensional data. Most existing methods, including principal component analysis (PCA), singular value decomposition (SVD), as well as their extensions, provide principal components that are often linear combinations of features, which are often challenging to interpret. CUR decomposition, another matrix decomposition technique, is a more interpretable and efficient alternative, offers simultaneous feature and sample selection. Despite this, many biomedical studies involve two groups: a foreground (treatment or case) group and a background (control) group, where the objective is to identify features unique to or enriched in the foreground. This need for contrastive dimension reduction is not well addressed by existing CUR methods, nor by contrastive approaches rooted in SVDs. Furthermore, they fail to address a key challenge in biomedical studies: the need for selecting samples unique to the foreground. In this paper, we address this gap by proposing a Contrastive CUR (CCUR), a novel method specifically designed for case-control studies. Through extensive experiments, we demonstrate that CCUR outperforms existing techniques in isolating biologically relevant features as well as identifying sample-specific responses unique to the foreground, offering deeper insights into case-control biomedical data.

## Data 
For Small Molecule and Pathogen, we follow data preprocessing and downloading follows https://github.com/suinleelab/contrastiveVI/blob/main/contrastive_vi/. 

## Files
main.py contains the CCUR method and relevant functions. The following files contains the necessary code the generate each figure in the paper.

### Mouse Protein
This file contains experiments done for Mouse Protein. Data_Cortex_Nuclear.csv contains the data. The code used to generate images in Figures 2 and 5 can be found in this file.

### Small Molecules
This file contains experiments done for Small Molecules. The code used to generate images in Figure 3 and 6 can be found in this file.

### Pathogen
This file contains experiments done for Pathogen. The code used to generate images in Figure 4 and 7 can be found in this file.


## CFS
We compare results with CFS in this folder for each data. For corresponding files, we follow the original paper and its implementation which can be found here https://github.com/suinleelab/CFS/blob/master/contrastive_fs/.

## Simulation
This file contains simulations for both contrastive column and row selection. The code used to generate Figure 1 can be find in these files.





