# Speeding up K-Nearest Neighbors Classification with Data Space Latticing and Precomputation using Distance Weighting

Alan Zhu and Leonardo Valli

Thomas Jefferson High School for Science and Technology

Machine Learning Quarter 2 Project

## Abstract
The KNN classifier is a simple supervised machine learning algorithm that performs well for 
classifying many clustered datasets. However, one major flaw with the KNN algorithm is its 
O(N) classification time complexity, making KNN impracticable for large datasets. In this 
project, we use data space latticing to enable O(1) classification using KNN. To speed up the 
latticing process, we use a tree-based proximity search method (also known as approximate 
nearest-neighbors, or ANN). We demonstrate that these techniques make KNN classification 
significantly faster without sacrificing accuracy.

## Reports
The project report paper is available in PDF and DOCX formats, and the project presentation is available in PDF and PPTX formats.

## Running the code

The `knn.py` file contains all the code for thsi project. Running the Python file will, by default, generate a KNN lattice trained on the provided `all-cfs_subset.csv` dataset, and then classify it using the lattice model. The code will output the accuracy, confusion matrices, and build and test times. 

### Customizing the code
- To train the code on your own dataset, edit the `FILE` variable in line 12.
- To adjust various hyperparameters, including `K` and the number of lattice points, edit lines 31-33. 
- To change the classification model, edit line 155. Available options are:
    - `classify_lattice` (default): Classify using the lattice model.
    - `classify_tree`: Classify using only the ANN tree.
    - `classify`: Classify using vanilla KNN.
- To specify your own weighted distance metric, edit the function on line 115. 
- Because building the lattice can take several minutes, pre-generated lattice files are provided for the `all-cfs_subset.csv` dataset. 
    - For the 3-point-dimension lattice model (3^12 points), extract `all-cfs_subset-lattice-3.tar.xz` to `all-cfs_subset-lattice-3.pkl`. 
    - For the 4-point-dimension lattice model (4^12 points), extract `all-cfs_subset-lattice-4.tar.xz` to `all-cfs_subset-lattice-4.pkl`. 
