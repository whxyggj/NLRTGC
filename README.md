# NLRTGC
Nonconvex Low-Rank Tensor Approximation with Graph and Consistent Regularizations for Multi-view Subspace Learning
If you find this code help, please cite:
@article{PAN2023,
title = {Nonconvex low-rank tensor approximation with graph and consistent regularizations for multi-view subspace learning},
journal = {Neural Networks},
year = {2023},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2023.02.016},
url = {https://www.sciencedirect.com/science/article/pii/S0893608023000795},
author = {Baicheng Pan and Chuandong Li and Hangjun Che},
keywords = {Multi-view clustering, Subspace clustering, Spectral clustering, Nonconvex low-rank tensor approximation},
abstract = {Multi-view clustering is widely used to improve clustering performance. Recently, the subspace clustering tensor learning method based on Markov chain is a crucial branch of multi-view clustering. Tensor learning is commonly used to apply tensor low-rank approximation to represent the relationships between data samples. However, most of the current tensor learning methods have the following shortcomings: the information of the local graph is not taken into account, the relationships between different views is not shown, and the existing tensor low-rank representation takes a biased tensor rank function for estimation. Therefore, a nonconvex low-rank tensor approximation with graph and consistent regularizations (NLRTGC) model is proposed for multi-view subspace learning. NLRTGC retains the local manifold information through graph regularization, and adopts a consistent regularization between multi-views to keep the diagonal block structure of representation matrices. Furthermore, a nonnegative nonconvex low-rank tensor kernel function is used to replace the existing classical tensor nuclear norm via tensor-singular value decomposition (t-SVD), so as to reduce the deviation from rank. Then, an alternating direction method of multipliers (ADMM) which makes the objective function monotonically non-increasing is proposed to solve NLRTGC. Finally, the effectiveness and superiority of the NLRTGC are shown through abundant comparative experiments with various state-of-the-art algorithms on noisy datasets and real world datasets.}
}
If you have any questions or wish to communicate, please contact me on panbaicheng@email.swu.edu.cn.
