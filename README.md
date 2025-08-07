# **DiffusionOT** 
![Doc/Overview.jpg](https://github.com/liujuntan/DiffusionOT/blob/main/Doc/Overview.jpg)




**DiffusionOT** is a dynamic **Regularized Unbalanced Optimal Transport (RUOT)** model that reconstructs cellular trajectories and landscape from time-series scRNA-seq snapshots. Additionally, a **Stochastic Trajectory Analysis (STA)** approach is utilized to predict cell fate decisions and estimate cell ancestry. Furthermore, a **gene perturbation** method is introduced to simulate in silico gene knockouts and overexpression experiments. 





# Requirements



The training framework is implemented in PyTorch and Neural ODEs. Given a stable internet connection, it will take several minutes to install these packages:



* pytorch 1.13.1

* scipy 1.10.1

* [TorchDiffEqPack](https://jzkay12.github.io/TorchDiffEqPack/TorchDiffEqPack.odesolver.html) 1.0.1

* [torchdiffeq](https://github.com/rtqichen/torchdiffeq) 0.2.3

* sklearn 1.2.2

* pandas 1.5.3

* seaborn 0.12.2

* matplotlib 3.5.3

* joblib 1.2.0

* scanpy 1.11.2



Recommended: An Nvidia GPU with CUDA support for GPU acceleration.



For generating plots to visualize results, the required packages are listed:



* numpy 1.23.5

* seaborn 0.12.2

* matplotlib 3.5.3





## Input Files

`$Dataset.npy`: data coordinates from different time points. One simulation and four real datasets used for DiffusionOT paper are provided in folder `Input/`. 





# How to use

## Inputs:

`--dataset` Name of the dataset. Options: EMT; Mouse; Zebrafish; Spatial; MISA, default= 'MISA'. \
`--input\_dir` Input Files Directory, default='Input/'. \
`--save\_dir` Output Files Directory, default='Output/'. \
`--timepoints` Time points of data. \
`--niters` Number of traning iterations. \
`--lr` Learning rate. \
`--num\_samples` Number of sampling points per epoch. \
`--hidden\_dim` Dimension of hidden layer. \
`--n\_hiddens` Number of hidden layers for the neural network learning velocity. \
`--activation` Activation function, default= Tanh. \
`--d` Initiated diffusion coefficient, default= 0.001. 





## Outputs:

`ckpt.pth`: save model’s parameters and training errors.

## Downstream analysis

- **Cell velocity and trajectory**
- **Time-varying landscape**
- **Identifying underlying GRNs and growth-related genes**
- **Stochastic Trajectory Analysis (STA)**
  - **Forward process**: Predicting cell fate decision
  - **Backward process**: Estimating cell ancestry
- **Gene perturbation analysis**


## Examples:

* Gene regulatory network (MISA) model

* Epithelial-mesenchymal transition (EMT) scRNA data

* Mouse hematopoietic scRNA data

A Jupyter Notebook of the step-by-step tutorial is accessible from `notebook` directory. 



# Sources

## EMT dataset

Data for the single-cell lung cancer TGFB1-induced epithelial-mesenchymal transition (EMT) (raw data of `EMT.npz`)was downloaded from a Source Data file available at: [Karacosta LG, et al. Mapping lung cancer epithelial-mesenchymal transition states and trajectories with single-cell resolution. Nat Commun 10, 5587 (2019).](https://www.nature.com/articles/s41467-019-13441-6#Sec3042)

## Mouse hematopoietic dataset

Data for the single-cell mouse hematopoietic dataset (raw data of `Mouse.npz`) was downloaded from the NCBI Gene Expression Omnibus (GEO) under accession number GSE140802, or alternatively from: [Weinreb C, Rodriguez-Fraticelli A, Camargo FD, Klein AM. Lineage tracing on transcriptional landscapes links state to fate during differentiation. Science 367, eaaw3381 (2020).](https://github.com/AllonKleinLab/paper-data/tree/master/Lineage\_tracing\_on\_transcriptional\_landscapes\_links\_state\_to\_fate\_during\_differentiation#experiment-3-in-vitro-cytokine-perturbations)



# Acknowledgments

We thank the following projects for their great work to make our code possible: [TIGON](https://github.com/yutongo/TIGON/), [DeepRUOT](https://github.com/zhenyiizhang/DeepRUOT/). We are also grateful for the exciting work in trajectory inference, which has greatly inspired and influenced our work.



# Contact information



Juntan Liu (ISTBI, FDU)-23110850006@m.fdu.edu.cn

Chunhe Li (SCMS, FDU) (Corresponding author)-chunheli@fudan.edu.cn

Peijie Zhou (CMLR, PKU) (Corresponding author)-pjzhou@pku.edu.cn

Qing Nie (DM, UCI) (Corresponding author)-qnie@uci.edu





# Reference

[1] Tong, A., Huang, J., Wolf, G., van Dijk, D. & Krishnaswamy, S. TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics. Proc. Mach. Learn. Res. 119, 9526–9536 (2020).

[2] Sha, Y., Qiu, Y., Zhou, P. & Nie, Q. Reconstructing growth and dynamic trajectories from single-cell transcriptomics data. Nat. Mach. Intell. 6, 25–39 (2024).

[3] Zhang, Z., Li, T. & Zhou, P. Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport. Proc. 13th Int. Conf. Learn. Represent. (2025).

# License

DiffusionOT is licensed under the MIT License, and the code from TIGON used in this project is subject to the MIT License.
```
MIT License

Copyright (c) 2025 Juntan Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
