# scSwinTNet
scSwinTNet: a cell type annotation method for large-scale single-cell RNA-seq data based on shifted window attention.

The annotation of cell types based on single-cell RNA sequencing (scRNA-seq) data is a critical downstream task in single-cell analysis, with significant implications for a deeper understanding of biological processes. Most analytical methods cluster cells by unsupervised clustering, which requires manual annotation for cell type determination. This process is time-consuming and non-repeatable. To accommodate the exponential growth of sequencing cells, reduce the impact of data bias, and integrate large-scale datasets for further improvement of type annotation accuracy, we proposed scSwinTNet. It is a pre-trained cell type annotation tool for scRNA-seq data, which uses self-attention based on shifted windows and enables intelligent information extraction from gene data. We demonstrated the effectiveness and robustness of scSwinTNet by using 399,760 cells from human and mouse tissues. Above all, scSwinTNet is the first attempt to annotate cell types of scRNA-seq data with a pre-trained shifted window attention-based model. It does not require a priori knowledge and accurately annotates cell types without manual annotation.
# Install
* Python 3.7.12
* tensorflow-gpu 2.6.2
* keras 2.6.0
* numpy 1.19.5
* pandas 1.3.5
* scikit-learn 1.0.2
* scipy 1.7.3
# Data availability
The data used in the method are stored in the Releases-Dataset.
