# Code for "Structural-Spectral Graph Convolution with Evidential Edge Learning for Hyperspectral Image Clustering"

## To reproduce
**1. install torch-clustering**
<br>
a package support K-means with PyTorch GPU, see https://github.com/Hzzone/torch_clustering

**2. install other packages**
<br>
for PyTorch, we use 1.12.1+cu113

**3. run**
<br>
`python train.py`

## Switch dataset
In `train.py`, line 39, support Indian Pines (IP), Pavia University (PU), Botswana (BO), and Trento (TR)

## For new dataset
**1. download HSI dataset**
<br>
plz save files to `EntropyRateSuperpixel/HSI_datasets/`, usually include a feature matrix (H\*W\*C), and a groud truth matrix (H\*W)

**2. generate one-channel gray image through PCA**
<br>
see `EntropyRateSuperpixel/generate_pca_img.py`

**3. superpixel segmentation through ERS**
<br>
see `EntropyRateSuperpixel/HSI_ERS_demo.m`, need to define the number of superpixles

**4. generate superpixel spatial adjacency matrix**
<br>
see `EntropyRateSuperpixel/generate_sp_adj.py`

**5. training, clustering, and evaluating**
<br>
see `train.py`, hyperparameters should be set in `config/xxx.yml`
