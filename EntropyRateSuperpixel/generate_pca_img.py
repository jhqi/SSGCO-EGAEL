import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def gen_pca_img(dataset):
    dataset_dir = "HSI_datasets/"
    if dataset == "IP":
        file_name = "Indian_pines_corrected.mat"
    elif dataset == "PU":
        file_name = "PaviaU.mat"
    elif dataset == "BO":
        file_name = "Botswana.mat"
    elif dataset == "TR":
        file_name = "HSI_Trento.mat"
    img_path = dataset_dir + file_name

    img_mat = sio.loadmat(img_path)
    img_keys = img_mat.keys()
    img_key = [k for k in img_keys if k != "__version__" and k != "__header__" and k != "__globals__"]
    img = img_mat.get(img_key[0]).astype(np.float32)
    n_row, n_col, n_band = img.shape

    # 转2维
    img = img.reshape(n_row * n_col, n_band)

    # 标准化
    scaler = StandardScaler()
    img = scaler.fit_transform(img)
    pca_img_file_name = f"pca_img/{dataset}_pca"

    # PCA降成1维
    n_pca = 1
    pca = PCA(n_components=n_pca)
    img = pca.fit_transform(img)

    # min-max归一化
    scaler = MinMaxScaler()
    img = scaler.fit_transform(img)

    # 转0-255、resize
    img = np.rint(img * 255).astype(np.uint8)
    img = img.reshape(n_row, n_col, n_pca)
    img = np.squeeze(img, axis=2)

    # 保存精确数值
    sio.savemat(f"{pca_img_file_name}.mat", {"img": img})


for dataset in ["IP", "PU", "TR", "BO"]:
    setup_seed(42)
    gen_pca_img(dataset)
