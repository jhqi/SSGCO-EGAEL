import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix, save_npz
import os


def gen_sp_spatial_adj(dataset, num_sp):
    print(f"start {dataset}, n_sp = {num_sp}")
    os.makedirs(f"sp_adj/{dataset}/", exist_ok=True)
    filename = f"seg_res/{dataset}/{dataset}_sp_map_{num_sp}.mat"
    sp_mat = sio.loadmat(filename)
    sp_key = [k for k in sp_mat.keys() if k != "__version__" and k != "__header__" and k != "__globals__"]
    sp_map = sp_mat[sp_key[0]].astype(np.int16)

    n_row, n_col = sp_map.shape
    adj_spatial = np.zeros((num_sp, num_sp), dtype=np.int8)

    for i in range(n_row):
        for j in range(n_col):
            sp1 = sp_map[i][j]

            # 四联通
            if i - 1 >= 0:
                sp2 = sp_map[i - 1][j]
                adj_spatial[sp1][sp2] = 1
                adj_spatial[sp2][sp1] = 1
            if i + 1 < n_row:
                sp2 = sp_map[i + 1][j]
                adj_spatial[sp1][sp2] = 1
                adj_spatial[sp2][sp1] = 1
            if j - 1 >= 0:
                sp2 = sp_map[i][j - 1]
                adj_spatial[sp1][sp2] = 1
                adj_spatial[sp2][sp1] = 1
            if j + 1 < n_col:
                sp2 = sp_map[i][j + 1]
                adj_spatial[sp1][sp2] = 1
                adj_spatial[sp2][sp1] = 1

    np.fill_diagonal(adj_spatial, 0)

    # 保存sp邻接矩阵，稀疏
    sparse_adj = coo_matrix(adj_spatial)
    save_npz(f"sp_adj/{dataset}/{dataset}_sp_adj_{num_sp}.npz", sparse_adj)

    print(f"finish {dataset}, n_sp = {num_sp}")


if __name__ == "__main__":
    for dataset in ["BO"]:
        for num_sp in range(2900, 5900 + 1, 150):
            gen_sp_spatial_adj(dataset, num_sp)
