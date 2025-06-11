import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn
import scipy.io as sio
from scipy.sparse import load_npz
from scipy.optimize import linear_sum_assignment
import os
import math
from torch_clustering import PyTorchKMeans
from model import Net
from sklearn.metrics import cohen_kappa_score, accuracy_score, adjusted_rand_score, normalized_mutual_info_score


# 固定随机数种子
def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# 邻接矩阵对称归一化
def process_adj_dense(adj: torch.Tensor):
    # 输入: 无自环, cuda上, torch, 稠密矩阵
    A_tilde = adj.clone().fill_diagonal_(1)
    D_tilde = torch.pow(torch.sum(A_tilde, dim=1), -0.5)
    A_norm = D_tilde.unsqueeze(1) * A_tilde * D_tilde.unsqueeze(0)
    return A_norm


# 读取.mat文件
def read_mat(filename) -> np.ndarray:
    mat = sio.loadmat(filename)
    keys = [k for k in mat.keys() if k != "__version__" and k != "__header__" and k != "__globals__"]
    arr = mat[keys[0]]
    return arr


# 读取邻接矩阵
def load_adj(dataset_name, n_sp):
    adj_spatial_path = f"EntropyRateSuperpixel/sp_adj/{dataset_name}/{dataset_name}_sp_adj_{n_sp}.npz"
    adj_spatial = load_npz(adj_spatial_path)
    adj_spatial = adj_spatial.toarray().astype(np.int8)
    return adj_spatial


# 每个超像素内部随机采样一个pixel
def random_sample_pixel_from_sp(p_in_sp_list, pixel_feat_np):
    n_sp = len(p_in_sp_list)
    sampled_encoding = np.zeros((n_sp, pixel_feat_np.shape[1]), dtype=np.float32)
    for i in range(n_sp):
        ids = p_in_sp_list[i]
        sampled_idx = int(np.random.choice(ids, size=1))
        sampled_encoding[i, :] = pixel_feat_np[sampled_idx, :]
    return sampled_encoding


# 读取hsi图像文件
def load_data(dataset_name):
    data_path = "EntropyRateSuperpixel/HSI_datasets"
    if dataset_name == "IP":
        filename = "Indian_pines_corrected.mat"
    elif dataset_name == "PU":
        filename = "PaviaU.mat"
    elif dataset_name == "TR":
        filename = "HSI_Trento.mat"
    elif dataset_name == "BO":
        filename = "Botswana.mat"
    filename = os.path.join(data_path, filename)
    data = read_mat(filename)
    return data


# 读取hsi的标注和超像素分割结果
def load_label(dataset_name, num_sp):
    data_path = "EntropyRateSuperpixel/HSI_datasets"
    if dataset_name == "IP":
        filename = "Indian_pines_gt.mat"
    elif dataset_name == "PU":
        filename = "PaviaU_gt.mat"
    elif dataset_name == "TR":
        filename = "GT_Trento.mat"
    elif dataset_name == "BO":
        filename = "Botswana_gt.mat"
    filename = os.path.join(data_path, filename)
    label = read_mat(filename).astype(np.int32)
    label = label - (np.min(label) + 1)  # 最小值调整为-1，-1表示未标注，0~C-1是对应类别
    label = label.reshape(-1)
    labeled_ids = np.where(label >= 0)[0]
    label_gt = label[labeled_ids]
    # read sp seg res
    sp_map = (
        read_mat(f"EntropyRateSuperpixel/seg_res/{dataset_name}/{dataset_name}_sp_map_{num_sp}.mat")
        .astype(np.int16)
        .reshape(-1)
    )
    labeled_p_in_sp = sp_map[labeled_ids]
    return label_gt, labeled_p_in_sp, sp_map


# 余弦学习率调整
def adjust_learning_rate(optimizer, init_lr, epoch, epochs, is_warm_up=False):
    """Decay the learning rate based on schedule"""
    if is_warm_up:
        lr = init_lr * epoch / epochs
    else:
        lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
    # encoder_q
    optimizer.param_groups[0]["lr"] = lr
    # projector_q
    optimizer.param_groups[1]["lr"] = lr
    # predictor
    optimizer.param_groups[2]["lr"] = 10 * lr
    # infer adj
    optimizer.param_groups[3]["lr"] = lr


# 获取无向边的两端节点
def get_edges(adj: torch.Tensor):
    # 输入：无自环，对称，torch.int
    # 获取所有边，上三角
    adj_np = adj.numpy()
    upper_triangle_mask = np.triu(np.ones_like(adj_np, dtype=bool), k=1)
    edge_indices = np.argwhere((adj_np > 0) & upper_triangle_mask)
    return edge_indices


def min_max_norm(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)


# 构造用于推理边权的输入数据，每条边：用两端节点到各个簇心距离来表征，即2*n_class维度的向量
def get_infer_adj_data(emb, center, edge_indices, psedo_labels):
    # 样本与簇心余弦相似度
    emb_center_sim = torch.matmul(emb, center.T)
    edge_num = edge_indices.shape[0]
    class_num = emb_center_sim.shape[1]
    edge_feat = torch.zeros((edge_num, 2 * class_num), dtype=torch.float32).to(emb.device)
    edge_feat[:, 0:class_num] = emb_center_sim[edge_indices[:, 0], :]
    edge_feat[:, class_num : 2 * class_num] = emb_center_sim[edge_indices[:, 1], :]

    # 打分，作为边权预测的参考
    # 节点之间的相似度
    emb_emb_sim = torch.matmul(emb, emb.T)
    # 每个节点聚类置信度
    clustering_conf = torch.max(emb_center_sim, dim=1)[0]
    # 每条边节点1的聚类置信度
    node_conf_1 = clustering_conf[edge_indices[:, 0]]
    # 每条边节点2的聚类置信度
    node_conf_2 = clustering_conf[edge_indices[:, 1]]
    # 每条边对应的两个节点相似度，即边的置信度
    edge_conf = emb_emb_sim[edge_indices[:, 0], edge_indices[:, 1]]

    node_conf_1 = min_max_norm(node_conf_1)
    node_conf_2 = min_max_norm(node_conf_2)
    edge_conf = min_max_norm(edge_conf)

    # 每条边两端节点是否预测为同一类别，同类则1，不同类则0
    indicator = (psedo_labels[edge_indices[:, 0]] == psedo_labels[edge_indices[:, 1]]).to(torch.int8)

    # 预测为不同类的位置，置信度翻转
    edge_conf = torch.where(indicator == 0, 1 - edge_conf, edge_conf)

    # w^{emp}
    edge_score = (2 * indicator - 1) * node_conf_1 * node_conf_2 * edge_conf
    edge_score = F.sigmoid(edge_score)

    return edge_feat, edge_score


# adj动量融合
def get_new_adj(ori_adj, edge_indices, new_edge_weight, gamma):
    new_adj = torch.zeros_like(ori_adj, dtype=torch.float32).to(ori_adj.device)
    new_adj[edge_indices[:, 0], edge_indices[:, 1]] = new_edge_weight
    # 对称化
    new_adj = new_adj + new_adj.T
    fused_adj = gamma * ori_adj + (1 - gamma) * new_adj
    fused_adj_norm = process_adj_dense(fused_adj)
    return fused_adj, fused_adj_norm


def eval_clustering(model: Net, sp_feat, adj_norm, num_cluster, label_gt, labeled_p_in_sp):
    model.eval()
    with torch.no_grad():
        emb = model.forward_k(sp_feat, adj_norm)
        emb = F.normalize(emb, dim=1)
        km = PyTorchKMeans(metric="cosine", init="k-means++", n_clusters=num_cluster, n_init=15, verbose=False)
        # psedo_labels = km.fit_predict(emb)
        psedo_labels = km.fit_predict(emb.double())  # 用双精度，否则可能存在不可控的随机性
    sp_pred = psedo_labels.cpu().numpy()
    pixel_pred = sp_pred[labeled_p_in_sp]
    center = km.cluster_centers_
    center = F.normalize(center.float(), dim=1)
    acc, kappa, nmi, ari, purity = clustering_metric(label_gt, pixel_pred, is_refined=False)
    return acc, kappa, nmi, ari, purity, psedo_labels, center, emb


# knn
def find_k_nearest_neighbors(feat: torch.Tensor, k):
    normalized_feat = feat / feat.norm(dim=1, keepdim=True)
    similarity_matrix = torch.mm(normalized_feat, normalized_feat.T)
    similarity_matrix.fill_diagonal_(-float("inf"))
    _, indices = torch.topk(similarity_matrix, k, dim=1)
    N = feat.size(0)
    adj_spectral = torch.zeros((N, N), dtype=torch.int)
    adj_spectral[torch.arange(N).unsqueeze(1), indices] = 1
    adj_spectral = adj_spectral + adj_spectral.T
    adj_spectral = torch.where(adj_spectral > 1, 1, adj_spectral)
    return adj_spectral


def BestMap(L1, L2):
    L1 = L1.flatten(order="F").astype(np.int32)
    L2 = L2.flatten(order="F").astype(np.int32)
    if L1.size != L2.size:
        raise Exception("size(L1) must == size(L2)")
    Label1 = np.unique(L1)
    nClass1 = Label1.size
    Label2 = np.unique(L2)
    nClass2 = Label2.size
    nClass = max(nClass1, nClass2)

    # For Hungarian - Label2 are Workers, Label1 are Tasks.
    G = np.zeros([nClass, nClass]).astype(np.int32)
    for i in range(0, nClass2):
        for j in range(0, nClass1):
            G[i, j] = np.sum(np.logical_and(L2 == Label2[i], L1 == Label1[j]))

    _, c = linear_sum_assignment(-G)
    newL2 = np.zeros(L2.shape, dtype=np.int32)
    for i in range(0, nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def purity_score(y_true, y_pred):
    """Purity score
    Args:
        y_true(np.ndarray): n*1 matrix Ground truth labels
        y_pred(np.ndarray): n*1 matrix Predicted clusters

    Returns:
        float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    # Labels might be missing e.g with set like 0,2 where 1 is missing
    # First find the unique labels, then map the labels to an ordered set
    # 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def clustering_metric(label_gt, label_predict, is_refined=False):
    if not is_refined:
        label_predict = BestMap(label_gt, label_predict)
    acc = accuracy_score(label_gt, label_predict)
    kappa = cohen_kappa_score(label_gt, label_predict)
    nmi = normalized_mutual_info_score(label_gt, label_predict)
    ari = adjusted_rand_score(label_gt, label_predict)
    purity = purity_score(label_gt, label_predict)
    return 100 * acc, 100 * kappa, 100 * nmi, 100 * ari, 100 * purity
