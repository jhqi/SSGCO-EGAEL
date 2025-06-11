import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import (
    setup_seed,
    random_sample_pixel_from_sp,
    load_data,
    load_label,
    load_adj,
    find_k_nearest_neighbors,
    get_edges,
    process_adj_dense,
    adjust_learning_rate,
    eval_clustering,
    get_infer_adj_data,
    get_new_adj
)
import yaml
from model import Net
import warnings
import os
import sys

# 设置运行目录
warnings.filterwarnings("ignore")
run_dir = os.path.dirname(sys.argv[0])
if run_dir == "":
    run_dir = "."
os.chdir(run_dir)

# cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")

dataset_name = "IP"  # IP, PU, TR, BO

epochs = 500
init_lr = 0.05
wd = 5e-4
sgd_momentum = 0.9

# 读取配置文件的参数
with open(f"config/{dataset_name}.yml", "r", encoding="utf-8") as yml_file:
    config = yaml.load(yml_file, Loader=yaml.FullLoader)

num_cluster = config["num_cluster"]  # 类簇数量
M = config["M"]  # 超像素数量
L = config["L"]  # SSGCO网络层数
d = config["d"]  # PCA降维后的光谱特征维度
alpha = config["alpha"]  # 簇原型对比损失权重
warm_up_epochs = config["warm_up_epochs"]  # 预训练epoch数量
gamma = config["gamma"]  # 边权更新的动量系数
beta = config["beta"]  # 边权损失的权重
k = config["k"]  # KNN数量

# pixel feature
pixel_feat = load_data(dataset_name)
n_row, n_col, n_band = pixel_feat.shape
pixel_feat = pixel_feat.reshape(n_row * n_col, n_band)

# 标准化
scaler = StandardScaler()
pixel_feat_scaled = scaler.fit_transform(pixel_feat)

# 表格结果的目录
res_dir = "./result/"
os.makedirs(res_dir, exist_ok=True)
res_filename = os.path.join(res_dir, f"{dataset_name}_res.csv")

# prepare label
label_gt, labeled_p_in_sp, sp_map = load_label(dataset_name, num_sp=M)

# superpixel spatial adj
init_adj_spatial = load_adj(dataset_name, M)

# 每个sp中包含哪些pixel
p_in_sp_list = []
for i in range(M):
    ids = np.where(sp_map == i)[0]
    p_in_sp_list.append(ids)

# 重复多次
repeat_time = 5
num_metric = 5  # acc, kappa, nmi, ari, purity
res = np.zeros((repeat_time, num_metric), dtype=np.float32)

for seed in range(repeat_time):
    # 固定随机数种子
    setup_seed(seed)

    # PCA降维
    pca = PCA(n_components=d)
    pixel_feat = pca.fit_transform(pixel_feat_scaled)

    # sp encoding
    sp_feat = np.zeros((M, d), dtype=np.float32)
    for i, ids in enumerate(p_in_sp_list):
        sp_feat[i, :] = np.mean(pixel_feat[ids, :], axis=0)
    sp_feat = torch.from_numpy(sp_feat)

    # 邻接矩阵
    if k > 0:
        adj_spectral = find_k_nearest_neighbors(sp_feat, k)
        adj_spatial = torch.from_numpy(init_adj_spatial.copy())
        adj = adj_spectral + adj_spatial
    else:
        adj = torch.from_numpy(init_adj_spatial.copy())
    # 对称化
    adj = adj + adj.T
    adj = torch.where(adj > 1, 1, adj)

    # 所有边（的两端节点），shape：边数量*2，无向边只计1条
    edge_indices = get_edges(adj)

    # adj对称归一化
    adj = adj.float().to(device)
    adj_norm = process_adj_dense(adj)

    # sp特征to cuda
    sp_feat = sp_feat.to(device)

    # model创建
    model = Net(
        num_cluster=num_cluster,
        num_layer=L,
        in_dim=d,
    ).to(device)

    # 优化器
    optim_params = [
        {"params": model.encoder_q.parameters()},
        {"params": model.projector_q.parameters()},
        {"params": model.predictor.parameters()},
        {"params": model.infer_adj.parameters()},
    ]
    optimizer = SGD(params=optim_params, lr=init_lr, momentum=sgd_momentum, weight_decay=wd)

    # warm up
    for epoch in range(1, warm_up_epochs + 1):
        model.train()

        # 采样像素点作为增强
        aug_feat = random_sample_pixel_from_sp(p_in_sp_list, pixel_feat)
        aug_feat = torch.from_numpy(aug_feat).to(device)

        # 学习率调整
        adjust_learning_rate(optimizer, init_lr, epoch, warm_up_epochs, is_warm_up=True)

        # warm up, 仅进行样本对齐
        loss = model.pretrain_forward(sp_feat, aug_feat, adj_norm)

        # 反向传播更新q分支
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA更新k分支
        model._momentum_update_key_encoder()
        print(f"epoch: {epoch}, warm up stage, loss: {loss.item():.4f}")

    _, _, _, _, _, psedo_labels, center, emb = eval_clustering(
        model, sp_feat, adj_norm, num_cluster, label_gt, labeled_p_in_sp)

    # contrastive clustering
    best_metric = 0
    no_better = 0
    for epoch in range(1, epochs + 1):
        model.train()

        # 构建边权推理的输入, 打分值
        edge_feat, edge_score = get_infer_adj_data(emb, center, edge_indices, psedo_labels)

        # 推理边权
        refined_edge_weight = model.infer_adj(edge_feat)
        refined_edge_weight = refined_edge_weight.view(-1)

        # 边权损失
        edge_weight_loss = F.mse_loss(refined_edge_weight, edge_score)

        # 构建新adj
        adj, adj_norm = get_new_adj(adj, edge_indices, refined_edge_weight, gamma)

        # 采样像素点作为增强
        aug_feat = random_sample_pixel_from_sp(p_in_sp_list, pixel_feat)
        aug_feat = torch.from_numpy(aug_feat).to(device)

        # 学习率调整
        adjust_learning_rate(optimizer, init_lr, epoch, epochs, is_warm_up=False)

        # 样本对齐+中心对比
        contrastive_loss, cluster_loss = model.forward(sp_feat, aug_feat, adj_norm, psedo_labels)
        loss = contrastive_loss + alpha * cluster_loss + beta * edge_weight_loss

        # 反向传播更新q分支
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 清adj梯度
        adj = adj.detach()
        adj_norm = adj_norm.detach()

        # EMA更新k分支
        model._momentum_update_key_encoder()

        print(f"epoch: {epoch}, contrastive clustering stage, loss: {loss.item():.4f}")

        # 更新聚类结果
        acc, kappa, nmi, ari, purity, psedo_labels, center, emb = eval_clustering(
            model, sp_feat, adj_norm, num_cluster, label_gt, labeled_p_in_sp)
        metric = acc + kappa + nmi + ari
        if metric > best_metric:
            best_metric = metric
            best_acc = acc
            best_kappa = kappa
            best_nmi = nmi
            best_ari = ari
            best_purity = purity
            no_better = 0
        else:
            no_better += 1
        print(
            f"epoch:{epoch} | acc:{acc:.4f}/{best_acc:.4f} | kappa:{kappa:.4f}/{best_kappa:.4f} | nmi:{nmi:.4f}/{best_nmi:.4f} | ari:{ari:.4f}/{best_ari:.4f} | purity:{purity:.4f}/{best_purity:.4f}"
        )

        # early stopping
        if no_better >= 60:
            break

    # 保存聚类结果
    res[seed, :] = (best_acc, best_kappa, best_nmi, best_ari, best_purity)
    dic = {
        "seed": [seed],
        "acc": [best_acc],
        "kappa": [best_kappa],
        "nmi": [best_nmi],
        "ari": [best_ari],
        "purity": [best_purity],
    }
    df = pd.DataFrame(dic)
    if not os.path.exists(res_filename):
        df.to_csv(res_filename, index=False, header=True, encoding="utf-8")
    else:
        df.to_csv(res_filename, mode="a", index=False, header=False, encoding="utf-8")

res_mean = np.mean(res, axis=0)
res_std = np.std(res, axis=0)
print("\nfinish!")
print(f"acc: {res_mean[0]:.2f}±{res_std[0]:.2f}")
print(f"kappa: {res_mean[1]:.2f}±{res_std[1]:.2f}")
print(f"nmi: {res_mean[2]:.2f}±{res_std[2]:.2f}")
print(f"ari: {res_mean[3]:.2f}±{res_std[3]:.2f}")
print(f"purity: {res_mean[4]:.2f}±{res_std[4]:.2f}")
