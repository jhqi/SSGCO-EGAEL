import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import copy


class SSGCO(nn.Module):
    def __init__(
        self,
        encoder_in_dim,
        num_layer,
        init_out_channel,
        init_kernel_size,
    ):
        super(SSGCO, self).__init__()
        self.num_layer = num_layer

        # encoder
        self.cnn_layers = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        self.bn_layers_cnn = nn.ModuleList()
        self.bn_layers_gcn = nn.ModuleList()

        # layers
        in_channels = 1
        out_channels = init_out_channel
        kernel_size = init_kernel_size
        dim = encoder_in_dim
        for _ in range(num_layer):
            self.cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
            )
            self.bn_layers_cnn.append(nn.BatchNorm1d(out_channels))
            dim = dim - kernel_size + 1
            self.gcn_layers.append(GraphConvolution(dim * out_channels, dim * out_channels))
            self.bn_layers_gcn.append(nn.BatchNorm1d(dim * out_channels))
            in_channels = out_channels
            out_channels = out_channels * 2
            kernel_size = max(3, kernel_size - 2)

    def forward(self, x: torch.Tensor, adj_norm):
        x = x.unsqueeze(1)
        for i in range(self.num_layer):
            x = self.cnn_layers[i](x)
            x = self.bn_layers_cnn[i](x)
            x = F.relu(x, inplace=True)
            c, d = x.shape[1], x.shape[2]
            x = x.view(-1, c * d)
            x = self.gcn_layers[i](x, adj_norm)
            x = self.bn_layers_gcn[i](x)
            x = F.relu(x, inplace=True)
            if i < self.num_layer - 1:
                x = x.view(-1, c, d)
        return x


class Net(nn.Module):
    def __init__(
        self,
        num_cluster,
        num_layer,
        in_dim,
        temperature=0.7,
        init_out_channel=16,
        init_kernel_size=7,
        hidden_size=512,
        byol_momentum=0.99,
        sigma=0.001,
    ):
        super(Net, self).__init__()
        self.num_cluster = num_cluster
        self.temperature = temperature
        self.m = byol_momentum
        self.sigma = sigma

        # adj_infer
        self.infer_adj = nn.Sequential(
            nn.Linear(2 * num_cluster, num_cluster),
            nn.BatchNorm1d(num_cluster),
            nn.ReLU(inplace=True),
            nn.Linear(num_cluster, 1),
            nn.Sigmoid(),
        )

        # 计算encoder输出dim
        dim = in_dim
        for i in range(num_layer):
            kernel_size = max(3, init_kernel_size - 2 * i)
            out_channels = init_out_channel * (2**i)
            dim = dim - kernel_size + 1
        encoder_out_dim = dim * out_channels

        # create the encoders
        self.encoder_q = SSGCO(in_dim, num_layer, init_out_channel, init_kernel_size)
        self.projector_q = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, encoder_out_dim),
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.projector_k = copy.deepcopy(self.projector_q)

        self.predictor = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, encoder_out_dim),
        )
        self.q_params = list(self.encoder_q.parameters()) + list(self.projector_q.parameters())
        self.k_params = list(self.encoder_k.parameters()) + list(self.projector_k.parameters())

        for param_q, param_k in zip(self.q_params, self.k_params):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0)
            elif isinstance(
                m,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.GroupNorm,
                    nn.SyncBatchNorm,
                ),
            ):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.q_params, self.k_params):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward_q(self, im_q, adj_norm):
        q = self.encoder_q(im_q, adj_norm)
        q = self.projector_q(q)
        return q

    @torch.no_grad()
    def forward_k(self, im_k, adj_norm):
        k = self.encoder_k(im_k, adj_norm)
        k = self.projector_k(k)
        k = k.detach_()
        return k

    def pretrain_forward_half(self, im_q, im_k, adj_norm):
        q = self.forward_q(im_q, adj_norm)
        noise_q = q + torch.randn_like(q) * self.sigma
        k = self.forward_k(im_k, adj_norm)
        # 样本对齐损失
        contrastive_loss = (2 - 2 * F.cosine_similarity(self.predictor(noise_q), k)).mean()
        return contrastive_loss

    def pretrain_forward(self, im_q, im_k, adj_norm):
        contrastive_loss1 = self.pretrain_forward_half(im_q, im_k, adj_norm)
        contrastive_loss2 = self.pretrain_forward_half(im_k, im_q, adj_norm)
        contrastive_loss = (contrastive_loss1 + contrastive_loss2) / 2
        return contrastive_loss

    def forward_half(self, im_q, im_k, adj_norm, psedo_labels):
        q = self.forward_q(im_q, adj_norm)
        noise_q = q + torch.randn_like(q) * self.sigma
        k = self.forward_k(im_k, adj_norm)
        # 样本对齐损失
        contrastive_loss = (2 - 2 * F.cosine_similarity(self.predictor(noise_q), k)).mean()
        # 计算中心
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        q_centers = self.compute_centers(q, psedo_labels)
        k_centers = self.compute_centers(k, psedo_labels)
        # 聚类中心对比损失
        cluster_loss = self.compute_cluster_loss(q_centers, k_centers, self.temperature, psedo_labels)
        return contrastive_loss, cluster_loss

    def forward(self, im_q, im_k, adj_norm, psedo_labels):
        # 前向传播+计算损失
        contrastive_loss1, cluster_loss1 = self.forward_half(im_q, im_k, adj_norm, psedo_labels)
        contrastive_loss2, cluster_loss2 = self.forward_half(im_k, im_q, adj_norm, psedo_labels)
        contrastive_loss = (contrastive_loss1 + contrastive_loss2) / 2
        cluster_loss = (cluster_loss1 + cluster_loss2) / 2
        return contrastive_loss, cluster_loss

    def compute_centers(self, x, psedo_labels):
        n_samples = x.size(0)
        weight = torch.zeros(self.num_cluster, n_samples).to(x)
        weight[psedo_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)
        return centers

    def compute_cluster_loss(self, q_centers, k_centers, temperature, psedo_labels):
        d_q = q_centers.mm(q_centers.T) / temperature
        d_k = (q_centers * k_centers).sum(dim=1) / temperature
        d_q = d_q.float()
        d_q[torch.arange(self.num_cluster), torch.arange(self.num_cluster)] = d_k
        zero_classes = torch.arange(self.num_cluster).cuda()[
            torch.sum(F.one_hot(torch.unique(psedo_labels), self.num_cluster), dim=0) == 0
        ]
        mask = torch.zeros(
            (self.num_cluster, self.num_cluster),
            dtype=torch.bool,
            device=d_q.device,
        )
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((self.num_cluster, self.num_cluster))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.num_cluster - 1)
        loss = -pos + torch.logsumexp(torch.cat([pos.reshape(self.num_cluster, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.0
        loss = loss.sum() / (self.num_cluster - len(zero_classes))
        return loss
