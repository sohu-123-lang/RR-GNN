# -*- coding: utf-8 -*-
"""Node Classification on Cora using GNN with Cross-Entropy Loss and CQR2 (Custom Split Version)"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.model_selection import train_test_split
from community import community_louvain
import networkx as nx

# ================== 新增代码：自定义节点划分 ==================
def custom_node_split(data, split_ratios=(0.6, 0.2, 0.1, 0.1), seed=42):
    """
    Stratified split nodes into train/calibration/val/test sets
    Args:
        split_ratios: (train, calibration, val, test)
    """
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1"
    y = data.y.cpu().numpy()
    indices = np.arange(data.num_nodes)

    # First split: train+calibration vs val+test
    train_cal_ratio = split_ratios[0] + split_ratios[1]
    val_test_ratio = split_ratios[2] + split_ratios[3]
    
    train_cal_idx, val_test_idx = train_test_split(
        indices,
        test_size=val_test_ratio,
        stratify=y,
        random_state=seed
    )

    # Second split: train vs calibration
    cal_ratio = split_ratios[1] / train_cal_ratio
    train_idx, cal_idx = train_test_split(
        train_cal_idx,
        test_size=cal_ratio,
        stratify=y[train_cal_idx],
        random_state=seed
    )

    # Third split: val vs test
    test_ratio = split_ratios[3] / val_test_ratio
    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=test_ratio,
        stratify=y[val_test_idx],
        random_state=seed
    )

    # Create masks
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.calibration_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.calibration_mask[cal_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    # Verify no overlap
    assert (data.train_mask & data.calibration_mask).sum() == 0
    assert (data.train_mask & data.val_mask).sum() == 0
    assert (data.train_mask & data.test_mask).sum() == 0
    assert (data.calibration_mask & data.val_mask).sum() == 0
    assert (data.calibration_mask & data.test_mask).sum() == 0
    assert (data.val_mask & data.test_mask).sum() == 0

    return data

# ================== 修改后的CQR2函数 ==================
def cqr2_new(cal_labels2, cal_labels, cal_lower, cal_upper, test_labels2, test_labels, test_lower, test_upper, alpha):
    cal_scores = np.maximum(
        (cal_labels - cal_upper)/np.abs(cal_labels2),
        (cal_lower - cal_labels)/np.abs(cal_labels2)
    )
    qhat = np.quantile(cal_scores, 1 - alpha, method='higher')
    prediction_sets = [
        test_lower - qhat * np.abs(test_labels2),
        test_upper + qhat * np.abs(test_labels2)
    ]
    cov = ((test_labels >= prediction_sets[0]) & (test_labels <= prediction_sets[1])).mean()
    #print(max(test_upper))
    eff = np.mean((prediction_sets[1] - prediction_sets[0])/max(test_upper))
    return prediction_sets, cov, eff

def run_conformal_regression(cal_labels2, cal_labels, cal_lower, cal_upper,
                             test_labels2, test_labels, test_lower, test_upper, 
                             alpha=0.05):
    """
    修改后的保形预测函数，使用独立校准集
    """
    num_runs = 100
    cov_all = []
    eff_all = []
    
    for k in range(num_runs):
        np.random.seed(k)
        # 随机子采样校准集（保持原始比例）
        n_calib = len(cal_labels)
        idx = np.random.choice(n_calib, size=n_calib, replace=True)
        
        # 运行CQR2
        prediction_sets, cov, eff = cqr2_new(
            cal_labels2[idx], cal_labels[idx],
            cal_lower[idx], cal_upper[idx],
            test_labels2, test_labels,
            test_lower, test_upper,
            alpha
        )
        cov_all.append(cov)
        eff_all.append(eff)

    return np.mean(cov_all), np.mean(eff_all)

# ================== 主程序 ==================
if __name__ == "__main__":
    # Load and preprocess data
    dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    
    # 应用自定义划分 (train:60%, cal:20%, val:10%, test:10%)
    data = custom_node_split(data, split_ratios=(0.6, 0.2, 0.1, 0.1))

    # Louvain clustering
    def louvain_clustering(edge_index, num_nodes):
        edge_index_np = edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(edge_index_np.T)
        partition = community_louvain.best_partition(G)
        clusters = np.zeros(num_nodes, dtype=np.int64)
        for node, community in partition.items():
            clusters[node] = community
        return torch.from_numpy(clusters).long().to(device), len(set(partition.values()))
    
    data.clusters, n_clusters = louvain_clustering(data.edge_index, data.num_nodes)

    # Model definitions (保持不变)
    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, gconv=GCNConv, n_clusters=5):
            super().__init__()
            self.conv1 = gconv(in_channels + n_clusters, hidden_channels)
            self.conv2 = gconv(hidden_channels, out_channels)
            self.n_clusters = n_clusters
            
        def forward(self, x, edge_index, clusters):
            cluster_onehot = F.one_hot(clusters.long(), num_classes=self.n_clusters).float()
            x = torch.cat([x, cluster_onehot], dim=1)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    class UncertaintyModel(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, gconv=GCNConv, n_clusters=5):
            super().__init__()
            self.conv1 = gconv(in_channels + n_clusters, hidden_channels)
            self.conv2 = gconv(hidden_channels, out_channels)
            self.n_clusters = n_clusters

        def forward(self, x, edge_index, clusters):
            cluster_onehot = F.one_hot(clusters.long(), num_classes=self.n_clusters).float()
            x = torch.cat([x, cluster_onehot], dim=1)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x.squeeze()

    # Initialize models
    model1 = GNN(
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=dataset.num_classes,
        n_clusters=n_clusters
    ).to(device)

    model2 = UncertaintyModel(
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=1,
        n_clusters=n_clusters
    ).to(device)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01, weight_decay=5e-4)

    # ================== 修改后的训练函数 ==================
    def train(epoch, val=False):
        if val:
            model1.eval()
            model2.train()
        else:
            model1.train()
            model2.eval()
        
        # Model 1 training
        out1 = model1(data.x, data.edge_index, data.clusters)
        loss1 = F.cross_entropy(out1[data.train_mask], data.y[data.train_mask])
        
        # Model 2 training
        with torch.no_grad():
            pred_probs = F.softmax(out1, dim=1)
            max_probs = pred_probs.max(dim=1)[0]
            uncertainty = 1 - max_probs
        
        out2 = model2(data.x, data.edge_index, data.clusters)
        loss2 = F.mse_loss(out2[data.train_mask], uncertainty[data.train_mask])
        
        if not val:
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
        else:
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
        
        return loss1.item(), loss2.item()

    # ================== 修改后的测试函数 ==================
    @torch.no_grad()
    def test():
        model1.eval()
        model2.eval()
        
        # Get predictions
        out1 = model1(data.x, data.edge_index, data.clusters)
        pred_probs = F.softmax(out1, dim=1)
        pred = pred_probs.argmax(dim=1)
        max_probs = pred_probs.max(dim=1)[0]
        uncertainty = model2(data.x, data.edge_index, data.clusters).squeeze()
        
        # Calculate accuracies
        acc_dict = {}
        for name, mask in [("train", data.train_mask),
                          ("cal", data.calibration_mask),
                          ("val", data.val_mask),
                          ("test", data.test_mask)]:
            correct = pred[mask] == data.y[mask]
            acc_dict[name] = correct.sum().item() / mask.sum().item()
        
        # Prepare conformal prediction data
        cal_data = {
            "labels": data.y[data.calibration_mask].cpu().numpy(),
            "pred": pred[data.calibration_mask].cpu().numpy(),
            "uncertainty": uncertainty[data.calibration_mask].cpu().numpy()
        }
        
        test_data = {
            "labels": data.y[data.test_mask].cpu().numpy(),
            "pred": pred[data.test_mask].cpu().numpy(),
            "uncertainty": uncertainty[data.test_mask].cpu().numpy()
        }
        
        # Run conformal regression
        cov, eff = run_conformal_regression(
            cal_labels2=cal_data["uncertainty"],
            cal_labels=cal_data["labels"],
            cal_lower=cal_data["pred"] - cal_data["uncertainty"],
            cal_upper=cal_data["pred"] + cal_data["uncertainty"],
            test_labels2=test_data["uncertainty"],
            test_labels=test_data["labels"],
            test_lower=test_data["pred"] - test_data["uncertainty"],
            test_upper=test_data["pred"] + test_data["uncertainty"],
            alpha=0.05
        )
        
        return acc_dict, cov, eff

    # Training loop
    best_val_acc = 0
    for epoch in range(1, 71):
        if epoch % 4 == 0:
            loss1, loss2 = train(epoch, val=True)
        else:
            loss1, loss2 = train(epoch)
        
        if epoch % 10 == 0:
            acc_dict, cov, eff = test()
            print(f'Coverage: {cov:.4f}, Efficiency: {eff:.4f}\n')
            
            if acc_dict["val"] > best_val_acc:
                best_val_acc = acc_dict["val"]
                best_model = copy.deepcopy(model1)

    print("Training completed!")
