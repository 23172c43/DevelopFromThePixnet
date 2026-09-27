"""pixnet_utils.py -- Ham tien ich dung chung cho train/eval PixNet:
aggregate_sparse_spots (gop dense map -> gia tri gene tung spot), PixNetLoss
(MSE + 1-PCC), pcc_per_gene (Pearson tung gene bang numpy). Chuyen NGUYEN VEN tu
pixnet.ipynb (Muc 4: DATASET VA UTILS, phan con lai), khong doi logic -- chi tach
thanh module rieng va them import can thiet.
"""
import torch
import torch.nn as nn
import numpy as np


def aggregate_sparse_spots(dense_map, spot_coords, spot_radii, orig_img_size):
    _, _, H_out, W_out = dense_map.shape
    device = dense_map.device
    N = spot_coords.shape[0]
    scale_factor = orig_img_size / W_out
    scaled_coords = spot_coords / scale_factor
    predictions = []

    for i in range(N):
        cx, cy = scaled_coords[i, 0], scaled_coords[i, 1]
        r = spot_radii[i] / scale_factor
        x_min, x_max = max(0, int(cx - r)), min(W_out, int(cx + r) + 1)
        y_min, y_max = max(0, int(cy - r)), min(H_out, int(cy + r) + 1)
        
        if x_min >= x_max or y_min >= y_max:
            predictions.append(torch.zeros(dense_map.shape[1], device=device))
            continue
            
        y_grid, x_grid = torch.meshgrid(torch.arange(y_min, y_max, device=device), torch.arange(x_min, x_max, device=device), indexing='ij')
        mask = (((x_grid.float() - cx) ** 2 + (y_grid.float() - cy) ** 2) <= (r ** 2)).float().unsqueeze(0)
        predictions.append(torch.log1p(torch.sum(dense_map[0, :, y_min:y_max, x_min:x_max] * mask, dim=(1, 2))))

    return torch.stack(predictions) if len(predictions) > 0 else torch.zeros((0, dense_map.shape[1]), device=device)

class PixNetLoss(nn.Module):
    def __init__(self, lambda_weight=5.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_weight = lambda_weight

    def forward(self, y_pred, y_true):
        loss_mse = self.mse(y_pred, y_true)
        if y_pred.shape[0] < 2: return loss_mse
        
        mean_pred, mean_true = torch.mean(y_pred, dim=0, keepdim=True), torch.mean(y_true, dim=0, keepdim=True)
        vx, vy = y_pred - mean_pred, y_true - mean_true
        eps = 1e-6
        pcc = torch.sum(vx * vy, dim=0) / (torch.sqrt(torch.sum(vx ** 2, dim=0) + eps) * torch.sqrt(torch.sum(vy ** 2, dim=0) + eps))
        return loss_mse + self.lambda_weight * torch.mean(1 - pcc)

def pcc_per_gene(pred, true):
    vx, vy = pred - pred.mean(axis=0, keepdims=True), true - true.mean(axis=0, keepdims=True)
    return (vx * vy).sum(axis=0) / (np.sqrt((vx ** 2).sum(axis=0)) * np.sqrt((vy ** 2).sum(axis=0)) + 1e-8)