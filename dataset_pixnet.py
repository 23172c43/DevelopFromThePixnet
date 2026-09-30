"""dataset_pixnet.py -- Dataset HER2ST cho PixNet (crop 896x896, spot radius 50px,
augment flip khi train). Chuyen NGUYEN VEN tu pixnet.ipynb (Muc 4: DATASET VA UTILS,
phan Her2STDataset/her2st_collate/_compute_tile_grid), khong doi logic -- chi tach
thanh module rieng va them import can thiet.
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from seed_utils import seed_everything

SEED = 42
seed_everything(SEED)

def _compute_tile_grid(W, H, tile_size, margin):
    stride = max(64, tile_size - 2 * margin)
    max_x = max(0, W - tile_size)
    max_y = max(0, H - tile_size)
    xs = sorted(set(list(range(0, max_x + 1, stride)) + [max_x]))
    ys = sorted(set(list(range(0, max_y + 1, stride)) + [max_y]))
    return [(x, y) for y in ys for x in xs]

class Her2STDataset(Dataset):
    def __init__(self, processed_dir, top_genes_path, sample_list, spot_radius=50, crop_size=896, mode='train'):
        self.img_dir = os.path.join(processed_dir, 'images')
        self.lbl_dir = os.path.join(processed_dir, 'labels')
        self.spot_radius = spot_radius
        self.crop_size = crop_size
        self.mode = mode
        
        with open(top_genes_path, 'r') as f:
            self.target_genes = [line.strip() for line in f if line.strip()]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self._img_size, self._coords_cache, self._genes_cache, self.items = {}, {}, {}, []

        r = spot_radius
        for sid in sample_list:
            img_path = os.path.join(self.img_dir, f"{sid}.jpg")
            if not os.path.exists(img_path): continue
            with Image.open(img_path) as im: W, H = im.size
            self._img_size[sid] = (W, H)
            
            df = pd.read_csv(os.path.join(self.lbl_dir, f"{sid}.csv"))
            coords = df[['cropped_pixel_x', 'cropped_pixel_y']].values.astype(float)
            
            gene_to_idx = {g: i for i, g in enumerate(df.columns.tolist())}
            extract_idx = [gene_to_idx.get(g, -1) for g in self.target_genes]
            
            raw_genes = df.values
            genes = np.zeros((raw_genes.shape[0], len(self.target_genes)), dtype=float)
            for i, target_col_idx in enumerate(extract_idx):
                if target_col_idx != -1: genes[:, i] = raw_genes[:, target_col_idx].astype(float)
                    
            self._coords_cache[sid], self._genes_cache[sid] = coords, genes

            W_pad, H_pad = max(crop_size, W), max(crop_size, H)
            for (cx, cy) in _compute_tile_grid(W_pad, H_pad, crop_size, r):
                rel_x, rel_y = coords[:, 0] - cx, coords[:, 1] - cy
                keep = (rel_x >= r) & (rel_x <= crop_size - r) & (rel_y >= r) & (rel_y <= crop_size - r)
                if keep.any(): self.items.append((sid, cx, cy))

        print(f"[{mode}] Nạp {len(sample_list)} slides -> {len(self.items)} mảnh cắt")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        sample_id, cx, cy = self.items[idx]
        W, H = self._img_size[sample_id]
        coords, genes = self._coords_cache[sample_id], self._genes_cache[sample_id]

        img = Image.open(os.path.join(self.img_dir, f"{sample_id}.jpg")).convert('RGB')
        img_crop = img.crop((cx, cy, min(cx + self.crop_size, W), min(cy + self.crop_size, H)))
        img_tensor = self.transform(img_crop)

        pad_right, pad_bottom = self.crop_size - img_tensor.shape[2], self.crop_size - img_tensor.shape[1]
        if pad_right > 0 or pad_bottom > 0:
            img_tensor = F.pad(img_tensor.unsqueeze(0), (0, pad_right, 0, pad_bottom), mode='replicate').squeeze(0)

        r = self.spot_radius
        rel_x, rel_y = coords[:, 0] - cx, coords[:, 1] - cy
        keep = (rel_x >= r) & (rel_x <= self.crop_size - r) & (rel_y >= r) & (rel_y <= self.crop_size - r)
        rel_coords = np.stack([rel_x[keep], rel_y[keep]], axis=1) if keep.any() else np.zeros((0, 2))
        rel_genes = genes[keep] if keep.any() else np.zeros((0, genes.shape[1]))

        if self.mode == 'train' and len(rel_coords) > 0:
            if random.random() < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
                rel_coords[:, 0] = self.crop_size - rel_coords[:, 0]
            if random.random() < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])
                rel_coords[:, 1] = self.crop_size - rel_coords[:, 1]

        return (img_tensor, torch.tensor(rel_coords, dtype=torch.float32),
                torch.full((len(rel_coords),), float(r), dtype=torch.float32),
                torch.tensor(rel_genes, dtype=torch.float32), sample_id)

def her2st_collate(batch):
    imgs, coords_list, radii_list, genes_list, sids = zip(*batch)
    return torch.stack(imgs, dim=0), list(coords_list), list(radii_list), list(genes_list), list(sids)