"""run_pixnet_train.py -- Entry point: setup, train, va danh gia full-slide PixNet
tren HER2ST. Chuyen NGUYEN VEN tu pixnet.ipynb (Muc 0: import + HF login, Muc 5: CHUAN
BI HUAN LUYEN, Muc 6: VONG LAP HUAN LUYEN, Muc 7: DANH GIA TRUC QUAN FULL SLIDE), khong
doi logic -- chi tach thanh script rieng, them import tu cac module da tach
(models/PixNet.py, dataset_pixnet.py, pixnet_utils.py, prepare_pixnet_data.py).

Yeu cau chay: prepare_pixnet_data se duoc import ben duoi -- lan dau chay se tu dong
tai + tien xu ly du lieu (giu nguyen cac if-guard idempotent cua ban goc), lan sau
tu dong bo qua neu da co san. Neu muon tach rieng buoc chuan bi du lieu, chay truoc:
    python prepare_pixnet_data.py
roi moi chay:
    python run_pixnet_train.py
"""
import os, math, gc, random, time, glob, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from huggingface_hub import login
from tqdm.auto import tqdm
from getpass import getpass
from seed_utils import seed_everything

SEED = 42
seed_everything(SEED)

# Cố gắng lấy Token từ Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("-> Đã login Hugging Face qua Kaggle Secrets.")
except Exception:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        hf_token = getpass("Nhập Hugging Face access token (cần quyền truy cập MahmoodLab/UNI2-h): ")
    login(token=hf_token)

# ==========================================
# Import cac module da tach (kien truc / dataset / ham tien ich / chuan bi du lieu)
# ==========================================
from models.PixNet import PixNet
from dataset_pixnet import Her2STDataset, her2st_collate
from pixnet_utils import aggregate_sparse_spots, PixNetLoss, pcc_per_gene, compute_regression_metrics
from prepare_pixnet_data import (WORK_DIR, PROCESSED_DIR, TOP_GENES_FILE,
                                  train_patients, val_patients, test_patients)

# ==========================================
# 5. CHUẨN BỊ HUẤN LUYỆN
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

# Heuristic BATCH_SIZE cho Kaggle T4
BATCH_SIZE = 1 if (device.type == 'cuda' and torch.cuda.get_device_properties(0).total_memory / 1e9 < 18) else 2
TARGET_EFFECTIVE_BATCH = 4
ACCUM_STEPS = max(1, round(TARGET_EFFECTIVE_BATCH / BATCH_SIZE))

CROP_SIZE = 896
SPOT_RADIUS = 50
NUM_EPOCHS = 70
BASE_LR = 5e-4
WEIGHT_DECAY = 0
MAX_GRAD_NORM = 0.5
TEST_EVAL_INTERVAL = 5
TRAIN_TILES_PER_EPOCH = 100

CKPT_PATH = os.path.join(WORK_DIR, 'pixnet_v2_original_best.pt')

# --- PATIENT-WISE SPLIT ---
all_sample_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(os.path.join(PROCESSED_DIR, 'images')) if f.endswith('.jpg')])

train_ids = [sid for sid in all_sample_ids if sid[0] in train_patients]
val_ids = [sid for sid in all_sample_ids if sid[0] in val_patients]
test_ids = [sid for sid in all_sample_ids if sid[0] in test_patients]

train_dataset = Her2STDataset(PROCESSED_DIR, TOP_GENES_FILE, train_ids, SPOT_RADIUS, CROP_SIZE, 'train')
val_dataset = Her2STDataset(PROCESSED_DIR, TOP_GENES_FILE, val_ids, SPOT_RADIUS, CROP_SIZE, 'test')

# BẮT BUỘC num_workers=0 trên Kaggle khi đọc ảnh để tránh tràn Shared Memory
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    sampler=RandomSampler(train_dataset, replacement=True, num_samples=TRAIN_TILES_PER_EPOCH),
    num_workers=0, collate_fn=her2st_collate, drop_last=False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, collate_fn=her2st_collate
)

USE_LOWRANK = True
LOW_RANK_K = 64   # chạy 2 phiên: 64 và 256 (cùng seed/split)

model = PixNet(num_genes=250, use_lowrank=USE_LOWRANK, lowrank_k=LOW_RANK_K).to(device)
criterion = PixNetLoss(lambda_weight=5.0).to(device)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)

def lr_lambda(epoch):
    if epoch < 5: return (epoch + 1) / 5
    return 0.5 * (1 + math.cos(math.pi * ((epoch - 5) / max(1, NUM_EPOCHS - 5))))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# Tối ưu cho T4: Ép float16 thay vì bfloat16
AMP_DTYPE = torch.float16
use_scaler = (AMP_DTYPE == torch.float16 and device.type == 'cuda')
scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)



# ==========================================
# 6. VÒNG LẶP HUẤN LUYỆN
# ==========================================
best_pcc_m = -9999
print(f"\n🚀 BẮT ĐẦU TRAIN (PIXNET V2) | BATCH_SIZE: {BATCH_SIZE} | ACCUM_STEPS: {ACCUM_STEPS}")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, n_train = 0.0, 0
    optimizer.zero_grad()

    # Thêm mininterval=2.0 để tránh lỗi IOPub của Kaggle
    for step, (imgs, coords_b, radii_b, genes_b, sid_b) in enumerate(tqdm(train_loader, desc=f"E{epoch+1} Train", leave=False, mininterval=2.0)):
        valid_idx = [i for i in range(len(coords_b)) if coords_b[i].shape[0] > 0]
        if not valid_idx: continue
        imgs = imgs.to(device)

        with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
            dense_map = model(imgs)  
            batch_losses = []
            for i in valid_idx:
                preds_i = aggregate_sparse_spots(dense_map[i:i+1], coords_b[i].to(device), radii_b[i].to(device), CROP_SIZE)
                batch_losses.append(criterion(preds_i, genes_b[i].to(device)))
            loss = torch.stack(batch_losses).mean() / ACCUM_STEPS

        if not torch.isfinite(loss): continue

        if use_scaler: scaler.scale(loss).backward()
        else: loss.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            if use_scaler: scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * ACCUM_STEPS
        n_train += 1

    avg_train_loss = train_loss / max(1, n_train)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # --- ĐÁNH GIÁ (VAL) ---
    if (epoch % TEST_EVAL_INTERVAL == (TEST_EVAL_INTERVAL - 1)) or (epoch == NUM_EPOCHS - 1):
        model.eval()
        val_loss, n_val = 0.0, 0
        all_preds, all_trues = [], []

        with torch.no_grad():
            for imgs, coords_b, radii_b, genes_b, sid_b in tqdm(val_loader, desc=f"E{epoch+1} Val", leave=False, mininterval=2.0):
                valid_idx = [i for i in range(len(coords_b)) if coords_b[i].shape[0] > 0]
                if not valid_idx: continue
                imgs = imgs.to(device)

                with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                    dense_map = model(imgs)
                    for i in valid_idx:
                        preds_i = aggregate_sparse_spots(dense_map[i:i+1], coords_b[i].to(device), radii_b[i].to(device), CROP_SIZE)
                        val_loss += criterion(preds_i, genes_b[i].to(device)).item()
                        n_val += 1
                        all_preds.append(preds_i.float().cpu().numpy())
                        all_trues.append(genes_b[i].float().cpu().numpy())

        avg_val_loss = val_loss / max(1, n_val)
        if all_preds:
            preds_np, trues_np = np.concatenate(all_preds, axis=0), np.concatenate(all_trues, axis=0)
            m = compute_regression_metrics(preds_np, trues_np)
            mse, rmse, mae = m['mse'], m['rmse'], m['mae']
            pcc_f, pcc_s, pcc_m = m['pcc_f'], m['pcc_s'], m['pcc_m']
        else:
            mse = rmse = mae = pcc_f = pcc_s = pcc_m = 0.0

        print(f"[E{epoch+1}/{NUM_EPOCHS}] lr={current_lr:.1e} train={avg_train_loss:.4f} val={avg_val_loss:.4f} | "
              f"RMSE={rmse:.4f} MAE={mae:.4f} PCC@F={pcc_f:.4f} PCC@S={pcc_s:.4f} PCC@M={pcc_m:.4f}", end="")
        
        if pcc_m > best_pcc_m:
            best_pcc_m = pcc_m
            torch.save(model.state_dict(), CKPT_PATH)
            print(f" ⭐ (NEW BEST)")
        else:
            print("")
    else:
        print(f"[E{epoch+1}/{NUM_EPOCHS}] lr={current_lr:.1e} train={avg_train_loss:.4f}")

# ==========================================
# 7. ĐÁNH GIÁ TRỰC QUAN FULL SLIDE TRÊN TẬP TEST
# ==========================================
print("\n--- TIẾN HÀNH ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST ---")
_EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

@torch.no_grad()
def evaluate_full_slide(model, sample_id):
    with open(TOP_GENES_FILE, 'r') as f: target_genes = [line.strip() for line in f if line.strip()]
    img = Image.open(os.path.join(PROCESSED_DIR, 'images', f"{sample_id}.jpg")).convert('RGB')
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'labels', f"{sample_id}.csv"))
    coords = df[['cropped_pixel_x', 'cropped_pixel_y']].values.astype(float)
    
    gene_to_idx = {g: i for i, g in enumerate(df.columns.tolist())}
    extract_idx = [gene_to_idx.get(g, -1) for g in target_genes]
    genes = np.zeros((df.values.shape[0], len(target_genes)), dtype=float)
    for i, t_idx in enumerate(extract_idx):
        if t_idx != -1: genes[:, i] = df.values[:, t_idx].astype(float)

    W_pad, H_pad = max(CROP_SIZE, img.size[0]), max(CROP_SIZE, img.size[1])
    img_tensor = _EVAL_TRANSFORM(img)
    if W_pad > img.size[0] or H_pad > img.size[1]:
        img_tensor = F.pad(img_tensor.unsqueeze(0), (0, W_pad - img.size[0], 0, H_pad - img.size[1]), mode='replicate').squeeze(0)

    stride = max(64, CROP_SIZE - 4 * SPOT_RADIUS)
    xs = sorted(set(list(range(0, max(0, W_pad - CROP_SIZE) + 1, stride)) + [max(0, W_pad - CROP_SIZE)]))
    ys = sorted(set(list(range(0, max(0, H_pad - CROP_SIZE) + 1, stride)) + [max(0, H_pad - CROP_SIZE)]))

    covered = np.zeros(len(coords), dtype=bool)
    pred_accum = np.zeros_like(genes)

    model.eval()
    for cy in ys:
        for cx in xs:
            if covered.all(): break
            rel_x, rel_y = coords[:, 0] - cx, coords[:, 1] - cy
            keep = (~covered) & (rel_x >= SPOT_RADIUS) & (rel_x <= CROP_SIZE - SPOT_RADIUS) & (rel_y >= SPOT_RADIUS) & (rel_y <= CROP_SIZE - SPOT_RADIUS)
            if not keep.any(): continue
            crop = img_tensor[:, cy:cy + CROP_SIZE, cx:cx + CROP_SIZE].unsqueeze(0).to(device)
            idxs = np.where(keep)[0]
            rel_coords_t = torch.tensor(np.stack([rel_x[idxs], rel_y[idxs]], axis=1), dtype=torch.float32, device=device)
            radii_t = torch.full((len(idxs),), float(SPOT_RADIUS), device=device)
            
            preds_list = []
            for flip_h, flip_w in [(False, False), (True, False), (False, True)]:
                crop_i = crop
                if flip_h: crop_i = torch.flip(crop_i, dims=[3])
                if flip_w: crop_i = torch.flip(crop_i, dims=[2])
                with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                    dm = model(crop_i)
                if flip_h: dm = torch.flip(dm, dims=[3])
                if flip_w: dm = torch.flip(dm, dims=[2])
                preds_list.append(aggregate_sparse_spots(dm, rel_coords_t, radii_t, CROP_SIZE))
            pred_accum[idxs] = torch.stack(preds_list).mean(dim=0).float().cpu().numpy()
            covered[idxs] = True

    return pred_accum[covered], genes[covered], int(covered.sum()), len(coords)

model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
all_p, all_t = [], []
for sid in test_ids:
    p, t, n_cov, n_tot = evaluate_full_slide(model, sid)
    print(f"  {sid}: {n_cov}/{n_tot} spots predicted")
    if n_cov > 0:
        all_p.append(p)
        all_t.append(t)

if all_p:
    preds_full, trues_full = np.concatenate(all_p, axis=0), np.concatenate(all_t, axis=0)
    m_final = compute_regression_metrics(preds_full, trues_full)
    print(f"\n=== KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC (N={preds_full.shape[0]} spots, "
          f"{m_final['n_genes_valid']}/{preds_full.shape[1]} gene hợp lệ) ===")
    print(f"RMSE  = {m_final['rmse']:.4f}")
    print(f"MAE   = {m_final['mae']:.4f}")
    print(f"PCC@F (Q1)     = {m_final['pcc_f']:.4f}")
    print(f"PCC@S (median) = {m_final['pcc_s']:.4f}")
    print(f"PCC@M (mean)   = {m_final['pcc_m']:.4f}")