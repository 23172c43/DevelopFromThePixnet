"""prepare_pixnet_data.py -- Tai du lieu tho HER2ST tu Zenodo va tien xu ly (chon
top 250 gene tren Train+Val, cat anh + chuan hoa log1p(raw count)) cho PixNet. Chay
1 LAN truoc khi train (run_pixnet_train.py doc PROCESSED_DIR/TOP_GENES_FILE nay).
Chuyen NGUYEN VEN tu pixnet.ipynb (Muc 1: TAI VA GIAI NEN DU LIEU THO, Muc 2: TIEN XU
LY), khong doi logic -- chi tach thanh module rieng va them import can thiet.
"""
import os
import glob
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


WORK_DIR = '/kaggle/working'
RAW_DIR = os.path.join(WORK_DIR, 'her2st_raw_data')
PROCESSED_DIR = os.path.join(WORK_DIR, 'her2st_processed_v2')

# ==========================================
# 1. TẢI VÀ GIẢI NÉN DỮ LIỆU THÔ TỪ ZENODO
# ==========================================
print("\n--- BƯỚC 1: KIỂM TRA VÀ TẢI DỮ LIỆU THÔ ---")
if shutil.which("7z") is None:
    print("Đang cài đặt p7zip-full...")
    os.system("apt-get update -qq && apt-get install -y p7zip-full -qq")

if not os.path.isdir(os.path.join(RAW_DIR, 'images', 'HE')):
    print("Đang tải dữ liệu HER2ST từ Zenodo...")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.system(f'wget -q -O /tmp/count-matrices.zip "https://zenodo.org/records/3957257/files/count-matrices.zip?download=1"')
    os.system(f'wget -q -O /tmp/images.zip "https://zenodo.org/records/3957257/files/images.zip?download=1"')
    os.system(f'wget -q -O /tmp/meta.zip "https://zenodo.org/records/3957257/files/meta.zip?download=1"')
    os.system(f'wget -q -O /tmp/spot-selections.zip "https://zenodo.org/records/3957257/files/spot-selections.zip?download=1"')

    print("Đang giải nén dữ liệu...")
    os.system(f'7z x /tmp/count-matrices.zip -p"zNLXkYk3Q9znUseS" -o{RAW_DIR} -y > /dev/null 2>&1')
    os.system(f'7z x /tmp/images.zip -p"zNLXkYk3Q9znUseS" -o{RAW_DIR} -y > /dev/null 2>&1')
    os.system(f'7z x /tmp/spot-selections.zip -p"yUx44SzG6NdB32gY" -o{RAW_DIR} -y > /dev/null 2>&1')
    os.system(f'7z x /tmp/meta.zip -p"yUx44SzG6NdB32gY" -o{RAW_DIR} -y > /dev/null 2>&1')
    
    # Xóa file zip rác để giải phóng dung lượng Kaggle
    os.system('rm /tmp/*.zip')
    print("✅ Tải và giải nén thành công.")
else:
    print("✅ Dữ liệu thô đã tồn tại, bỏ qua bước tải.")

# ==========================================\n# 2. TIỀN XỬ LÝ (TÌM TOP 250 GENES & CHUẨN HÓA)\n# ==========================================\n
print("\n--- BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU ---")
Image.MAX_IMAGE_PIXELS = None
TOP_GENES_FILE = os.path.join(PROCESSED_DIR, 'top_250_genes.txt')

# [FIX RÒ RỈ] Định nghĩa phân chia bệnh nhân NGAY TẠI ĐÂY trước khi chọn Gene Panel
train_patients = ['A', 'B', 'C', 'D', 'E', 'F']
val_patients = ['G']
test_patients = ['H']

# Gene Panel chỉ được chọn dựa trên dữ liệu Train + Val (tuyệt đối không chứa Test 'H')
allowed_panel_patients = set(train_patients + val_patients)

if not os.path.exists(TOP_GENES_FILE):
    os.makedirs(os.path.join(PROCESSED_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, 'labels'), exist_ok=True)

    print("Đang quét 250 gen chung tốt nhất (Chỉ trên tập Train/Val)...")
    
    # Lấy toàn bộ file gốc
    all_matrix_files = glob.glob(os.path.join(RAW_DIR, 'count-matrices/*.tsv.gz')) + \
                       glob.glob(os.path.join(RAW_DIR, 'count-matrices/*.tsv'))
    
    # [FIX RÒ RỈ] Lọc chỉ giữ lại các file thuộc bệnh nhân trong allowed_panel_patients
    matrix_files = [
        f for f in all_matrix_files 
        if os.path.basename(f)[0] in allowed_panel_patients
    ]
    
    print(f"-> Số lượng file matrix được dùng để tính Top 250 Genes: {len(matrix_files)}/{len(all_matrix_files)}")

    global_gene_sum = None
    common_genes = None

    for f in matrix_files:
        df = pd.read_csv(f, sep='\t', index_col=0)
        if common_genes is None:
            common_genes = set(df.columns)
        else:
            common_genes = common_genes.intersection(set(df.columns))
            
        if global_gene_sum is None:
            global_gene_sum = df.sum(axis=0)
        else:
            global_gene_sum = global_gene_sum.add(df.sum(axis=0), fill_value=0)

    filtered_sum = global_gene_sum[list(common_genes)]
    top_250_genes = filtered_sum.nlargest(250).index.tolist()
    
    with open(TOP_GENES_FILE, 'w') as f:
        for gene in top_250_genes:
            f.write(f"{gene}\n")
    print("✅ Đã lưu top_250_genes.txt (Clean)")

    print("Đang cắt ảnh viền và chuẩn hóa log1p(raw count)...")
    def normalize_counts(counts_df):
        return np.log1p(counts_df)

    image_files = glob.glob(os.path.join(RAW_DIR, 'images/HE/*.jpg'))
    sample_ids = sorted([os.path.splitext(os.path.basename(p))[0] for p in image_files])
    pad = 100
    success_count = 0

    for sample_id in tqdm(sample_ids, desc="Processing Samples"):
        try:
            img_path = os.path.join(RAW_DIR, f'images/HE/{sample_id}.jpg')
            coord_path = os.path.join(RAW_DIR, f'{sample_id}_selection.tsv.gz')
            if not os.path.exists(coord_path):
                coord_path = os.path.join(RAW_DIR, f'{sample_id}_selection.tsv')
            count_matrix_path = os.path.join(RAW_DIR, f'count-matrices/{sample_id}.tsv.gz')
            if not os.path.exists(count_matrix_path):
                count_matrix_path = os.path.join(RAW_DIR, f'count-matrices/{sample_id}.tsv')
                
            if not os.path.exists(coord_path) or not os.path.exists(count_matrix_path):
                continue

            img = Image.open(img_path)
            coords_df = pd.read_csv(coord_path, sep='\t')
            counts_df = pd.read_csv(count_matrix_path, sep='\t', index_col=0)
            counts_df = counts_df[top_250_genes]
            counts_df = normalize_counts(counts_df)

            coords_df['barcode'] = coords_df['x'].astype(str) + 'x' + coords_df['y'].astype(str)
            clean_data = coords_df.merge(counts_df, left_on='barcode', right_index=True, how='inner')
            if clean_data.shape[0] == 0:
                continue

            x_min = int(max(0, clean_data['pixel_x'].min() - pad))
            y_min = int(max(0, clean_data['pixel_y'].min() - pad))
            x_max = int(min(img.width, clean_data['pixel_x'].max() + pad))
            y_max = int(min(img.height, clean_data['pixel_y'].max() + pad))
            cropped_img = img.crop((x_min, y_min, x_max, y_max))

            clean_data['cropped_pixel_x'] = clean_data['pixel_x'] - x_min
            clean_data['cropped_pixel_y'] = clean_data['pixel_y'] - y_min

            cropped_img.convert('RGB').save(os.path.join(PROCESSED_DIR, 'images', f'{sample_id}.jpg'), quality=95)
            label_cols = ['cropped_pixel_x', 'cropped_pixel_y', 'barcode'] + top_250_genes
            clean_data[label_cols].to_csv(os.path.join(PROCESSED_DIR, 'labels', f'{sample_id}.csv'), index=False)
            success_count += 1
        except Exception as e:
            print(f"Lỗi ở mẫu {sample_id}: {e}")
    print(f"✅ Đã tiền xử lý {success_count}/{len(sample_ids)} mẫu.")
else:
    print("✅ Dữ liệu đã được tiền xử lý, bỏ qua.")