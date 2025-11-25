import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random
import gc
from tqdm import tqdm

# =============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG)
# =============================================================================
FILE_DU_LIEU = 'du_lieu_chiem_tinh_chuan_gio.parquet'

# --- CẤU HÌNH TAM HỢP (3 KHUNG THỜI GIAN) ---
# Đại ca chạy 3 khung này để bắt đủ loại cầu (ngắn, trung, dài)
LIST_LOOKBACK = [7, 30, 90]

# Mỗi khung chạy 3 model (Seed) để loại bỏ rủi ro
NUM_SEEDS_PER_LB = 3

# Tổng số epoch (vòng lặp) train cho mỗi model
EPOCHS = 1000

# Bắt đầu chế độ SWA (Gom bi) từ epoch này
SWA_START_EPOCH = 800

# Các thông số cố định khác
NUM_CLASSES = 100            # 100 số (00-99)
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⚙️ KHỞI ĐỘNG CHIẾN DỊCH TAM HỢP (FULL EXPANDED) TRÊN: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")


# =============================================================================
# 2. CÁC LỚP HỖ TRỢ (LOSS & LAYERS)
# =============================================================================

class FocalLossWithSmoothing(nn.Module):
    """
    Hàm Loss kết hợp Focal Loss và Label Smoothing.
    - Focal Loss: Phạt nặng model nếu đoán sai số trúng (số hiếm).
    - Smoothing: Giúp model không bị quá tự tin (Overconfidence).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', smoothing=0.05):
        super(FocalLossWithSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # Làm mềm nhãn (0 -> 0.05, 1 -> 0.95)
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Tính BCE Loss cơ bản
        bce_loss = self.bce(inputs, targets_smooth)
        
        # Tính xác suất (pt)
        pt = torch.exp(-bce_loss)
        
        # Áp dụng công thức Focal Loss
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GatedResidualNetwork(nn.Module):
    """
    Mạng Gated Residual (GRN) - Công nghệ lọc nhiễu.
    """
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        
        if input_size != hidden_size:
            self.project = nn.Linear(input_size, hidden_size)
        else:
            self.project = None

    def forward(self, x):
        # Nhánh tắt (Residual)
        residual = self.project(x) if self.project is not None else x
        
        # Nhánh chính
        x_val = self.fc1(x)
        x_val = self.elu(x_val)
        x_val = self.fc2(x_val)
        x_val = self.dropout(x_val)
        
        # Cổng lọc (Gate)
        gate_val = torch.sigmoid(self.gate(x))
        
        # Kết hợp
        out = (x_val * gate_val) + residual
        return self.norm(out)


# =============================================================================
# 3. KIẾN TRÚC MÔ HÌNH (ULTIMATE ASTRO MODEL)
# =============================================================================

class UltimateAstroModel(nn.Module):
    def __init__(self, num_classes, lookback_days, astro_features, d_model=64, nhead=4):
        super(UltimateAstroModel, self).__init__()
        
        # --- A. NHÁNH LỊCH SỬ ---
        self.hist_proj = nn.Linear(num_classes, d_model)
        # Positional Embedding (Học vị trí thời gian)
        self.pos_embedding = nn.Parameter(torch.randn(1, lookback_days, d_model))
        # Cổng lọc nhiễu lịch sử
        self.hist_gate = GatedResidualNetwork(d_model, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True, 
            dropout=0.1
        )
        self.hist_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # --- B. NHÁNH CHIÊM TINH ---
        # Cổng lọc nhiễu sao
        self.astro_gate = GatedResidualNetwork(astro_features, d_model)
        
        # --- C. CROSS ATTENTION ---
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model)
        
        # --- D. FUSION ---
        self.fusion_gate = GatedResidualNetwork(d_model * 2, 128)
        self.final_out = nn.Linear(128, num_classes)

    def forward(self, x_hist, x_astro):
        # 1. Xử lý Lịch sử
        h = self.hist_proj(x_hist) + self.pos_embedding
        h = self.hist_gate(h)
        h = self.hist_encoder(h) # Output: (Batch, Lookback, d_model)
        
        # 2. Xử lý Chiêm tinh
        a = self.astro_gate(x_astro)
        a_query = a.unsqueeze(1) # Output: (Batch, 1, d_model)
        
        # 3. Cross Attention (Dùng Sao soi Lịch sử)
        attn_out, _ = self.cross_attention(query=a_query, key=h, value=h)
        attn_out = self.norm_cross(attn_out + a_query)
        
        # 4. Hợp nhất
        a_squeezed = a_query.squeeze(1)
        attn_squeezed = attn_out.squeeze(1)
        combined = torch.cat((a_squeezed, attn_squeezed), dim=1)
        
        # 5. Output
        return self.final_out(self.fusion_gate(combined))


# =============================================================================
# 4. DATASET VÀ HÀM LOAD DỮ LIỆU
# =============================================================================

class LotteryDataset(Dataset):
    def __init__(self, x_hist, x_astro, y):
        self.x_hist = torch.tensor(x_hist, dtype=torch.float32)
        self.x_astro = torch.tensor(x_astro, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_hist[idx], self.x_astro[idx], self.y[idx]


def load_data_dynamic(lookback_days):
    """
    Hàm load dữ liệu động dựa theo tham số lookback_days.
    Tự động xử lý ngày rỗng và chuyển đổi vector.
    """
    if not os.path.exists(FILE_DU_LIEU):
        print(f"❌ Lỗi: Không tìm thấy file {FILE_DU_LIEU}")
        return None
    
    print(f"📂 Đang xử lý dữ liệu với Lookback = {lookback_days} ngày...")
    df = pd.read_parquet(FILE_DU_LIEU)
    
    # Xác định cột
    res_cols = [c for c in df.columns if c.startswith('Res_')]
    astro_cols = [c for c in df.columns if c not in res_cols and c != 'Date' and not c.endswith('_Deg')]
    
    total_days = len(df)
    daily_vectors = np.zeros((total_days, NUM_CLASSES), dtype='float32')
    raw_results = df[res_cols].values 
    
    # Chuyển đổi dữ liệu thô sang Vector 100 chiều
    for i in range(total_days):
        for num in raw_results[i]:
            if pd.notna(num):
                try:
                    idx = int(num)
                    if 0 <= idx <= 99:
                        daily_vectors[i, idx] = 1.0
                except:
                    pass
    
    astro_data = df[astro_cols].values.astype('float32')
    
    X_h, X_a, Y = [], [], []
    
    # Cắt cửa sổ trượt (Sliding Window)
    for i in range(lookback_days, len(daily_vectors)):
        target = daily_vectors[i]
        
        # BỎ QUA NGÀY NGHỈ (Nếu target toàn số 0)
        if np.sum(target) == 0:
            continue
            
        # Lấy lịch sử (bao gồm cả ngày nghỉ để giữ mạch thời gian)
        X_h.append(daily_vectors[i-lookback_days : i])
        X_a.append(astro_data[i])
        Y.append(target)
        
    return np.array(X_h), np.array(X_a), np.array(Y), len(astro_cols)


def seed_everything(seed=42):
    """Thiết lập hạt giống ngẫu nhiên để kết quả tái lập được"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# =============================================================================
# 5. CƠ CHẾ CHECKPOINT (LƯU GAME & CHƠI TIẾP)
# =============================================================================
# Đây là phần đại ca cần kiểm tra kỹ, em đã viết rất rõ ràng

def save_checkpoint(model, optimizer, scheduler, swa_model, swa_scheduler, scaler, epoch, loss, filename):
    print(f"   💾 [Checkpoint] Đang lưu trạng thái tại Epoch {epoch+1}...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'swa_model_state_dict': swa_model.state_dict() if swa_model else None,
        'swa_scheduler_state_dict': swa_scheduler.state_dict() if swa_scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }, filename)


def load_checkpoint(model, optimizer, scheduler, swa_model, swa_scheduler, scaler, filename):
    if os.path.isfile(filename):
        print(f"♻️  PHÁT HIỆN CHECKPOINT '{filename}'. ĐANG KHÔI PHỤC...")
        
        # --- SỬA LỖI TẠI ĐÂY (Thêm weights_only=False) ---
        try:
            checkpoint = torch.load(filename, weights_only=False)
        except Exception as e:
            # Dự phòng cho các phiên bản PyTorch cũ hơn không có tham số này
            checkpoint = torch.load(filename)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if swa_model and checkpoint.get('swa_model_state_dict'):
            swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
        
        if swa_scheduler and checkpoint.get('swa_scheduler_state_dict'):
            swa_scheduler.load_state_dict(checkpoint['swa_scheduler_state_dict'])
            
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"✅ KHÔI PHỤC THÀNH CÔNG! TIẾP TỤC TRAIN TỪ EPOCH {start_epoch}")
        return start_epoch, loss
    else:
        print("🆕 Không có checkpoint cũ. Bắt đầu train mới.")
        return 0, None

# =============================================================================
# 6. HÀM TRAIN MỘT MODEL CỤ THỂ
# =============================================================================

def train_model(lookback, seed_idx, X_h, X_a, Y, num_astro_features):
    # Thiết lập Seed riêng biệt
    seed = 42 + seed_idx
    seed_everything(seed)
    
    # Định danh các file (QUAN TRỌNG ĐỂ KHÔNG BỊ GHI ĐÈ)
    base_name = f"lb{lookback}_seed{seed}"
    model_name_swa = f"model_{base_name}_SWA.pth"   # File kết quả cuối cùng
    model_name_best = f"model_{base_name}_BEST.pth" # File kết quả tốt nhất giữa chừng
    checkpoint_name = f"checkpoint_{base_name}.pth" # File lưu tạm
    
    print(f"\n{'='*60}")
    print(f"⚡ BẮT ĐẦU TRAIN: Lookback={lookback} | Seed={seed}")
    print(f"{'='*60}")
    
    # Chia dữ liệu 9/1
    X_h_train, X_h_test, X_a_train, X_a_test, Y_train, Y_test = train_test_split(
        X_h, X_a, Y, test_size=0.1, shuffle=False
    )
    
    train_loader = DataLoader(LotteryDataset(X_h_train, X_a_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(LotteryDataset(X_h_test, X_a_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # Khởi tạo Model & Optimizer
    model = UltimateAstroModel(NUM_CLASSES, lookback, num_astro_features).to(DEVICE)
    swa_model = AveragedModel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-5)
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    
    criterion = FocalLossWithSmoothing(smoothing=0.05)
    scaler = GradScaler() if DEVICE.type == 'cuda' else None
    
    # --- KHÔI PHỤC CHECKPOINT NẾU CÓ ---
    start_epoch, _ = load_checkpoint(model, optimizer, scheduler, swa_model, swa_scheduler, scaler, checkpoint_name)
    
    best_val_loss = float('inf')

    # Nếu model này đã train xong trước đó rồi thì bỏ qua
    if start_epoch >= EPOCHS:
        print(f"⏩ Model {base_name} đã hoàn thành trước đó. Bỏ qua.")
        return model_name_swa

    # --- VÒNG LẶP TRAIN ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Chỉ hiển thị progress bar cho 5 epoch đầu và các epoch chia hết cho 10
        show_progress = (epoch < 5) or ((epoch+1) % 10 == 0)
        
        if show_progress:
            loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)
        else:
            loop = train_loader

        for x_h, x_a, y in loop:
            x_h, x_a, y = x_h.to(DEVICE), x_a.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            # Mixed Precision Training
            if scaler:
                with autocast():
                    outputs = model(x_h, x_a)
                    loss = criterion(outputs, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x_h, x_a)
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            running_loss += loss.item()
            if show_progress:
                loop.set_postfix(loss=loss.item())

        # Cập nhật SWA hoặc Scheduler thường
        if epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
            
        # --- VALIDATION & SAVE ---
        # Kiểm tra mỗi 5 epoch
        if (epoch+1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_h, x_a, y in test_loader:
                    x_h, x_a, y = x_h.to(DEVICE), x_a.to(DEVICE), y.to(DEVICE)
                    out = model(x_h, x_a)
                    val_loss += criterion(out, y).item()
            
            avg_val_loss = val_loss / len(test_loader)
            avg_train_loss = running_loss / len(train_loader)
            
            msg = "SWA" if epoch >= SWA_START_EPOCH else "Normal"
            print(f"   [{msg}] Ep {epoch+1}: TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f}")
            
            # 1. Lưu Best Model (Model thường có Loss thấp nhất)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_name_best)
                print(f"      🏆 New Best Model Saved (Loss: {best_val_loss:.4f})")
            
            # 2. Lưu Checkpoint (Để lỡ mất điện thì chạy lại)
            save_checkpoint(model, optimizer, scheduler, swa_model, swa_scheduler, scaler, epoch, avg_train_loss, checkpoint_name)

    # --- KẾT THÚC VÒNG LẶP ---
    print(f"💾 Đang lưu SWA Model hoàn chỉnh: {model_name_swa}...")
    update_bn(train_loader, swa_model, device=DEVICE)
    torch.save(swa_model.state_dict(), model_name_swa)
    
    # Xóa file checkpoint tạm đi cho sạch ổ cứng
    if os.path.exists(checkpoint_name):
        os.remove(checkpoint_name)
    
    # Dọn dẹp bộ nhớ GPU
    del model, swa_model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()
    
    return model_name_swa


# =============================================================================
# 7. MAIN PROGRAM
# =============================================================================
def main():
    saved_files = []
    
    # Vòng lặp 1: Duyệt qua từng khung thời gian (7, 30, 90)
    for lb in LIST_LOOKBACK:
        print(f"\n\n{'#'*60}")
        print(f"🌐 CHUYỂN SANG KHUNG THỜI GIAN: LOOKBACK = {lb} NGÀY")
        print(f"{'#'*60}")
        
        # Load lại dữ liệu theo lookback mới
        data = load_data_dynamic(lb)
        if data is None: continue
        X_h, X_a, Y, num_features = data
        
        # Vòng lặp 2: Chạy nhiều Seeds để Ensemble
        for s in range(NUM_SEEDS_PER_LB):
            fname = train_model(lb, s, X_h, X_a, Y, num_features)
            saved_files.append(fname)
            
    print("\n" + "="*60)
    print("🎉 SỨ MỆNH HOÀN THÀNH! ĐẠI CA ĐÃ CÓ ĐỦ BỘ SƯU TẬP:")
    for f in saved_files:
        print(f"   ✅ {f}")
    print("="*60)
    print("👉 Đại ca giữ kỹ các file này để dùng cho code Dự Báo nhé!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng chương trình! Checkpoint an toàn.")