import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ephem  # Thư viện tính sao xịn sò
import datetime
import os
import schedule
import time
import requests
from bs4 import BeautifulSoup
from datetime import timedelta

# =============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG)
# =============================================================================
MODEL_PATH = 'model_lb90_seed44_BEST.pth'  # Đường dẫn model xịn nhất của đại ca
DATA_FILE = 'du_lieu_chiem_tinh_chuan_gio.parquet'
LOOKBACK_DAYS = 90
NUM_CLASSES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tọa độ Hà Nội (để tính sao cho chuẩn giờ quay)
LATITUDE = '21.0285'
LONGITUDE = '105.8542'

print(f"🔮 KHỞI ĐỘNG BOT THẦN CƠ DIỆU TOÁN TRÊN: {DEVICE}")

# =============================================================================
# 2. KIẾN TRÚC MODEL (PHẢI GIỐNG HỆT FILE TRAIN)
# =============================================================================
# Copy nguyên xi từ file train của đại ca để load được trọng số

class GatedResidualNetwork(nn.Module):
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
        residual = self.project(x) if self.project is not None else x
        x_val = self.fc1(x)
        x_val = self.elu(x_val)
        x_val = self.fc2(x_val)
        x_val = self.dropout(x_val)
        gate_val = torch.sigmoid(self.gate(x))
        out = (x_val * gate_val) + residual
        return self.norm(out)

class UltimateAstroModel(nn.Module):
    def __init__(self, num_classes, lookback_days, astro_features, d_model=64, nhead=4):
        super(UltimateAstroModel, self).__init__()
        self.hist_proj = nn.Linear(num_classes, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, lookback_days, d_model))
        self.hist_gate = GatedResidualNetwork(d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.hist_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.astro_gate = GatedResidualNetwork(astro_features, d_model)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        self.fusion_gate = GatedResidualNetwork(d_model * 2, 128)
        self.final_out = nn.Linear(128, num_classes)

    def forward(self, x_hist, x_astro):
        h = self.hist_proj(x_hist) + self.pos_embedding
        h = self.hist_gate(h)
        h = self.hist_encoder(h)
        a = self.astro_gate(x_astro)
        a_query = a.unsqueeze(1)
        attn_out, _ = self.cross_attention(query=a_query, key=h, value=h)
        attn_out = self.norm_cross(attn_out + a_query)
        a_squeezed = a_query.squeeze(1)
        attn_squeezed = attn_out.squeeze(1)
        combined = torch.cat((a_squeezed, attn_squeezed), dim=1)
        return self.final_out(self.fusion_gate(combined))

# =============================================================================
# 3. CÁC HÀM XỬ LÝ DỮ LIỆU & CÀO KẾT QUẢ (SCRAPING)
# =============================================================================

def get_astro_features(date_str, columns_template):
    """
    Tính vị trí các sao vào 18:30 của ngày dự báo.
    Dùng thư viện Ephem để tính toán chuẩn xác.
    """
    obs = ephem.Observer()
    obs.lat = LATITUDE
    obs.lon = LONGITUDE
    # Giờ quay số là 18:30
    obs.date = f"{date_str} 11:30:00" # Ephem dùng giờ UTC, VN là UTC+7 nên trừ đi 7

    # Danh sách các hành tinh cơ bản (Đại ca có thể thêm nếu file cũ có nhiều hơn)
    planets = {
        'Sun': ephem.Sun(obs),
        'Moon': ephem.Moon(obs),
        'Mercury': ephem.Mercury(obs),
        'Venus': ephem.Venus(obs),
        'Mars': ephem.Mars(obs),
        'Jupiter': ephem.Jupiter(obs),
        'Saturn': ephem.Saturn(obs),
        'Uranus': ephem.Uranus(obs),
        'Neptune': ephem.Neptune(obs)
    }

    # Tạo dictionary dữ liệu
    data_dict = {}
    for name, body in planets.items():
        # Lấy kinh độ hoàng đạo (Ecliptic Longitude) - Quy ra độ (0-360)
        lon = np.degrees(body.hlon)
        data_dict[name] = lon

    # Mapping vào vector đúng theo thứ tự cột của file Parquet cũ
    # Lưu ý: Hàm này giả định tên cột trong file Parquet là tên hành tinh (ví dụ 'Sun', 'Moon'...)
    # Nếu file đại ca dùng tên khác (ví dụ 'Sun_Deg'), code sẽ tự khớp nếu tên chứa từ khóa.
    
    feature_vector = []
    for col in columns_template:
        val = 0.0
        found = False
        for p_name, p_val in data_dict.items():
            if p_name in col: # So sánh khớp tên
                val = p_val
                found = True
                break
        feature_vector.append(val)
    
    return np.array(feature_vector, dtype='float32')

def fetch_xskt_results(date_str):
    """
    Hàm cào dữ liệu xổ số miền Bắc cho ngày date_str (format YYYY-MM-DD).
    Trả về list các số đã về.
    """
    # Convert YYYY-MM-DD -> DD-MM-YYYY để request web
    d = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    fmt_date = d.strftime('%d-%m-%Y')
    
    url = f"https://xoso.com.vn/xsmb-{fmt_date}.html"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Logic lấy giải ĐB -> Giải 7 (Tùy trang web, đây là logic phổ thông)
        # Tìm tất cả thẻ span có class là số kết quả (thường class chứa 'v-g')
        # Đây là ví dụ logic cào, đại ca cần check lại cấu trúc trang nếu web đổi source
        numbers = []
        
        # Cách lấy đơn giản: tìm tất cả các thẻ chứa số (thường nằm trong table)
        tables = soup.find_all('table', class_='table-result')
        if not tables:
            return None
            
        for table in tables:
            spans = table.find_all('span')
            for s in spans:
                txt = s.get_text().strip()
                if txt.isdigit():
                    # Lấy 2 số cuối
                    tail = txt[-2:]
                    numbers.append(int(tail))
        
        return list(set(numbers)) # Trả về danh sách các số unique (0-99)
    except Exception as e:
        print(f"⚠️ Lỗi cào dữ liệu ngày {date_str}: {e}")
        return None

def update_dataset():
    """
    Kiểm tra file Parquet, nếu thiếu ngày thì tự động cào thêm và update file.
    """
    print("🔄 Đang kiểm tra và cập nhật dữ liệu...")
    if not os.path.exists(DATA_FILE):
        print(f"❌ Không tìm thấy file {DATA_FILE}. Không thể chạy.")
        return None, None

    df = pd.read_parquet(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    last_date = df['Date'].max().date()
    today = datetime.date.today()
    
    # Xác định các cột chiêm tinh (để điền dữ liệu mới)
    res_cols = [c for c in df.columns if c.startswith('Res_')]
    astro_cols = [c for c in df.columns if c not in res_cols and c != 'Date' and not c.endswith('_Deg')]
    
    if last_date >= today - timedelta(days=1):
        print("✅ Dữ liệu đã cập nhật đến hôm qua. Sẵn sàng dự báo!")
        return df, astro_cols

    # Nếu thiếu dữ liệu, chạy vòng lặp update
    current_date = last_date + timedelta(days=1)
    new_rows = []
    
    while current_date < today: # Chỉ update đến hôm qua (vì hôm nay chưa xổ)
        str_date = current_date.strftime('%Y-%m-%d')
        print(f"   >> Đang cào dữ liệu ngày {str_date}...")
        
        results = fetch_xskt_results(str_date)
        if results:
            # Tạo row mới
            row = {'Date': pd.Timestamp(current_date)}
            # Điền Astro
            astro_vals = get_astro_features(str_date, astro_cols)
            for i, col in enumerate(astro_cols):
                row[col] = astro_vals[i]
            
            # Điền Kết quả (Res_0 ... Res_XX) - Lưu ý file cũ đại ca lưu kiểu gì
            # Giả sử file cũ lưu danh sách các số về. Ở đây em fill NaN trước
            for col in res_cols:
                row[col] = np.nan
            
            # Fill kết quả thực tế
            for i, num in enumerate(results):
                if i < len(res_cols):
                    row[f'Res_{i}'] = float(num)
            
            new_rows.append(row)
            print(f"      -> Đã thêm {len(results)} số.")
        else:
            print(f"      -> Không có dữ liệu hoặc web lỗi.")
        
        current_date += timedelta(days=1)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        # Lưu đè file cũ
        df.to_parquet(DATA_FILE)
        print(f"💾 Đã cập nhật thêm {len(new_rows)} ngày vào Data.")
    
    return df, astro_cols

# =============================================================================
# 4. HÀM DỰ BÁO (INFERENCE)
# =============================================================================

def predict_today():
    print(f"\n{'='*60}")
    print(f"🎲 BẮT ĐẦU PHIÊN DỰ BÁO NGÀY {datetime.date.today()}")
    print(f"{'='*60}")
    
    # 1. Update và Load dữ liệu
    df, astro_cols = update_dataset()
    if df is None: return

    # 2. Chuẩn bị Input Lịch sử (Last 90 days)
    # Cần lấy 90 ngày có dữ liệu xổ số gần nhất
    # Logic: Lọc những ngày có kết quả không null
    res_cols = [c for c in df.columns if c.startswith('Res_')]
    
    # Chuyển đổi dữ liệu sang vector 100 chiều
    # Chỉ lấy những dòng có dữ liệu xổ số để làm History
    valid_rows = df.dropna(subset=[res_cols[0]]) # Giả sử cột Res_0 phải có
    
    if len(valid_rows) < LOOKBACK_DAYS:
        print("❌ Không đủ dữ liệu lịch sử (cần ít nhất 90 ngày).")
        return

    # Lấy 90 ngày cuối cùng
    last_90_days = valid_rows.iloc[-LOOKBACK_DAYS:]
    raw_results = last_90_days[res_cols].values
    
    x_hist = np.zeros((1, LOOKBACK_DAYS, NUM_CLASSES), dtype='float32')
    
    for t in range(LOOKBACK_DAYS):
        day_res = raw_results[t]
        for num in day_res:
            if pd.notna(num):
                idx = int(num)
                if 0 <= idx <= 99:
                    x_hist[0, t, idx] = 1.0
    
    # 3. Chuẩn bị Input Chiêm tinh (HÔM NAY - TƯƠNG LAI)
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    astro_vals = get_astro_features(today_str, astro_cols)
    x_astro = torch.tensor(astro_vals, dtype=torch.float32).unsqueeze(0) # (1, num_features)
    x_hist = torch.tensor(x_hist, dtype=torch.float32)

    # 4. Load Model
    # Cần xác định num_features từ file dữ liệu để init model
    num_astro_features = len(astro_cols)
    
    model = UltimateAstroModel(NUM_CLASSES, LOOKBACK_DAYS, num_astro_features).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        # Lưu ý: Cần thêm weights_only=False nếu torch báo lỗi
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except:
             checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
             
        # Checkpoint của đại ca lưu cả optimizer, nên phải trọc vào lấy state_dict của model thôi
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) # Trường hợp file chỉ lưu weight
            
        model.eval()
        print("🤖 Model đã load thành công! Đang soi cầu...")
    else:
        print(f"❌ Không tìm thấy model {MODEL_PATH}")
        return

    # 5. Predict
    with torch.no_grad():
        x_hist = x_hist.to(DEVICE)
        x_astro = x_astro.to(DEVICE)
        
        output = model(x_hist, x_astro)
        # Output là Logits, qua Sigmoid để ra xác suất
        probs = torch.sigmoid(output).squeeze(0) # (100,)
        
    # 6. Lấy Top 5
    top5_prob, top5_idx = torch.topk(probs, 5)
    
    top5_numbers = top5_idx.cpu().numpy()
    top5_percent = top5_prob.cpu().numpy() * 100
    
    print("\n" + "*"*40)
    print(f"🌟 KẾT QUẢ DỰ BÁO NGÀY {today_str} 🌟")
    print("*"*40)
    for i in range(5):
        print(f"   🏆 TOP {i+1}: Số {top5_numbers[i]:02d} (Tỉ lệ: {top5_percent[i]:.2f}%)")
    print("*"*40)
    print("👉 Đại ca tham khảo, chúc đại ca may mắn!\n")

# =============================================================================
# 5. MAIN LOOP (CHẠY HẰNG NGÀY)
# =============================================================================

def job():
    print(f"\n⏰ Đã đến giờ G (Time check: {datetime.datetime.now()})")
    predict_today()

if __name__ == "__main__":
    # Test chạy ngay lần đầu tiên khi mở tool
    predict_today()
    
    # Hẹn giờ chạy hằng ngày vào lúc 16:30 (4h30 chiều)
    # Để có thời gian nghiên cứu trước giờ quay
    schedule.every().day.at("16:30").do(job)
    
    print("💤 Bot đang chuyển sang chế độ ngủ đông. Chờ đến 16:30 hằng ngày sẽ tự dậy làm việc...")
    
    while True:
        schedule.run_pending()
        time.sleep(60) # Ngủ 1 phút check 1 lần