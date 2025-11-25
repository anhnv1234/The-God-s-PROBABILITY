import ephem
import pandas as pd
import math
from datetime import datetime, timedelta
from tqdm import tqdm # <--- Hàng mới về đây đại ca

# --- CẤU HÌNH ---
FILE_DAU_VAO = 'abc.txt'
FILE_DAU_RA = 'du_lieu_chiem_tinh_chuan_gio.parquet'

# --- 1. HÀM TÍNH TOÁN CHIÊM TINH (ĐÃ FIX) ---
def get_astro_features(date_obj):
    # Bước 1: Thiết lập giờ 18:00 Hà Nội
    local_time = date_obj.replace(hour=18, minute=0, second=0)
    
    # Bước 2: Đổi sang UTC (18h - 7h = 11h UTC)
    utc_time = local_time - timedelta(hours=7)
    
    observer = ephem.Observer()
    observer.lat = '21.0285'  # Hà Nội
    observer.lon = '105.8542'
    observer.date = utc_time
    
    stars = {
        "Sun": ephem.Sun(),
        "Moon": ephem.Moon(),
        "Jupiter": ephem.Jupiter(),
        "Venus": ephem.Venus(),
        "Uranus": ephem.Uranus()
    }
    
    features = {}
    
    # Thứ trong tuần
    features["Weekday"] = date_obj.weekday() / 6.0
    
    # Pha Mặt Trăng
    moon = ephem.Moon()
    moon.compute(observer)
    features["Moon_Phase"] = moon.phase / 100.0
    
    # Tính toán từng hành tinh
    for name, body in stars.items():
        body.compute(observer)
        
        # --- FIX LỖI Ở ĐÂY ---
        # Thay vì gọi body.ecl_lon (dễ lỗi), ta ép kiểu sang hệ Ecliptic
        ecl = ephem.Ecliptic(body)
        lon_rad = ecl.lon # Đây là kinh độ hoàng đạo (radian)
        
        # Tính Sin/Cos
        features[f"{name}_Sin"] = math.sin(lon_rad)
        features[f"{name}_Cos"] = math.cos(lon_rad)
        
        # Tính Độ (0-360) để đại ca soi
        features[f"{name}_Deg"] = math.degrees(lon_rad)
        
        # Khoảng cách (AU)
        features[f"{name}_Dist"] = body.earth_distance

    return features

# --- 2. XỬ LÝ FILE (CÓ TQDM) ---
def process_lottery_data(input_file, output_file):
    data_rows = []
    print(f"🚀 Đang khởi động... Đọc file: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{input_file}'!")
        return

    # Lọc bỏ dòng trống trước để thanh progress bar hiện đúng số lượng
    valid_lines = [line.strip() for line in lines if line.strip()]
    
    print("🔮 Đang luận giải thiên cơ (Chạy thanh tiến trình)...")
    
    # --- THÊM TQDM VÀO VÒNG LẶP ---
    # desc="Processing": Tiêu đề thanh
    # unit="day": Đơn vị đếm
    for line in tqdm(valid_lines, desc="Processing", unit="day"):
        
        parts = [p.strip() for p in line.split(',')]
        
        # 1. Xử lý ngày
        try:
            date_str = parts[0]
            date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            continue # Bỏ qua dòng lỗi ngày

        # 2. Xử lý số
        raw_numbers = [x for x in parts[1:] if x]
        clean_numbers = []
        for n in raw_numbers:
            try:
                clean_numbers.append(int(n))
            except ValueError:
                clean_numbers.append(None)

        final_numbers = clean_numbers[:27] + [None] * (27 - len(clean_numbers))

        # 3. Tính Chiêm tinh
        astro_data = get_astro_features(date_obj)

        # 4. Đóng gói
        row = {
            "Date": date_obj,
            **astro_data
        }
        
        for i in range(27):
            row[f"Res_{i+1:02d}"] = final_numbers[i]

        data_rows.append(row)

    # --- PHẦN CÒN LẠI GIỮ NGUYÊN ---
    df = pd.DataFrame(data_rows)
    
    if df.empty:
        print("❌ Không có dữ liệu nào hợp lệ.")
        return

    df = df.sort_values(by="Date")

    # Check ngắt quãng
    print("\n--- 🔍 KẾT QUẢ SOI CẦU NGÀY THÁNG ---")
    df['date_diff'] = df['Date'].diff()
    gaps = df[df['date_diff'] > timedelta(days=1)]
    
    if not gaps.empty:
        print(f"😱 CẢNH BÁO: Dữ liệu bị thủng {len(gaps)} lỗ!")
        # In tối đa 5 lỗ hổng đầu tiên thôi cho đỡ dài dòng
        print("Một số khoảng mất tích tiêu biểu:")
        count = 0
        for index, row in gaps.iterrows():
            count += 1
            prev = row['Date'] - row['date_diff']
            curr = row['Date']
            days = row['date_diff'].days - 1
            print(f"   👉 Mất {days} ngày: {prev.strftime('%d/%m/%Y')} -> {curr.strftime('%d/%m/%Y')}")
            if count >= 5:
                print("   ... (và còn nữa) ...")
                break
    else:
        print("✅ TUYỆT VỜI: Dữ liệu liền mạch!")

    df.drop(columns=['date_diff'], inplace=True)

    # Xuất file
    try:
        df.to_parquet(output_file, index=False)
        print(f"\n💾 Đã lưu file Parquet: {output_file}")
        print(f"📊 Tổng số dòng: {len(df)}")
    except Exception as e:
        print(f"❌ Lỗi lưu file: {e}")

# --- CHẠY ---
if __name__ == "__main__":
    process_lottery_data(FILE_DAU_VAO, FILE_DAU_RA)