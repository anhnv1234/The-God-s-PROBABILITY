import pandas as pd
import os

# --- CẤU HÌNH ---
FILE_CAN_DOC = 'du_lieu_chiem_tinh_chuan_gio.parquet'
FILE_XUAT_RA = 'TEST_1000_DONG_SOI_CAU.xlsx'

def export_sample_data():
    print(f"📂 Đang mở file kho: {FILE_CAN_DOC} ...")
    
    if not os.path.exists(FILE_CAN_DOC):
        print(f"❌ Ối đại ca ơi, chưa có file '{FILE_CAN_DOC}'. Đại ca chạy code tạo data ở bước trước chưa?")
        return

    try:
        # Đọc file Parquet
        df = pd.read_parquet(FILE_CAN_DOC)
        total_rows = len(df)
        print(f"✅ Đã load xong! Tổng kho có: {total_rows} dòng dữ liệu.")

        # Lấy 1000 dòng đầu tiên
        so_luong_lay = min(1000, total_rows)
        df_sample = df.head(so_luong_lay)

        print(f"✂️ Đang cắt {so_luong_lay} dòng đầu tiên để xuất Excel...")
        
        # Xuất ra Excel (bỏ cột index số thứ tự đi cho đỡ rối)
        df_sample.to_excel(FILE_XUAT_RA, index=False)
        
        print(f"\n🎉 XONG HÀNG! Đại ca mở file này lên thẩm định nhé:")
        print(f"👉 {os.path.abspath(FILE_XUAT_RA)}")
        
        # --- IN THỬ VÀI DÒNG RA MÀN HÌNH LUÔN CHO NÓNG ---
        print("\n--- 👀 TRÍCH ĐOẠN 5 DÒNG ĐẦU ---")
        # Chỉ in vài cột quan trọng để check nhanh
        cols_to_show = ['Date', 'Weekday', 'Moon_Phase', 'Jupiter_Deg', 'Res_01', 'Res_27']
        # Lọc những cột nào thực sự tồn tại trong file (đề phòng file thiếu cột)
        valid_cols = [c for c in cols_to_show if c in df.columns]
        print(df_sample[valid_cols].head().to_string())

    except Exception as e:
        print(f"❌ Lỗi toang rồi đại ca ơi: {e}")

if __name__ == "__main__":
    export_sample_data()