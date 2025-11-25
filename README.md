# 🔮 THẦN CƠ DIỆU TOÁN - AI SOI CẦU ULTIMATE 🔮

![Build Status](https://img.shields.io/badge/Build-Nhân_Phẩm_Cao-brightgreen)
![AI Technology](https://img.shields.io/badge/Tech-Deep_Learning_%2B_Tâm_Linh-blueviolet)
![Money](https://img.shields.io/badge/Goal-Về_Bờ_An_Toàn-gold)

> *"Khoa học là đỉnh cao của tri thức, nhưng Lô Đề là vực thẳm của sự giàu sang."*

## 📜 Giới thiệu (Introduction)

Chào mừng các đồng đạo đến với dự án **Ultimate Astro Lottery Predictor**. Đây không phải là một tool random số thông thường. Đây là sự kết tinh giữa **Deep Learning (Transformer)** tối tân và **Chiêm tinh học (Astrology)** cổ đại.

Khi GPU NVIDIA kết hợp với Sao Thủy nghịch hành, chúng ta có công cụ này.
Biến động tài sản của mô hình tốt nhất với tỷ lê ăn 1:3
<img width="1203" height="603" alt="image" src="https://github.com/user-attachments/assets/0a473176-6df0-4c04-ae7c-6a0a9793a198" />


### 🎯 Mục tiêu dự án
- Giúp anh em xa bờ tìm đường về đất liền.
- Chứng minh rằng Thiên văn học có tác dụng thực tế chứ không chỉ để tán gái (xem cung hoàng đạo).
- Đốt cháy GPU để tìm ra con số của Chúa (hoặc của Chủ lô).

## 🚀 Tính năng nổi bật (Features)

- **🤖 Model AI "Siêu To Khổng Lồ":** Sử dụng kiến trúc `UltimateAstroModel` với cơ chế **Attention** (để soi kỹ hơn cả người yêu cũ soi facebook).
- **🌌 Dữ liệu Thiên văn:** Tính toán vị trí Mặt Trời, Mặt Trăng, Sao Hỏa, Sao Kim... chuẩn đến từng độ tại giờ quay số (18:30).
- **🕸️ Tự động Cào (Auto-Scraping):** Bot tự động chạy, tự đi chôm kết quả xổ số hàng ngày để cập nhật database. Không cần nhập tay (vì tay còn bận đếm tiền... hoặc gạt nước mắt).
- **⏰ Báo thức tự động:** Đúng 16:30 chiều hàng ngày, Bot tự dậy, vươn vai, tính toán và nhả ra 5 con số đẹp nhất.

## 🛠️ Cài đặt (Installation)

Dành cho các dân chơi biết code (hoặc biết copy-paste). Làm theo 3 bước chuẩn chỉ sau:

### Bước 1: Clone cái kho này về
Mở Terminal (hoặc CMD) lên và gõ lệnh triệu hồi:

```bash
git clonet
cd 
```

### Bước 2: Cài đặt các thư viện (Phép bổ trợ)
Cài đặt các gói cần thiết để bot có thể thông thiên văn, tường địa lý:

```bash
pip install torch pandas numpy ephem schedule requests beautifulsoup4 pyarrow tqdm
```
*(Lưu ý: Cần cài `torch` bản có GPU nếu muốn train nhanh như cách người yêu cũ trở mặt).*

### Bước 3: Chuẩn bị dữ liệu
- Đảm bảo file `du_lieu_chiem_tinh_chuan_gio.parquet` đã nằm trong folder dự án.
- Nếu chưa có thì cứ kệ nó, chạy bot vài ngày nó tự đi cào về (hơi lâu tí nhưng free).

## 🎮 Hướng dẫn sử dụng (Usage)

### 1. Chạy Bot dự đoán hàng ngày
Đây là chế độ "Ăn sẵn", chỉ việc chạy và chờ số:

```bash
python run_bot_daily.py
```
*Bot sẽ treo máy ở đó, đúng 16:30 chiều nó sẽ hiện lên 5 con số định mệnh.*

### 2. Train lại Model (Dành cho đại ca thích vọc vạch)
Nếu cảm thấy nhân phẩm Model cũ đã hết, hãy train lại một em mới "ngon" hơn:

```bash
python train_ultimate.py
```
*Lưu ý: Quá trình train có thể tốn vài giờ hoặc cả ngày tùy vào độ đẹp trai của GPU.*

## 🧠 Cơ chế hoạt động (How it works)

Tôi cũng không chắc lắm, nhưng về cơ bản là:
1. **Input:** Lấy lịch sử 90 ngày xổ số gần nhất + Vị trí các chòm sao trên trời hôm nay.
2. **Hidden Layers:** Cho qua một đống ma trận, hàm `GatedResidualNetwork`, `MultiheadAttention` (nghe cho nguy hiểm).
3. **Output:** 5 số có xác suất về cao nhất (theo ý kiến chủ quan của AI).

## ⚠️ Tuyên bố miễn trừ trách nhiệm (Disclaimer)

🚫 **ĐỌC KỸ TRƯỚC KHI SỬ DỤNG:**

1. **Code chỉ mang tính chất tham khảo, nghiên cứu khoa học.**
2. Tác giả **KHÔNG** chịu trách nhiệm nếu bạn bán nhà, bán xe, hay phải đi trốn nợ vì tin theo con AI này.
3. Thắng là do bạn tài năng, thua là do... máy lỗi (hoặc do hôm đó sao xấu).
4. **Cờ bạc là bác thằng bần**, code cho vui thôi, đừng all-in nhé đại ca!

## 🤝 Đóng góp (Contribution)

Nếu anh em có cao kiến gì để cải thiện độ chính xác (ví dụ: thêm dữ liệu thời tiết, phong thủy, hướng gió, độ ẩm không khí...), hãy mạnh dạn **Pull Request**.

Nếu trúng lớn, đừng quên quay lại thả cho cái **Star ⭐** nhé!

---
*"Code bằng đam mê, debug bằng nước mắt."* - **Đại Ca Đẹp Trai**
